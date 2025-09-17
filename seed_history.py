# seed_history.py — importa CSVs históricos para .uber_history/history.sqlite
import os, io, csv, zipfile, hashlib, sqlite3, re
from datetime import datetime
import pandas as pd

HIST_DB_PATH = os.environ.get("HIST_DB_PATH", ".uber_history/history.sqlite")
HIST_TABLE   = os.environ.get("HIST_TABLE", "uber_trips")
INPUT_DIR    = os.environ.get("INPUT_DIR", "history_csv")  # pasta com CSV/ZIP

EXPECTED_COLUMNS = {
    "first_name": ["Nome"],
    "last_name": ["Sobrenome"],
    "trip_id": ["ID da viagem/Uber Eats", "ID da viagem"],
    "tx_type": ["Tipo de transação"],
    "status": ["Status da transação"],
    "employee_name": ["Nome"],
    "employee_email": ["E-mail"],
    "vehicle_type": ["Serviço"],
    "program": ["Programa"],
    "employee_id": ["ID do funcionário"],
    "req_date": ["Data da solicitação (local)"],
    "req_time": ["Hora da solicitação (local)"],
    "end_date": ["Data de chegada (local)"],
    "end_time": ["Hora de chegada (local)"],
    "utc_datetime": ["Registro de data e hora da transação (UTC)"],
    "pickup_addr": ["Endereço de partida"],
    "dropoff_addr": ["Endereço de destino"],
    "city": ["Cidade"],
    "distance_mi": ["Distância (mi)"],
    "expense_code": ["Código da despesa"],
    "amount_brl": ["Valor da transação: BRL", "Valor total: BRL"],
    "currency_local": ["Código da moeda local"],
}

def find_col(df, key):
    low = {c.lower().strip(): c for c in df.columns}
    for cand in EXPECTED_COLUMNS.get(key, []):
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def parse_money_cell(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s or s == "--":
        return None
    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1].strip()
    s = (s.replace("\xa0"," ").replace("R$","").replace("BRL","").replace("USD","")
           .replace("$","").replace("€","").replace(" ","").strip())
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot != -1 and last_comma != -1:
        s = s.replace(",", "") if last_dot > last_comma else s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(".", "").replace(",", ".") if last_comma != -1 else s.replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None

def money_series(series):
    return pd.to_numeric(series.apply(parse_money_cell), errors="coerce")

def combine_local_date_time(date_col, time_col):
    if date_col is None:
        return pd.NaT
    if time_col is None:
        return pd.to_datetime(date_col, errors="coerce")
    s = date_col.fillna("").astype(str).str.strip() + " " + time_col.fillna("").astype(str).str.strip()
    s = s.str.replace(r"\s*(AM|PM)$", r" \1", regex=True)
    return pd.to_datetime(s, errors="coerce")

def normalize_uber(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame()
    keys = ["first_name","last_name","trip_id","tx_type","status","employee_name","employee_email",
            "vehicle_type","program","employee_id","pickup_addr","dropoff_addr","city",
            "distance_mi","expense_code","amount_brl","currency_local"]
    for k in keys:
        c = find_col(df, k)
        out[k] = df[c] if (c is not None and c in df.columns) else None

    fn = out.get("first_name", pd.Series([""]*len(out))).fillna("").astype(str).str.strip()
    ln = out.get("last_name",  pd.Series([""]*len(out))).fillna("").astype(str).str.strip()
    full = (fn + " " + ln).str.strip()
    out["employee_full"] = full.mask(full.eq(""), out.get("employee_name"))

    def _dept(x):
        if not isinstance(x, str): return None
        s = x.strip()
        for sep in [" - ", " | ", " — ", " – "]:
            if sep in s: return s.split(sep,1)[0].strip()
        return s or None
    out["department"] = out["employee_id"].apply(_dept)

    rd, rt = find_col(df,"req_date"), find_col(df,"req_time")
    ed, et = find_col(df,"end_date"), find_col(df,"end_time")
    utc = find_col(df,"utc_datetime")

    out["request_dt"] = combine_local_date_time(df[rd] if rd else None, df[rt] if rt else None)
    out["end_dt"]     = combine_local_date_time(df[ed] if ed else None, df[et] if et else None)
    if out["request_dt"].isna().all() and utc in df.columns:
        out["request_dt"] = pd.to_datetime(df[utc], errors="coerce")

    out["amount"]   = money_series(df[find_col(df,"amount_brl")]) if find_col(df,"amount_brl") else pd.NA
    out["currency"] = (df[find_col(df,"currency_local")] if find_col(df,"currency_local") else "BRL")
    dist_mi = pd.to_numeric(out.get("distance_mi", pd.Series([None]*len(out))).astype(str).str.replace(",", "."), errors="coerce")
    out["distance_km"] = dist_mi * 1.60934

    if "trip_id" not in out or out["trip_id"].isna().all():
        def make_id(row):
            base = f"{row.get('employee_email','')}|{row.get('request_dt')}|{row.get('pickup_addr','')}"
            return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
        out["trip_id"] = out.apply(make_id, axis=1)

    out["trip_dt"] = pd.to_datetime(out["request_dt"], errors="coerce")

    out["source_file"] = source_file
    # request_hour (inteiro) — útil para gráfico de horas
    out["request_hour"] = out["trip_dt"].dt.hour
    # colunas finais no padrão do app
    cols = ["first_name","last_name","trip_id","tx_type","status","employee_name","employee_email",
            "vehicle_type","program","employee_id","pickup_addr","dropoff_addr","city",
            "distance_km","expense_code","amount","currency","trip_dt","department","source_file","request_hour"]
    return out[cols]

def read_path_to_dfs(path: str):
    out = []
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    with z.open(n) as f:
                        buf = f.read()
                    out.extend(read_bytes_to_dfs(buf))
    else:
        with open(path, "rb") as f:
            buf = f.read()
        out.extend(read_bytes_to_dfs(buf))
    return out

def read_bytes_to_dfs(buf: bytes):
    def _read_text(text: str):
        lines = text.splitlines()
        start = 0
        for i, ln in enumerate(lines[:80]):
            if ("Tipo de transação" in ln) or ("ID da viagem" in ln) or ("ID da viagem/Uber Eats" in ln):
                start = i; break
        try:
            sep = csv.Sniffer().sniff("\n".join(lines[start:start+30]), delimiters=[",",";","\t","|"]).delimiter
        except Exception:
            sep = None
        return pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=sep, engine="python", encoding="utf-8", on_bad_lines="skip", dtype=str)
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            yield _read_text(buf.decode(enc)); return
        except Exception:
            pass
    yield _read_text(buf.decode("utf-8", errors="replace"))

def ensure_db():
    os.makedirs(os.path.dirname(HIST_DB_PATH), exist_ok=True)
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {HIST_TABLE} (
            trip_id TEXT PRIMARY KEY,
            first_name TEXT, last_name TEXT,
            tx_type TEXT, status TEXT,
            employee_name TEXT, employee_email TEXT,
            vehicle_type TEXT, program TEXT, employee_id TEXT,
            pickup_addr TEXT, dropoff_addr TEXT, city TEXT,
            distance_km REAL, expense_code TEXT,
            amount REAL, currency TEXT,
            trip_dt TEXT, department TEXT, source_file TEXT,
            request_hour INTEGER
        );
    """)
    # índice por data p/ filtros rápidos
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_trip_dt ON {HIST_TABLE}(trip_dt);")
    con.commit(); con.close()

def upsert_rows(df: pd.DataFrame):
    # Normaliza tipos para o SQLite (sem Pandas Timestamp/NA)
    d = df.copy()

    # trip_dt -> string "YYYY-MM-DD HH:MM:SS"
    d["trip_dt"] = pd.to_datetime(d["trip_dt"], errors="coerce")
    d["trip_dt"] = d["trip_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # amount/ distance_km -> float
    for col in ["amount", "distance_km"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)

    # request_hour -> int (ou None)
    if "request_hour" in d.columns:
        d["request_hour"] = pd.to_numeric(d["request_hour"], errors="coerce")
        d["request_hour"] = d["request_hour"].where(d["request_hour"].notna(), None)
        d["request_hour"] = d["request_hour"].astype(object)

    # campos texto: converte NaN -> None para o sqlite
    text_cols = [
        "trip_id","first_name","last_name","tx_type","status","employee_name",
        "employee_email","vehicle_type","program","employee_id","pickup_addr",
        "dropoff_addr","city","expense_code","currency","department","source_file"
    ]
    for c in text_cols:
        if c in d.columns:
            d[c] = d[c].astype(object).where(d[c].notna(), None)

    rows = d.to_dict("records")

    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.executemany(f"""
        INSERT INTO {HIST_TABLE} (
            trip_id, first_name, last_name, tx_type, status, employee_name, employee_email,
            vehicle_type, program, employee_id, pickup_addr, dropoff_addr, city, distance_km,
            expense_code, amount, currency, trip_dt, department, source_file, request_hour
        ) VALUES (
            :trip_id, :first_name, :last_name, :tx_type, :status, :employee_name, :employee_email,
            :vehicle_type, :program, :employee_id, :pickup_addr, :dropoff_addr, :city, :distance_km,
            :expense_code, :amount, :currency, :trip_dt, :department, :source_file, :request_hour
        )
        ON CONFLICT(trip_id) DO UPDATE SET
            tx_type=excluded.tx_type, status=excluded.status,
            vehicle_type=excluded.vehicle_type, program=excluded.program, employee_id=excluded.employee_id,
            pickup_addr=excluded.pickup_addr, dropoff_addr=excluded.dropoff_addr, city=excluded.city,
            distance_km=excluded.distance_km, expense_code=excluded.expense_code,
            amount=excluded.amount, currency=excluded.currency,
            trip_dt=excluded.trip_dt, department=excluded.department, source_file=excluded.source_file,
            request_hour=excluded.request_hour
    """, rows)
    con.commit()
    con.close()


def main():
    ensure_db()
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".csv",".zip"))]
    if not files:
        print(f"Nenhum CSV/ZIP encontrado em {INPUT_DIR}")
        return
    total_rows = 0
    for path in files:
        print("Lendo:", path)
        for df_raw in read_path_to_dfs(path):
            if df_raw is None or df_raw.empty: continue
            df_norm = normalize_uber(df_raw, os.path.basename(path))
            df_norm = df_norm[df_norm["trip_dt"].notna()]  # garante data válida
            if df_norm.empty: continue
            upsert_rows(df_norm)
            total_rows += len(df_norm)
            print(f"  ↳ {len(df_norm)} linhas")
    print(f"Concluído. Linhas inseridas/atualizadas: {total_rows}")
    print(f"Banco: {HIST_DB_PATH} | Tabela: {HIST_TABLE}")

if __name__ == "__main__":
    main()
