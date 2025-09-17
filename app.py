# app.py — Uber Business: Relatório + Admin (Upload/Parametrização) + Auditoria mensal
from __future__ import annotations

import io
import os
import re
import csv
import hmac
import sqlite3
import zipfile
import hashlib
import urllib.parse
from datetime import datetime, date
from typing import List, Optional, Iterable, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

APP_TITLE = "Uber Business"
SUBTITLE = "Visão geral de viagens, parametrização e auditoria"

# ===================== CONFIG BANCO =====================
HIST_DB_PATH = os.getenv("HIST_DB_PATH", st.secrets.get("HIST_DB_PATH", ".uber_history/history.sqlite"))
HIST_TABLE = st.secrets.get("HIST_TABLE", "uber_trips")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", None)  # opcional

SUPERVISOR_PW_SALT = st.secrets.get("SUPERVISOR_PW_SALT", "orbis-supervisor-salt")  # SALT legado (compat)

os.makedirs(os.path.dirname(HIST_DB_PATH), exist_ok=True)

# ---------------------- Utilidades de valores ----------------------
CURRENCY_SYMBOL = {"BRL": "R$", "USD": "$", "EUR": "€", "R$": "R$", "$": "$", "€": "€"}

def money_fmt(v: float, currency: str = "R$") -> str:
    symbol = CURRENCY_SYMBOL.get(str(currency).strip().upper(), currency or "R$")
    try:
        f = float(v or 0)
    except Exception:
        f = 0.0
    s = f"{symbol} {f:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def _strip_currency_tokens(s: str) -> str:
    return (
        s.replace("\xa0", " ")
        .replace("R$", "").replace("BRL", "").replace("USD", "").replace("$", "").replace("€", "")
        .replace(" ", "").strip()
    )

def parse_money_cell(x) -> Optional[float]:
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
    if neg:
        s = s[1:-1].strip()
    # trata R$- 123,45
    s = s.replace("R$-", "-").replace("R$ -", "-")
    s = _strip_currency_tokens(s)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            v = float(s)
            return -v if neg else v
        except Exception:
            return None
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot == -1 and last_comma == -1:
        s_clean = re.sub(r"[^\d\-]", "", s)
        try:
            v = float(s_clean)
            return -v if neg else v
        except Exception:
            return None
    if last_dot != -1 and last_comma != -1:
        s = s.replace(",", "") if last_dot > last_comma else s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(".", "").replace(",", ".") if last_comma != -1 else s.replace(",", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None

def money_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.apply(parse_money_cell), errors="coerce")

def combine_local_date_time(date_col: Optional[pd.Series], time_col: Optional[pd.Series]) -> pd.Series:
    """Interpreta data e hora, tentando autodetectar dayfirst quando necessário."""
    if date_col is None:
        return pd.NaT
    d = date_col.fillna("").astype(str).str.strip()
    t = (time_col.fillna("").astype(str).str.strip()) if time_col is not None else ""
    s = d + " " + t
    s = s.str.replace(r"\s*(AM|PM)$", r" \1", regex=True)
    dt = pd.to_datetime(s, errors="coerce")  # autodetect
    if dt.isna().mean() > 0.6:
        # fallback: dayfirst
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt

def parse_ampm_hour_any(s) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    # mantém apenas dígitos, :, e AM/PM
    t = re.sub(r"[^0-9APMapm: ]", "", s).strip().upper().replace(" ", "")
    t = re.sub(r"^00:", "12:", t)  # 00:00AM/PM -> 12:00...
    for fmt in ("%I:%M%p", "%I%p", "%H:%M", "%H"):
        try:
            return pd.to_datetime(t, format=fmt, errors="raise").hour
        except Exception:
            continue
    try:
        # último recurso: parsing livre
        return pd.to_datetime(s, errors="raise").hour
    except Exception:
        return None

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# NOVO: extração robusta do Departamento a partir de "ID do funcionário"
def _dept_from_employee_id(x: str) -> Optional[str]:
    """
    Extrai o departamento a partir do campo 'ID do funcionário'.
    Regras:
      - Se houver separadores (" - ", " | ", " — ", " – "), usa o trecho à esquerda.
      - Se NÃO houver separador, retorna o texto limpo (se não for vazio).
      - Entende 'DEP: Nome' ou 'Departamento: Nome'.
    """
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None

    m = re.match(r"^\s*(?:dep(?:artamento)?\s*[:\-])\s*(.+)$", s, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return val or None

    for sep in (" - ", " | ", " — ", " – "):
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            return left or None

    # Sem separador → usa o valor inteiro
    return s
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ---------------------- Leitura CSV (Uber) ----------------------
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

def find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    candidates = EXPECTED_COLUMNS.get(key, [])
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def _read_text_to_df(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines[:80]):
        if ("Tipo de transação" in ln) or ("ID da viagem" in ln) or ("ID da viagem/Uber Eats" in ln):
            start = i
            break
    try:
        sep = csv.Sniffer().sniff("\n".join(lines[start:start + 30]), delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        sep = None
    df = pd.read_csv(
        io.StringIO("\n".join(lines[start:])),
        sep=sep, engine="python", encoding="utf-8",
        on_bad_lines="skip", dtype=str
    )
    df = df.replace({"--": None, "—": None})
    return df

def read_bytes_to_df_list(buf: bytes) -> List[pd.DataFrame]:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return [_read_text_to_df(buf.decode(enc))]
        except Exception:
            pass
    return [_read_text_to_df(buf.decode("utf-8", errors="replace"))]

def read_any_csv(uploaded) -> pd.DataFrame:
    """Suporta CSV puro e ZIP contendo múltiplos CSVs (marca nome.zip::arquivo_interno.csv em source_file)."""
    name_lower = uploaded.name.lower()
    raw = uploaded.read()
    dfs: list[pd.DataFrame] = []
    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
            for member in z.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with z.open(member) as f:
                    buf = f.read()
                dlist = read_bytes_to_df_list(buf)
                for d in dlist:
                    # preserva nome do arquivo interno
                    d["source_file"] = f"{uploaded.name}::{os.path.basename(member)}"
                dfs.extend(dlist)
    else:
        dlist = read_bytes_to_df_list(raw)
        for d in dlist:
            d["source_file"] = uploaded.name
        dfs.extend(dlist)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)

def normalize_uber(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    out = pd.DataFrame()
    basic_keys = [
        "first_name","last_name","trip_id","tx_type","status","employee_name","employee_email",
        "vehicle_type","program","employee_id","pickup_addr","dropoff_addr","city",
        "distance_mi","expense_code","amount_brl","currency_local",
    ]
    for k in basic_keys:
        c = find_col(df, k)
        out[k] = df[c] if (c is not None and c in df.columns) else None

    fn = out.get("first_name", pd.Series([""] * len(out))).fillna("").astype(str).str.strip()
    ln = out.get("last_name", pd.Series([""] * len(out))).fillna("").astype(str).str.strip()
    full = (fn + " " + ln).str.strip()
    out["employee_full"] = full.mask(full.eq(""), out.get("employee_name"))

    out["department"] = out["employee_id"].apply(_dept_from_employee_id)

    req_d = find_col(df, "req_date"); req_t = find_col(df, "req_time")
    end_d = find_col(df, "end_date"); end_t = find_col(df, "end_time")
    utc_dt = find_col(df, "utc_datetime")

    out["request_dt"] = combine_local_date_time(df[req_d] if req_d else None, df[req_t] if req_t else None)
    out["end_dt"] = combine_local_date_time(df[end_d] if end_d else None, df[end_t] if end_t else None)
    if out["request_dt"].isna().all() and utc_dt in df.columns:
        out["request_dt"] = pd.to_datetime(df[utc_dt], errors="coerce")  # mantido em UTC

    out["request_time_str"] = df[req_t] if req_t in df.columns else None

    amount_col = find_col(df, "amount_brl")
    out["amount"] = money_series(df[amount_col]) if amount_col else pd.NA
    out["currency"] = out.get("currency_local", pd.Series(["BRL"] * len(out))).fillna("BRL")

    dist_mi = pd.to_numeric(out.get("distance_mi", pd.Series([None]*len(out))).astype(str).str.replace(",", "."), errors="coerce")
    out["distance_km"] = dist_mi * 1.60934

    # mantém a origem do arquivo (se o df já não trouxe ao ler .zip)
    if "source_file" in df.columns:
        out["source_file"] = df["source_file"]
    else:
        out["source_file"] = file_name

    if "trip_id" not in out or out["trip_id"].isna().all():
        def make_id(row) -> str:
            base = f"{row.get('employee_email','')}|{row.get('request_dt')}|{row.get('pickup_addr','')}"
            return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
        out["trip_id"] = out.apply(make_id, axis=1)

    out["trip_dt"] = pd.to_datetime(out["request_dt"], errors="coerce")
    hour_from_dt = out["trip_dt"].dt.hour
    hour_from_str = out["request_time_str"].apply(parse_ampm_hour_any) if "request_time_str" in out.columns else None
    out["request_hour"] = hour_from_dt.fillna(hour_from_str)
    return out

# ======================= BANCO (SQLite) =======================
def _ensure_supervisors_schema(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_supervisors (
            supervisor   TEXT PRIMARY KEY,
            pass_hash    BLOB NOT NULL,
            salt         BLOB,
            iters        INTEGER,
            updated_at   TEXT
        );
    """)
    # garante colunas em bases antigas
    cur.execute("PRAGMA table_info(audit_supervisors);")
    cols = {r[1] for r in cur.fetchall()}
    if "salt" not in cols:
        cur.execute("ALTER TABLE audit_supervisors ADD COLUMN salt BLOB;")
    if "iters" not in cols:
        cur.execute("ALTER TABLE audit_supervisors ADD COLUMN iters INTEGER;")

def ensure_db():
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    # performance pragmas
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

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
    # Índices úteis
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_trip_dt ON {HIST_TABLE}(trip_dt);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_prog ON {HIST_TABLE}(program);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_dept ON {HIST_TABLE}(department);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_city ON {HIST_TABLE}(city);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_email ON {HIST_TABLE}(employee_email);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{HIST_TABLE}_hour ON {HIST_TABLE}(request_hour);")

    # parametrização: colaborador -> supervisor
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_assignments (
            employee_email TEXT PRIMARY KEY,
            employee_full  TEXT,
            supervisor     TEXT,
            updated_at     TEXT
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_assign_sup ON audit_assignments(supervisor);")

    # auditoria: decisão por viagem
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_reviews (
            trip_id     TEXT PRIMARY KEY,
            status      TEXT,      -- APPROVED | REJECTED
            reason      TEXT,
            reviewer    TEXT,
            reviewed_at TEXT
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON audit_reviews(status);")

    # senhas de supervisores (PBKDF2 + compat legado)
    _ensure_supervisors_schema(con)

    con.commit()
    con.close()

def _hash_pw_legacy(pw: str) -> str:
    # compat com versões antigas (salt fixo + sha256 hex)
    return hashlib.sha256((SUPERVISOR_PW_SALT + (pw or "")).encode("utf-8")).hexdigest()

def _hash_pw_pbkdf2(password: str, salt: bytes, iters: int = 120_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, iters)

def insert_or_ignore(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    d = df.copy()
    d["trip_dt"] = pd.to_datetime(d["trip_dt"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["amount", "distance_km"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    if "request_hour" in d.columns:
        d["request_hour"] = pd.to_numeric(d["request_hour"], errors="coerce")
        d["request_hour"] = d["request_hour"].where(d["request_hour"].notna(), None).astype(object)
    text_cols = [
        "trip_id","first_name","last_name","tx_type","status","employee_name",
        "employee_email","vehicle_type","program","employee_id","pickup_addr",
        "dropoff_addr","city","expense_code","currency","department","source_file"
    ]
    for c in text_cols:
        if c in d.columns:
            d[c] = d[c].astype(object).where(d[c].notna(), None)

    cols = [
        "trip_id","first_name","last_name","tx_type","status","employee_name","employee_email",
        "vehicle_type","program","employee_id","pickup_addr","dropoff_addr","city","distance_km",
        "expense_code","amount","currency","trip_dt","department","source_file","request_hour"
    ]
    for c in cols:
        if c not in d.columns:
            d[c] = None

    rows = list(map(tuple, d[cols].to_numpy()))

    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("BEGIN;")
    before = con.total_changes
    cur.executemany(f"""
        INSERT OR IGNORE INTO {HIST_TABLE} (
            trip_id, first_name, last_name, tx_type, status, employee_name, employee_email,
            vehicle_type, program, employee_id, pickup_addr, dropoff_addr, city, distance_km,
            expense_code, amount, currency, trip_dt, department, source_file, request_hour
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    inserted = con.total_changes - before
    con.close()
    return inserted

def delete_month(year: int, month: int) -> int:
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    start = date(year, month, 1)
    end = date(year + (month // 12), ((month % 12) + 1), 1)
    cur.execute(f"""
        DELETE FROM {HIST_TABLE}
        WHERE date(trip_dt) >= date(?) AND date(trip_dt) < date(?)
    """, (start.isoformat(), end.isoformat()))
    deleted = cur.rowcount if hasattr(cur, "rowcount") else 0
    con.commit()
    con.close()
    return deleted

@st.cache_data(ttl=120)
def load_all_from_db() -> pd.DataFrame:
    if not os.path.exists(HIST_DB_PATH):
        return pd.DataFrame()
    con = sqlite3.connect(HIST_DB_PATH)
    try:
        df = pd.read_sql_query(f"""
            SELECT first_name,last_name,trip_id,tx_type,status,employee_name,employee_email,
                   vehicle_type,program,employee_id,pickup_addr,dropoff_addr,city,distance_km,
                   expense_code,amount,currency,trip_dt,department,source_file,request_hour
            FROM {HIST_TABLE}
            WHERE trip_dt IS NOT NULL
        """, con)
        df["trip_dt"] = pd.to_datetime(df["trip_dt"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        if "employee_full" not in df.columns or df["employee_full"].isna().all():
            fn = df.get("first_name", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            ln = df.get("last_name",  pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            emp = (fn + " " + ln).str.strip()
            df["employee_full"] = emp.mask(emp.eq(""), df.get("employee_name"))
        return df
    finally:
        con.close()

def _month_bounds(year: int, month: int):
    start = date(year, month, 1)
    end = date(year + (month // 12), ((month % 12) + 1), 1)
    return start, end

def list_month_sources(year: int, month: int) -> pd.DataFrame:
    if not os.path.exists(HIST_DB_PATH):
        return pd.DataFrame()
    start, end = _month_bounds(year, month)
    con = sqlite3.connect(HIST_DB_PATH)
    try:
        df = pd.read_sql_query(f"""
            SELECT
                COALESCE(source_file, '—') AS source_file,
                COUNT(*) AS viagens,
                MIN(trip_dt) AS dt_min,
                MAX(trip_dt) AS dt_max
            FROM {HIST_TABLE}
            WHERE trip_dt IS NOT NULL
              AND date(trip_dt) >= date(?)
              AND date(trip_dt) <  date(?)
            GROUP BY source_file
            ORDER BY viagens DESC, source_file ASC
        """, con, params=(start.isoformat(), end.isoformat()))
        df["dt_min"] = pd.to_datetime(df["dt_min"], errors="coerce")
        df["dt_max"] = pd.to_datetime(df["dt_max"], errors="coerce")
        return df
    finally:
        con.close()

# ====== Parametrização (assignments) & Auditoria (reviews) ======
@st.cache_data(ttl=120)
def list_assignments() -> pd.DataFrame:
    con = sqlite3.connect(HIST_DB_PATH)
    try:
        return pd.read_sql_query("SELECT employee_email, employee_full, supervisor, updated_at FROM audit_assignments ORDER BY employee_full", con)
    finally:
        con.close()

def upsert_assignment(employee_email: str, employee_full: str, supervisor: str):
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO audit_assignments (employee_email, employee_full, supervisor, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(employee_email) DO UPDATE SET
            employee_full=excluded.employee_full,
            supervisor=excluded.supervisor,
            updated_at=excluded.updated_at
    """, (employee_email.strip().lower(), (employee_full or "").strip(), supervisor.strip(), datetime.utcnow().isoformat(timespec="seconds")))
    con.commit()
    con.close()
    st.cache_data.clear()

def delete_assignment(employee_email: str):
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM audit_assignments WHERE employee_email = ?", (employee_email.strip().lower(),))
    con.commit()
    con.close()
    st.cache_data.clear()

def list_supervisors() -> List[str]:
    df = list_assignments()
    sups = sorted([s for s in df["supervisor"].dropna().unique().tolist()])
    return sups

def set_supervisor_password(supervisor: str, password: str):
    if not (supervisor and password):
        raise ValueError("Supervisor e senha são obrigatórios.")
    con = sqlite3.connect(HIST_DB_PATH)
    _ensure_supervisors_schema(con)
    cur = con.cursor()
    salt = os.urandom(16)
    iters = 120_000
    pass_hash = _hash_pw_pbkdf2(password, salt, iters)
    cur.execute("""
        INSERT INTO audit_supervisors (supervisor, pass_hash, salt, iters, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(supervisor) DO UPDATE SET
            pass_hash=excluded.pass_hash,
            salt=excluded.salt,
            iters=excluded.iters,
            updated_at=excluded.updated_at
    """, (supervisor.strip(), pass_hash, salt, iters, datetime.utcnow().isoformat(timespec="seconds")))
    con.commit()
    con.close()
    st.cache_data.clear()

def delete_supervisor_password(supervisor: str):
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM audit_supervisors WHERE supervisor = ?", (supervisor.strip(),))
    con.commit()
    con.close()
    st.cache_data.clear()

def _check_supervisor_password(supervisor: str, password: str) -> bool:
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT pass_hash, salt, iters FROM audit_supervisors WHERE supervisor = ?", (supervisor.strip(),))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    pass_hash, salt, iters = row[0], row[1], row[2]
    # compat: se não houver salt/iters (registro legado), valida pelo hash antigo
    if salt is None or iters is None:
        return pass_hash == _hash_pw_legacy(password or "")
    # PBKDF2
    try:
        candidate = _hash_pw_pbkdf2(password, salt, int(iters))
        return hmac.compare_digest(pass_hash, candidate)
    except Exception:
        return False

def get_month_trips_with_assignment(year: int, month: int, supervisor: Optional[str] = None) -> pd.DataFrame:
    """Retorna viagens do mês com coluna 'supervisor' (via assignment) e status de review."""
    start, end = _month_bounds(year, month)
    con = sqlite3.connect(HIST_DB_PATH)
    try:
        q = f"""
        SELECT t.*, a.supervisor,
               r.status AS review_status,
               r.reason AS review_reason,
               r.reviewer AS review_reviewer,
               r.reviewed_at AS review_time
        FROM {HIST_TABLE} t
        LEFT JOIN audit_assignments a ON LOWER(a.employee_email) = LOWER(t.employee_email)
        LEFT JOIN audit_reviews r ON r.trip_id = t.trip_id
        WHERE t.trip_dt IS NOT NULL
          AND date(t.trip_dt) >= date(?)
          AND date(t.trip_dt) <  date(?)
        """
        params = [start.isoformat(), end.isoformat()]
        if supervisor:
            q += " AND a.supervisor = ?"
            params.append(supervisor)
        df = pd.read_sql_query(q, con, params=params)
        df["trip_dt"] = pd.to_datetime(df["trip_dt"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        if "employee_full" not in df.columns or df["employee_full"].isna().all():
            fn = df.get("first_name", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            ln = df.get("last_name",  pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            emp = (fn + " " + ln).str.strip()
            df["employee_full"] = emp.mask(emp.eq(""), df.get("employee_name"))
        return df
    finally:
        con.close()

# ---- Reviews utilitários (para Relatório)
@st.cache_data(ttl=120)
def load_reviews_df() -> pd.DataFrame:
    con = sqlite3.connect(HIST_DB_PATH)
    try:
        df = pd.read_sql_query("SELECT trip_id, status AS review_status, reason AS review_reason, reviewer AS review_reviewer, reviewed_at AS review_time FROM audit_reviews", con)
        if not df.empty:
            df["review_time"] = pd.to_datetime(df["review_time"], errors="coerce")
        return df
    finally:
        con.close()

def enrich_with_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de review às viagens e garante que exista UMA coluna 'supervisor'.
    - Se o DF já tiver 'supervisor', não faz novo merge com assignments.
    - Se surgirem 'supervisor_x'/'supervisor_y', consolida em 'supervisor'.
    """
    j = df.copy()

    # Garantir coluna 'supervisor' sem duplicação
    if "supervisor" not in j.columns:
        ass = list_assignments()[["employee_email", "supervisor"]].copy()
        j = j.merge(ass, on="employee_email", how="left")

    # Consolidar caso existam sufixos
    if "supervisor_x" in j.columns or "supervisor_y" in j.columns:
        j["supervisor"] = j.get("supervisor_x").combine_first(j.get("supervisor_y"))
        j = j.drop(columns=[c for c in ["supervisor_x", "supervisor_y"] if c in j.columns])

    # Reviews
    rev = load_reviews_df()
    if not rev.empty:
        j = j.merge(rev, on="trip_id", how="left")
    else:
        j["review_status"] = None
        j["review_reason"] = None
        j["review_reviewer"] = None
        j["review_time"] = None

    return j


def set_review(trip_id: str, status: str, reason: str, reviewer: str):
    status = status.upper().strip()
    if status not in ("APPROVED", "REJECTED"):
        raise ValueError("status inválido")
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO audit_reviews (trip_id, status, reason, reviewer, reviewed_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(trip_id) DO UPDATE SET
            status=excluded.status,
            reason=excluded.reason,
            reviewer=excluded.reviewer,
            reviewed_at=excluded.reviewed_at
    """, (trip_id, status, reason, reviewer, datetime.utcnow().isoformat(timespec="seconds")))
    con.commit()
    con.close()

# ---------------------- Gráficos ----------------------
def bar_with_labels(df, x, y, title, number_format=",.2f"):
    if df.empty:
        st.info("Sem dados para o gráfico.")
        return
    d = df.sort_values(y, ascending=False)
    base = alt.Chart(d).mark_bar().encode(
        x=alt.X(f"{x}:N", sort="-y", title=None),
        y=alt.Y(f"{y}:Q", title=None),
        tooltip=[x, alt.Tooltip(y, format=number_format)],
    ).properties(height=300, title=title)
    text = alt.Chart(d).mark_text(dy=-5).encode(
        x=alt.X(f"{x}:N", sort="-y"),
        y=alt.Y(f"{y}:Q"),
        text=alt.Text(f"{y}:Q", format=number_format),
    )
    st.altair_chart(base + text, use_container_width=True)

def line_hour_chart(df, hour_col: str, title: str):
    if df.empty or hour_col not in df.columns:
        st.info("Sem dados para o gráfico de horas.")
        return
    d = df.copy()
    d = d[pd.to_numeric(d[hour_col], errors="coerce").notna()]
    d[hour_col] = d[hour_col].astype(int)
    full = pd.DataFrame({"hour": range(24)})
    agg = d.groupby(hour_col).size().reset_index(name="viagens")
    agg = full.merge(agg, left_on="hour", right_on=hour_col, how="left").drop(columns=[hour_col]).fillna(0)
    agg["label"] = agg["hour"].map(lambda h: f"{h:02d}:00")
    line = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X("hour:Q", axis=alt.Axis(values=list(range(24)), labelExpr="format(datum.value, '02') + ':00'"), title=None),
        y=alt.Y("viagens:Q", title="Viagens"),
        tooltip=["label", "viagens"],
    ).properties(height=320, title=title)
    labels = alt.Chart(agg).mark_text(dy=-10).encode(
        x="hour:Q", y="viagens:Q", text=alt.Text("viagens:Q", format=",.0f")
    )
    st.altair_chart(line + labels, use_container_width=True)

def monthly_line_with_labels(df: pd.DataFrame, date_col: str, value_col: str, title: str, y_title: str, fmt=",.2f"):
    """
    Corrigido: eixo horizontal por mês/ano (discreto), ordenado cronologicamente.
    """
    if df.empty:
        st.info("Sem dados para o gráfico.")
        return
    d = df.copy()
    month_idx = pd.to_datetime(d[date_col], errors="coerce").dt.to_period("M")
    d["month_start"] = month_idx.dt.to_timestamp()
    d = d.groupby("month_start", as_index=False)[value_col].sum().sort_values("month_start")
    d["month_label"] = d["month_start"].dt.strftime("%Y-%m")  # ex.: 2025-09
    line = alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("month_label:N", sort=list(d["month_label"]), title=None),
        y=alt.Y(f"{value_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip("month_start:T", title="Mês", format="%b %Y"),
            alt.Tooltip(f"{value_col}:Q", format=fmt, title=y_title),
        ],
    ).properties(height=320, title=title)
    labels = alt.Chart(d).mark_text(dy=-10).encode(
        x=alt.X("month_label:N", sort=list(d["month_label"])),
        y=f"{value_col}:Q",
        text=alt.Text(f"{value_col}:Q", format=fmt)
    )
    st.altair_chart(line + labels, use_container_width=True)

# ---------------------- UI Helpers ----------------------
def month_year_selector(label_prefix: str = "Período") -> Tuple[int, int]:
    cols = st.columns([1, 1, 2])
    with cols[0]:
        year = st.number_input(f"{label_prefix} — Ano", min_value=2018, max_value=2100, value=datetime.now().year, step=1)
    with cols[1]:
        month = st.selectbox(f"{label_prefix} — Mês", list(range(1,13)), index=(datetime.now().month-1))
    with cols[2]:
        st.caption("Selecione o mês/ano de referência.")
    return int(year), int(month)

# ======================= Páginas =======================
def filter_df_by_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    d = df.copy()
    d["trip_dt"] = pd.to_datetime(d["trip_dt"], errors="coerce")
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthBegin(1))
    return d[(d["trip_dt"] >= start) & (d["trip_dt"] < end)]

# ---- Admin: Upload ----
def tela_admin_upload():
    st.subheader("Upload mensal por período")
    st.caption("Envie CSV/ZIP do mês/ano escolhido. **Substituir** apaga e recarrega; **Mesclar** faz upsert por `trip_id`.")

    year, month = month_year_selector("Upload")
    mode = st.radio("Modo de importação", ["Substituir mês (apaga e carrega)", "Mesclar (upsert por ID)"], index=0, horizontal=True)

    with st.container(border=True):
        st.markdown(f"**Arquivos já vinculados a {month:02d}/{year}**")
        df_src = list_month_sources(year, month)
        if df_src.empty:
            st.info("Nenhum arquivo vinculado a este mês ainda.")
        else:
            pretty_table(
                df_src.rename(columns={
                    "source_file": "Arquivo",
                    "viagens": "Viagens",
                    "dt_min": "Primeira viagem",
                    "dt_max": "Última viagem",
                }),
                datetime_cols=("Primeira viagem", "Última viagem")
            )

    up = st.file_uploader(
        "Selecione um ou mais arquivos (CSV ou ZIP contendo CSV) **do mês selecionado**",
        type=["csv", "zip"],
        accept_multiple_files=True
    )

    if up and st.button("Processar e salvar"):
        ensure_db()
        st.cache_data.clear()
        total_linhas = 0
        novas_linhas = 0
        ignoradas_outro_mes = 0

        if mode.startswith("Substituir"):
            deleted = delete_month(year, month)
            st.info(f"Removidas {deleted} viagens existentes de {month:02d}/{year}.")

        for file in up:
            df_raw = read_any_csv(file)
            if df_raw is None or df_raw.empty:
                continue
            df_norm = normalize_uber(df_raw, file.name)
            df_norm = df_norm[df_norm["trip_dt"].notna()]

            # Sinalização de valores atípicos (opcional)
            altos = pd.to_numeric(df_norm["amount"], errors="coerce")
            if pd.notna(altos).any() and (altos > 10000).sum() > 0:
                st.warning(f"O arquivo {file.name} contém {int((altos > 10000).sum())} valor(es) de corrida acima de R$ 10.000 — verifique se não há erro de parsing.")

            before = len(df_norm)
            df_norm = filter_df_by_month(df_norm, year, month)
            ignoradas_outro_mes += (before - len(df_norm))
            if df_norm.empty:
                continue

            inserted = insert_or_ignore(df_norm)
            total_linhas += len(df_norm)
            novas_linhas += inserted

        st.success(
            f"{month:02d}/{year}: {len(up)} arquivo(s), {novas_linhas} novas viagens inseridas de {total_linhas}. "
            f"Ignoradas {ignoradas_outro_mes} linhas fora do mês selecionado."
        )
        # garante atualização de caches e listas após upload
        st.cache_data.clear()

        with st.container(border=True):
            st.markdown(f"**Arquivos vinculados após importação — {month:02d}/{year}**")
            df_src = list_month_sources(year, month)
            if df_src.empty:
                st.info("Nenhum arquivo vinculado a este mês ainda.")
            else:
                pretty_table(
                    df_src.rename(columns={
                        "source_file": "Arquivo",
                        "viagens": "Viagens",
                        "dt_min": "Primeira viagem",
                        "dt_max": "Última viagem",
                    }),
                    datetime_cols=("Primeira viagem", "Última viagem")
                )

# ---- Admin: Parametrização (+ senha de supervisor) ----
def tela_admin_parametrizacao():
    st.subheader("Parametrização — Vínculo Colaborador → Supervisor")
    st.caption("Cadastre os responsáveis pela auditoria de cada colaborador (por e-mail). Também defina a senha do supervisor para autorizar as auditorias.")

    with st.expander("Adicionar / Atualizar vínculo", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            emp_email = st.text_input("E-mail do colaborador").strip().lower()
            emp_name  = st.text_input("Nome do colaborador")
        with c2:
            supervisor = st.text_input("Supervisor (nome ou e-mail)").strip()
        if st.button("Salvar vínculo"):
            if not emp_email or not supervisor:
                st.warning("Informe ao menos e-mail do colaborador e supervisor.")
            else:
                upsert_assignment(emp_email, emp_name, supervisor)
                st.success("Vínculo salvo/atualizado.")

    with st.expander("Remover vínculo"):
        df_ass = list_assignments()
        if df_ass.empty:
            st.info("Não há vínculos cadastrados.")
        else:
            email_opts = [""] + df_ass["employee_email"].tolist()
            sel = st.selectbox("Selecione o colaborador", email_opts, index=0)
            if sel and st.button("Apagar vínculo selecionado"):
                delete_assignment(sel)
                st.success("Vínculo removido.")

    st.markdown("**Vínculos atuais**")
    df_ass = list_assignments()
    if df_ass.empty:
        st.info("Nenhum vínculo cadastrado ainda.")
    else:
        pretty_table(df_ass.rename(columns={
            "employee_email": "Colaborador (e-mail)",
            "employee_full": "Colaborador (nome)",
            "supervisor": "Supervisor",
            "updated_at": "Atualizado em"
        }))

    st.markdown("---")
    st.subheader("Senhas de Supervisores (autorização da auditoria)")
    st.caption("Defina uma senha por **Supervisor**. Na Auditoria, o revisor deverá informar este identificador e senha para registrar Aprovar/Reprovar.")

    with st.expander("Definir/Atualizar senha de um Supervisor", expanded=False):
        sup_for_pw = st.text_input("Supervisor (exatamente como usado nos vínculos)").strip()
        pw1, pw2 = st.columns(2)
        with pw1:
            s1 = st.text_input("Senha do supervisor", type="password")
        with pw2:
            s2 = st.text_input("Confirmar senha", type="password")
        if st.button("Salvar senha do Supervisor"):
            if not sup_for_pw or not s1:
                st.warning("Informe supervisor e senha.")
            elif s1 != s2:
                st.error("As senhas não conferem.")
            else:
                set_supervisor_password(sup_for_pw, s1)
                st.success("Senha do supervisor definida/atualizada.")

    with st.expander("Apagar senha de um Supervisor", expanded=False):
        sup_del = st.text_input("Supervisor (exatamente como usado nos vínculos)", key="sup_del").strip()
        if st.button("Apagar senha"):
            if not sup_del:
                st.warning("Informe o supervisor.")
            else:
                delete_supervisor_password(sup_del)
                st.success("Senha removida para este supervisor.")

# ---- Admin (aba com tabs) ----
def tela_admin():
    if ADMIN_PASSWORD:
        pw = st.text_input("Senha do administrador", type="password")
        if not pw:
            st.stop()
        if pw != ADMIN_PASSWORD:
            st.error("Senha incorreta.")
            st.stop()

    tab1, tab2 = st.tabs(["📤 Upload mensal", "🧭 Parametrização"])
    with tab1:
        tela_admin_upload()
    with tab2:
        tela_admin_parametrizacao()

# ---- Auditoria ----
CARD_CSS = """
<style>
.trip-card{
  padding: 12px 14px; margin: 10px 0; border-radius: 12px;
  background: #ffffff; border:1px solid #EEE; box-shadow: 0 1px 0 rgba(0,0,0,0.02);
}
.trip-card + .trip-card { margin-top:16px; border-top:1px dashed #EEE; padding-top:16px; }
.tc-head{display:flex; align-items:center; justify-content:space-between;}
.tc-title{font-size:15px;}
.tc-value{font-weight:700; font-size:15px;}
.tc-sub{color:#667085; font-size:12px; margin-top:4px; line-height:1.4;}
.badge{
  display:inline-block; padding:3px 8px; font-size:11px; border-radius:999px;
  background:#F2F4F7; color:#1F2937; margin-right:6px; border:1px solid #E5E7EB;
}
.badge.green{ background:#ECFDF5; color:#065F46; border-color:#A7F3D0;}
.badge.red{ background:#FEF2F2; color:#991B1B; border-color:#FECACA;}
</style>
"""

def render_trip_card(r: pd.Series):
    base_color = "#5B8DEF"
    if str(r.get("vehicle_type") or "").lower().startswith("uber black"):
        base_color = "#7B61FF"
    elif str(r.get("vehicle_type") or "").lower().startswith("uberx"):
        base_color = "#2EC4B6"
    elif str(r.get("program") or "").lower().startswith("business"):
        base_color = "#FF9F1C"

    dt_txt = pd.to_datetime(r.get("trip_dt")).strftime('%d/%m/%Y %H:%M') if pd.notna(r.get("trip_dt")) else "—"

    status = r.get("review_status")
    status_badge = ""
    if status == "APPROVED":
        status_badge = "<span class='badge green'>Aprovado</span>"
    elif status == "REJECTED":
        status_badge = "<span class='badge red'>Reprovado</span>"

    st.markdown(
        f"""
        <div class="trip-card" style="border-left:6px solid {base_color};">
          <div class="tc-head">
            <div class="tc-title"><strong>{r.get('employee_full','—')}</strong></div>
            <div class="tc-value">{money_fmt(r.get('amount',0), r.get('currency') or 'BRL')}</div>
          </div>
          <div class="tc-sub">
            <span>Data: {dt_txt}</span> ·
            <span>Cidade: {r.get('city') or '—'}</span> ·
            <span>Programa: {r.get('program') or '—'}</span> ·
            <span>Supervisor: {r.get('supervisor') or '—'}</span> {status_badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def tela_auditoria():
    st.header("Auditoria mensal de viagens")
    st.caption("Selecione o mês/ano de referência e, opcionalmente, um supervisor para filtrar as viagens dos colaboradores sob sua responsabilidade.")

    year, month = month_year_selector("Auditoria")
    sup_opts = ["(Todos)"] + list_supervisors()
    supervisor = st.selectbox("Supervisor", sup_opts, index=0)
    sup_filter = None if supervisor == "(Todos)" else supervisor

    status_opts = ["Todos", "Pendentes", "Aprovados", "Reprovados"]
    status_sel = st.selectbox("Status", status_opts, index=0)

    df = get_month_trips_with_assignment(year, month, sup_filter)
    if df.empty:
        st.info("Não há viagens para o período/seleção.")
        return

    # Filtro por status
    if status_sel != "Todos":
        if status_sel == "Pendentes":
            df = df[df["review_status"].isna()]
        elif status_sel == "Aprovados":
            df = df[df["review_status"] == "APPROVED"]
        elif status_sel == "Reprovados":
            df = df[df["review_status"] == "REJECTED"]

    # KPIs
    total = len(df)
    aprov = int((df["review_status"] == "APPROVED").sum())
    reprov = int((df["review_status"] == "REJECTED").sum())
    pend = total - aprov - reprov
    cobertura = (aprov + reprov) / total * 100 if total else 0.0

    # tempo médio até auditoria
    times = df.dropna(subset=["review_time", "trip_dt"]).copy()
    if not times.empty:
        times["review_time"] = pd.to_datetime(times["review_time"], errors="coerce")
        times["delta_h"] = (times["review_time"] - times["trip_dt"]).dt.total_seconds() / 3600.0
        t_medio_h = float(times["delta_h"].mean())
    else:
        t_medio_h = 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total de viagens", f"{total:,}".replace(",", "."))
    with c2: st.metric("Aprovadas", f"{aprov:,}".replace(",", "."))
    with c3: st.metric("Reprovadas", f"{reprov:,}".replace(",", "."))
    with c4: st.metric("Pendentes", f"{pend:,}".replace(",", "."))
    with c5: st.metric("Cobertura da auditoria", f"{cobertura:,.1f}%")

    if t_medio_h > 0:
        st.caption(f"⏱️ Tempo médio até auditoria: **{t_medio_h:.1f} h**")

    # ===== Listagem paginada =====
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Ordena por data e prepara paginação
    df = df.sort_values("trip_dt", ascending=True).reset_index(drop=True)

    with st.container(border=True):
        lc1, lc2, lc3 = st.columns([1, 1, 2])
        with lc1:
            page_size = st.slider("Quantidade por página", 5, 50, 10, key="aud_page_size")
        with lc2:
            total_pages = max(1, (len(df) + page_size - 1) // page_size)
            page = st.number_input("Página", 1, total_pages, 1, key="aud_page_num")
        with lc3:
            st.caption(f"Mostrando página {page}/{total_pages} — {len(df)} viagens filtradas")

    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end].copy()

    for _, r in page_df.iterrows():
        render_trip_card(r)
        with st.expander("Detalhar / Validar", expanded=False):
            colA, colB = st.columns([1, 1])
            dt_txt = pd.to_datetime(r.get("trip_dt")).strftime('%d/%m/%Y %H:%M') if pd.notna(r.get("trip_dt")) else "—"

            with colA:
                st.write("**Data:**", dt_txt)
                st.write("**Colaborador:**", r.get("employee_full"))
                st.write("**E-mail:**", r.get("employee_email"))
                st.write("**Programa:**", r.get("program") or "—")
                st.write("**Departamento:**", r.get("department") or "—")
                st.write("**Serviço:**", r.get("vehicle_type") or "—")
                st.write("**Cidade:**", r.get("city") or "—")
                st.write("**Código da despesa (motivo):**", r.get("expense_code") or "—")
                st.write("**Supervisor responsável:**", r.get("supervisor") or "—")

            with colB:
                st.write("**Valor:**", money_fmt(r.get("amount", 0), r.get("currency") or "BRL"))
                dist_txt = f"{float(r.get('distance_km')):.2f}" if pd.notna(r.get("distance_km")) else "—"
                st.write("**Distância (km):**", dist_txt)
                st.write("**Origem:**", r.get("pickup_addr") or "—")
                st.write("**Destino:**", r.get("dropoff_addr") or "—")
                st.caption(f"Arquivo: {r.get('source_file')} · ID: {r.get('trip_id')}")

            # ===== Mapa (igual ao Relatório) =====
            origem = (r.get("pickup_addr") or "").strip()
            destino = (r.get("dropoff_addr") or "").strip()
            if origem and destino:
                q = f"https://www.google.com/maps?output=embed&saddr={urllib.parse.quote(origem)}&daddr={urllib.parse.quote(destino)}"
                components.html(
                    f'<iframe src="{q}" width="100%" height="360" style="border:0;" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>',
                    height=380,
                )
                st.markdown(
                    f"[📍 Abrir rota no Google Maps]({ 'https://www.google.com/maps/dir/?api=1&origin=' + urllib.parse.quote(origem) + '&destination=' + urllib.parse.quote(destino) })  ·  "
                    f"[🔎 Pesquisar origem]({ 'https://www.google.com/maps/search/?api=1&query=' + urllib.parse.quote(origem) })  ·  "
                    f"[🔎 Pesquisar destino]({ 'https://www.google.com/maps/search/?api=1&query=' + urllib.parse.quote(destino) })",
                    help="Se o iframe não for exibido pelo seu navegador, use os links."
                )

            # ===== Form de aprovação/reprovação (com senha do supervisor) =====
            with st.form(key=f"review-{r['trip_id']}"):
                cols = st.columns([1, 2, 1])
                with cols[0]:
                    decision = st.radio(
                        "Decisão",
                        ["Aprovar", "Reprovar"],
                        horizontal=True,
                        index=(0 if r.get("review_status") == "APPROVED"
                               else (1 if r.get("review_status") == "REJECTED" else 0))
                    )
                with cols[1]:
                    reason = st.text_input(
                        "Observação (obrigatória se reprovar)",
                        value=(r.get("review_reason") or "")
                    )
                with cols[2]:
                    reviewer = st.text_input(
                        "Revisor (identificador do supervisor)",
                        value=(r.get("review_reviewer") or (r.get("supervisor") or ""))
                    ).strip()

                pw_cols = st.columns([1, 2])
                with pw_cols[0]:
                    sup_pw = st.text_input("Senha do Supervisor", type="password")
                with pw_cols[1]:
                    st.caption("Somente o supervisor vinculado pode autorizar.")

                submitted = st.form_submit_button("Salvar decisão")
                if submitted:
                    trip_supervisor = (r.get("supervisor") or "").strip()
                    if not trip_supervisor:
                        st.error("Viagem sem supervisor vinculado. Cadastre no Admin > Parametrização.")
                    elif reviewer != trip_supervisor:
                        st.error("O revisor informado deve ser exatamente o supervisor vinculado a este colaborador.")
                    elif not _check_supervisor_password(reviewer, sup_pw):
                        st.error("Senha do supervisor inválida.")
                    elif decision == "Reprovar" and not reason.strip():
                        st.error("Observação é obrigatória para reprovação.")
                    else:
                        set_review(
                            trip_id=r["trip_id"],
                            status=("REJECTED" if decision == "Reprovar" else "APPROVED"),
                            reason=reason.strip(),
                            reviewer=reviewer,
                        )
                        st.cache_data.clear()
                        st.success("Decisão registrada com autorização do supervisor.")
                        st.toast("Status atualizado.", icon="✅")

    # Exportar CSV do mês com status (dados filtrados atuais, não só a página)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar CSV do mês (com status de auditoria)",
        data=csv_bytes,
        file_name=f"auditoria_{year}_{month:02d}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ---- Relatório ----
def render_trip_card_report(r: pd.Series):
    base_color = "#5B8DEF"
    if str(r.get("vehicle_type") or "").lower().startswith("uber black"):
        base_color = "#7B61FF"
    elif str(r.get("vehicle_type") or "").lower().startswith("uberx"):
        base_color = "#2EC4B6"
    elif str(r.get("program") or "").lower().startswith("business"):
        base_color = "#FF9F1C"

    dt_txt = pd.to_datetime(r.get("trip_dt")).strftime('%d/%m/%Y %H:%M') if pd.notna(r.get("trip_dt")) else "—"
    st.markdown(
        f"""
        <div class="trip-card" style="border-left:6px solid {base_color};">
          <div class="tc-head">
            <div class="tc-title"><strong>{r.get('employee_full','—')}</strong></div>
            <div class="tc-value">{money_fmt(r.get('amount',0), r.get('currency') or 'BRL')}</div>
          </div>
          <div class="tc-sub">
            <span>Data: {dt_txt}</span> ·
            <span>Cidade: {r.get('city') or '—'}</span> ·
            <span>Programa: {r.get('program') or '—'}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
def pretty_table(
    df: pd.DataFrame,
    money_cols: Iterable[str] = (),
    km_cols: Iterable[str] = (),
    datetime_cols: Iterable[str] = ()
):
    """
    Exibe um dataframe padronizado no Streamlit, aplicando formatação monetária,
    de distância (km) e de data/hora onde indicado.
    """
    if df.empty:
        st.info("Sem dados para exibir.")
        return

    d = df.copy()

    # Formatação monetária
    for c in money_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(
                lambda v: money_fmt(v) if pd.notna(v) else "—"
            )

    # Formatação km
    for c in km_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(
                lambda x: (
                    f"{x:,.2f} km".replace(",", "X").replace(".", ",").replace("X", ".")
                    if pd.notna(x) else "—"
                )
            )

    # Formatação de datas/horas
    for c in datetime_cols:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce").map(
                lambda x: x.strftime("%d/%m/%Y %H:%M") if pd.notna(x) else "—"
            )

    st.dataframe(d, use_container_width=True, hide_index=True)

def tela_relatorio():
    # Proteção por senha de admin
    if ADMIN_PASSWORD:
        pw = st.text_input("Senha do administrador (necessária para visualizar o Relatório)", type="password")
        if not pw:
            st.stop()
        if pw != ADMIN_PASSWORD:
            st.error("Senha incorreta.")
            st.stop()

    st.header("Relatório de viagens e gastos")

    df = load_all_from_db()
    if df.empty:
        st.info("Ainda não há dados no banco. Vá à aba **Admin** e faça o upload dos CSVs.")
        return

    # Fallback: preencher department a partir de employee_id se necessário
    if "department" not in df.columns:
        df["department"] = None
    mask_fix = df["department"].isna() & df["employee_id"].notna()
    if mask_fix.any():
        df.loc[mask_fix, "department"] = df.loc[mask_fix, "employee_id"].apply(_dept_from_employee_id)

    # Filtrar só viagens (fare)
    df = df[df["trip_dt"].notna()]
    if "tx_type" in df.columns:
        df = df[df["tx_type"].astype(str).str.lower().str.contains("fare", na=False)]
    if df.empty:
        st.warning("Não há viagens após aplicar os filtros iniciais.")
        return

    # Juntar supervisor (para filtros e painéis)
    ass = list_assignments()[["employee_email", "supervisor"]]
    df = df.merge(ass, on="employee_email", how="left")

    min_dt = df["trip_dt"].min().date()
    max_dt = df["trip_dt"].max().date()

    with st.container(border=True):
        f1, f2, f3, f4, f5, f6 = st.columns([2, 2, 2, 3, 2, 2])
        with f1:
            dr = st.date_input(
                "Período",
                value=(min_dt, max_dt),
                min_value=min_dt,
                max_value=max_dt,
                key="period"
            )

            # --- Tratamento robusto do retorno do date_input ---
            if isinstance(dr, (tuple, list)):
                if len(dr) == 2 and all(dr):
                    start_d, end_d = dr[0], dr[1]
                elif len(dr) == 1 and dr[0]:
                    start_d = end_d = dr[0]
                else:
                    start_d, end_d = min_dt, max_dt
            else:
                # um único date
                start_d = end_d = dr

            # Normaliza para date e garante ordem
            start_d = pd.to_datetime(start_d).date()
            end_d   = pd.to_datetime(end_d).date()
            if end_d < start_d:
                start_d, end_d = end_d, start_d

        with f2:
            prog_all = sorted(df["program"].dropna().unique().tolist())
            prog = st.multiselect("Programa(s)", options=prog_all, default=prog_all)
        with f3:
            dept_all = sorted(df["department"].dropna().unique().tolist())
            dept = st.multiselect("Departamento(s)", options=dept_all, default=dept_all)
        with f4:
            city_all = sorted(df["city"].dropna().unique().tolist())
            city_sel = st.multiselect("Cidade(s)", options=city_all, default=city_all)
        with f5:
            sup_all = sorted(df["supervisor"].dropna().unique().tolist())
            sup_sel = st.multiselect("Supervisor(es)", options=sup_all, default=sup_all)
        with f6:
            q = st.text_input("Buscar (nome/e-mail/cidade/serviço)", placeholder="Ex.: maria, goiânia, uberx").strip().lower()

    # Filtros
    flt = (df["trip_dt"].dt.date >= start_d) & (df["trip_dt"].dt.date <= end_d)
    if prog:
        flt &= df["program"].fillna("").isin(prog)
    if dept:
        flt &= df["department"].fillna("").isin(dept)
    if city_sel:
        flt &= df["city"].fillna("").isin(city_sel)
    if sup_sel:
        flt &= df["supervisor"].fillna("").isin(sup_sel)
    if q:
        flt &= (
            df["employee_full"].fillna("").str.lower().str.contains(q) |
            df["employee_email"].fillna("").str.lower().str.contains(q) |
            df["city"].fillna("").str.lower().str.contains(q) |
            df["vehicle_type"].fillna("").str.lower().str.contains(q)
        )

    dff = df[flt].copy()
    if dff.empty:
        st.warning("Nenhum registro para o período/filtros. Ajuste os filtros acima.")
        return

    # KPIs
    total_trips = len(dff)
    total_spend = float(pd.to_numeric(dff["amount"], errors="coerce").sum())
    avg_ticket = total_spend / total_trips if total_trips else 0.0
    total_km = float(pd.to_numeric(dff["distance_km"], errors="coerce").sum())
    curr_code = dff["currency"].mode().iloc[0] if "currency" in dff.columns and not dff["currency"].mode().empty else "BRL"
    curr_symbol = CURRENCY_SYMBOL.get(str(curr_code).upper(), "R$")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.caption("Número de viagens")
            st.metric(label="Viagens", value=f"{total_trips:,}".replace(",", "."))
    with c2:
        with st.container(border=True):
            st.caption("Total de gastos")
            st.metric(label="Gastos", value=money_fmt(total_spend, curr_symbol))
    with c3:
        with st.container(border=True):
            st.caption("Tíquete médio")
            st.metric(label="Tíquete médio", value=money_fmt(avg_ticket, curr_symbol))
    with c4:
        with st.container(border=True):
            st.caption("Distância total (km)")
            st.metric(label="Distância (km)", value=f"{total_km:,.2f}".replace(",", "."))

    # ===== Painéis de Auditoria (por supervisor) =====
    with st.container(border=True):
        st.subheader("Status de auditoria por supervisor")
        dfa = enrich_with_audit(dff)

        # GARANTE coluna 'supervisor' e normaliza vazio
        if "supervisor" not in dfa.columns:
            dfa["supervisor"] = None
        dfa["supervisor"] = dfa["supervisor"].fillna("—").astype(str).str.strip().replace({"": "—"})

        grp = dfa.groupby("supervisor", dropna=False).agg(
            total=("trip_id", "count"),
            aprovadas=("review_status", lambda s: (s == "APPROVED").sum()),
            reprovadas=("review_status", lambda s: (s == "REJECTED").sum())
        ).reset_index()
        grp["pendentes"] = grp["total"] - grp["aprovadas"] - grp["reprovadas"]
        grp["pct_auditado"] = ((grp["aprovadas"] + grp["reprovadas"]) / grp["total"] * 100).round(1)

        tbl = grp.rename(columns={
            "supervisor": "Supervisor",
            "total": "Viagens",
            "aprovadas": "Aprovadas",
            "reprovadas": "Reprovadas",
            "pendentes": "Pendentes",
            "pct_auditado": "% auditado"
        }).assign(**{"% auditado": grp["pct_auditado"]}).sort_values(
            ["Pendentes", "% auditado", "Supervisor"], ascending=[False, False, True]
        )
        pretty_table(tbl, datetime_cols=())

        if not grp.empty:
            bar_with_labels(
                grp.rename(columns={"supervisor": "Supervisor", "pendentes": "Pendentes"}),
                "Supervisor", "Pendentes", "Pendentes por supervisor", number_format=",.0f"
            )

    # Séries no tempo (corrigido eixo mês/ano)
    with st.container(border=True):
        st.subheader("Séries no tempo")
        gm = dff[["trip_dt", "amount"]].rename(columns={"trip_dt": "dt", "amount": "Gasto"})
        monthly_line_with_labels(gm, "dt", "Gasto", "Gastos por mês (linha)", "Valor", fmt=",.2f")

        vm = dff[["trip_dt", "trip_id"]].rename(columns={"trip_dt": "dt", "trip_id": "Viagens"})
        vm["Viagens"] = 1
        monthly_line_with_labels(vm, "dt", "Viagens", "Quantidade de viagens por mês (linha)", "Viagens", fmt=",.0f")

    # Métricas por categoria
    with st.container(border=True):
        st.subheader("Métricas por categoria")
        g1, g2 = st.columns(2)
        with g1:
            gp = (dff.groupby("program")["amount"].sum().sort_values(ascending=False)
                  .rename("Valor").reset_index())
            gp = gp.copy()
            gp["program"] = gp["program"].fillna("—")
            bar_with_labels(gp, "program", "Valor", "Gasto por programa", number_format=",.2f")
        with g2:
            gp_tbl = gp.rename(columns={"program": "Programa", "Valor": "Gasto"})
            pretty_table(gp_tbl.head(15), money_cols=("Gasto",))

    # Análises por participante/local
    with st.container(border=True):
        st.subheader("Análises por participante e localização")
        a1, a2 = st.columns(2)
        with a1:
            topn_col = st.slider("Top colaboradores (N)", 3, 30, 10, key="top_colab")
            colab = (dff.groupby("employee_full")["amount"].sum()
                     .sort_values(ascending=False).head(topn_col)
                     .rename("Gasto").reset_index())
            colab["Colaborador"] = colab["employee_full"].fillna("—")
            bar_with_labels(colab[["Colaborador", "Gasto"]], "Colaborador", "Gasto",
                            f"Top {topn_col} colaboradores por gasto", number_format=",.2f")
            pretty_table(colab[["Colaborador", "Gasto"]], money_cols=("Gasto",))
        with a2:
            topn_dep = st.slider("Top departamentos (N)", 3, 30, 10, key="top_dept")
            dep = (dff.assign(Departamento=dff["department"].fillna("—"))
                   .groupby("Departamento")["amount"].sum()
                   .sort_values(ascending=False).head(topn_dep)
                   .rename("Gasto").reset_index())
            bar_with_labels(dep, "Departamento", "Gasto", f"Top {topn_dep} departamentos por gasto", number_format=",.2f")
            pretty_table(dep, money_cols=("Gasto",))
        b1, b2 = st.columns(2)
        with b1:
            topn_city = st.slider("Top cidades (N)", 3, 30, 10, key="top_city")
            city = (dff.assign(Cidade=dff["city"].fillna("—"))
                    .groupby("Cidade")["amount"].sum()
                    .sort_values(ascending=False).head(topn_city)
                    .rename("Gasto").reset_index())
            bar_with_labels(city, "Cidade", "Gasto", f"Top {topn_city} cidades por gasto", number_format=",.2f")
            pretty_table(city, money_cols=("Gasto",))
        with b2:
            line_hour_chart(dff, "request_hour", "Viagens por hora do dia (solicitação)")

    # Lista de viagens (cards)
    st.subheader("Viagens")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    with st.container(border=True):
        oc1, oc2, oc3, oc4 = st.columns([2, 1, 1, 2])
        with oc1:
            how = st.selectbox("Ordenar por", ["Valor da viagem", "Distância (km)"])
        with oc2:
            asc = st.toggle("Crescente", value=False)
        with oc3:
            topn = st.slider("Quantidade por página", 5, 40, 10)
        with oc4:
            show_table = st.checkbox("Exibir também uma tabela harmonizada da página", value=False)

        dff["distance_km_num"] = pd.to_numeric(dff["distance_km"], errors="coerce")
        dff["amount_num"] = pd.to_numeric(dff["amount"], errors="coerce")
        dff = dff.sort_values("amount_num" if how == "Valor da viagem" else "distance_km_num",
                              ascending=asc, na_position="last")

        total = len(dff)
        pages = max(1, (total + topn - 1) // topn)
        page = st.number_input("Página", 1, pages, 1)
        start = (page - 1) * topn
        end = start + topn
        page_df = dff.iloc[start:end].copy()

        if show_table:
            tbl = page_df[[
                "trip_dt","employee_full","employee_email","program","department","city","vehicle_type","amount","distance_km"
            ]].rename(columns={
                "trip_dt":"Data","employee_full":"Colaborador","employee_email":"E-mail",
                "program":"Programa","department":"Departamento","city":"Cidade",
                "vehicle_type":"Serviço","amount":"Valor","distance_km":"Distância (km)"
            })
            pretty_table(tbl, money_cols=("Valor",), km_cols=("Distância (km)",), datetime_cols=("Data",))

        for _, r in page_df.iterrows():
            render_trip_card_report(r)
            with st.expander("Detalhar", expanded=False):
                colA, colB = st.columns([1,1])
                dt_txt = pd.to_datetime(r.get("trip_dt")).strftime('%d/%m/%Y %H:%M') if pd.notna(r.get("trip_dt")) else "—"
                with colA:
                    st.write("**Data:**", dt_txt)
                    st.write("**Colaborador:**", r.get("employee_full"))
                    st.write("**E-mail:**", r.get("employee_email"))
                    st.write("**Programa:**", r.get("program") or "—")
                    st.write("**Departamento:**", r.get("department") or "—")
                    st.write("**Serviço:**", r.get("vehicle_type") or "—")
                    st.write("**Cidade:**", r.get("city") or "—")
                    st.write("**Código da despesa (motivo):**", r.get("expense_code") or "—")
                with colB:
                    st.write("**Valor:**", money_fmt(r.get("amount", 0), r.get("currency") or "BRL"))
                    dist_txt = f"{float(r.get('distance_km')):.2f}" if pd.notna(r.get("distance_km")) else "—"
                    st.write("**Distância (km):**", dist_txt)
                    st.write("**Origem:**", r.get("pickup_addr") or "—")
                    st.write("**Destino:**", r.get("dropoff_addr") or "—")
                    st.caption(f"Arquivo: {r.get('source_file')} · ID: {r.get('trip_id')}")
                origem = (r.get("pickup_addr") or "").strip()
                destino = (r.get("dropoff_addr") or "").strip()
                if origem and destino:
                    q = f"https://www.google.com/maps?output=embed&saddr={urllib.parse.quote(origem)}&daddr={urllib.parse.quote(destino)}"
                    components.html(
                        f'<iframe src="{q}" width="100%" height="360" style="border:0;" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>',
                        height=380,
                    )
                    st.markdown(
                        f"[📍 Abrir rota no Google Maps]({ 'https://www.google.com/maps/dir/?api=1&origin=' + urllib.parse.quote(origem) + '&destination=' + urllib.parse.quote(destino) })  ·  "
                        f"[🔎 Pesquisar origem]({ 'https://www.google.com/maps/search/?api=1&query=' + urllib.parse.quote(origem) })  ·  "
                        f"[🔎 Pesquisar destino]({ 'https://www.google.com/maps/search/?api=1&query=' + urllib.parse.quote(destino) })",
                        help="Se o iframe não for exibido pelo seu navegador, use os links."
                    )

    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar dados filtrados (CSV)",
        data=csv_bytes,
        file_name="uber_gastos_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------- App ----------------------
def main():
    st.set_page_config(APP_TITLE, page_icon="🚗", layout="wide")
    st.title(APP_TITLE)
    st.caption(SUBTITLE)

    ensure_db()

    modo = st.sidebar.radio("Selecione a área", ["Relatório", "Auditoria", "Admin"], index=0)
    if modo == "Admin":
        tela_admin()
    elif modo == "Auditoria":
        tela_auditoria()
    else:
        tela_relatorio()

if __name__ == "__main__":
    main()
    
