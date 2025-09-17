from __future__ import annotations

import io
import re
import csv
import zipfile
import hashlib
from typing import List, Optional, Tuple

import pandas as pd

# ---------- Parsing de valores monetários ----------
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

# ---------- Datas/horas ----------
def combine_local_date_time(date_col: Optional[pd.Series], time_col: Optional[pd.Series]) -> pd.Series:
    """Interpreta data/hora e tenta autodetectar dayfirst quando necessário."""
    if date_col is None:
        return pd.NaT
    d = date_col.fillna("").astype(str).str.strip()
    t = (time_col.fillna("").astype(str).str.strip()) if time_col is not None else ""
    s = d + " " + t
    s = s.str.replace(r"\s*(AM|PM)$", r" \1", regex=True)
    dt = pd.to_datetime(s, errors="coerce")  # autodetect
    if dt.isna().mean() > 0.6:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt

def parse_ampm_hour_any(s) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip().upper().replace(" ", "")
    t = re.sub(r"^00:", "12:", t)  # 00:00AM/PM -> 12:00...
    for fmt in ["%I:%M%p", "%I%p"]:
        try:
            return pd.to_datetime(t, format=fmt, errors="raise").hour
        except Exception:
            pass
    try:
        return pd.to_datetime(s, format="%H:%M", errors="raise").hour
    except Exception:
        return None

def _dept_from_employee_id(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    for sep in (" - ", " | ", " — ", " – "):
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            return left or None
    return None

# ---------- Leitura CSV Uber ----------
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
    """Suporta CSV puro e ZIP contendo múltiplos CSVs."""
    name = uploaded.name.lower()
    raw = uploaded.read()
    dfs: list[pd.DataFrame] = []
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
            for member in z.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with z.open(member) as f:
                    buf = f.read()
                dfs.extend(read_bytes_to_df_list(buf))
    else:
        dfs.extend(read_bytes_to_df_list(raw))
    if not dfs:
        return pd.DataFrame()
    for df in dfs:
        if "source_file" not in df.columns:
            df["source_file"] = uploaded.name
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
        out["request_dt"] = pd.to_datetime(df[utc_dt], errors="coerce")

    out["request_time_str"] = df[req_t] if req_t in df.columns else None

    amount_col = find_col(df, "amount_brl")
    out["amount"] = money_series(df[amount_col]) if amount_col else pd.NA
    out["currency"] = out.get("currency_local", pd.Series(["BRL"] * len(out))).fillna("BRL")

    dist_mi = pd.to_numeric(out.get("distance_mi", pd.Series([None]*len(out))).astype(str).str.replace(",", "."), errors="coerce")
    out["distance_km"] = dist_mi * 1.60934

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

def filter_df_by_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    d = df.copy()
    d["trip_dt"] = pd.to_datetime(d["trip_dt"], errors="coerce")
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthBegin(1))
    return d[(d["trip_dt"] >= start) & (d["trip_dt"] < end)]
