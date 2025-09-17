from __future__ import annotations

import os
import sqlite3
import hashlib
from datetime import datetime, date
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# ------- Config -------
HIST_DB_PATH = st.secrets.get("HIST_DB_PATH", ".uber_history/history.sqlite")
HIST_TABLE   = st.secrets.get("HIST_TABLE", "uber_trips")
SUPERVISOR_PW_SALT = st.secrets.get("SUPERVISOR_PW_SALT", "orbis-supervisor-salt")

os.makedirs(os.path.dirname(HIST_DB_PATH), exist_ok=True)

# ------- Helpers -------
def _hash_pw(pw: str) -> str:
    return hashlib.sha256((SUPERVISOR_PW_SALT + (pw or "")).encode("utf-8")).hexdigest()

def ensure_db():
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    # viagens
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
    # Índices
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

    # senhas de supervisores
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_supervisors (
            supervisor   TEXT PRIMARY KEY,
            pass_hash    TEXT NOT NULL,
            updated_at   TEXT
        );
    """)

    con.commit()
    con.close()

# ------- Inserts e leitura -------
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
    cur.execute("BEGIN;")
    cur.executemany(f"""
        INSERT OR IGNORE INTO {HIST_TABLE} (
            trip_id, first_name, last_name, tx_type, status, employee_name, employee_email,
            vehicle_type, program, employee_id, pickup_addr, dropoff_addr, city, distance_km,
            expense_code, amount, currency, trip_dt, department, source_file, request_hour
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    inserted = cur.rowcount if hasattr(cur, "rowcount") else 0
    con.commit()
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

def _month_bounds(year: int, month: int) -> Tuple[date, date]:
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

# ------- Parametrização & auditoria -------
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

def delete_assignment(employee_email: str):
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM audit_assignments WHERE employee_email = ?", (employee_email.strip().lower(),))
    con.commit()
    con.close()

def list_supervisors() -> List[str]:
    df = list_assignments()
    sups = sorted([s for s in df["supervisor"].dropna().unique().tolist()])
    return sups

def set_supervisor_password(supervisor: str, password: str):
    if not (supervisor and password):
        raise ValueError("Supervisor e senha são obrigatórios.")
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO audit_supervisors (supervisor, pass_hash, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(supervisor) DO UPDATE SET
            pass_hash=excluded.pass_hash,
            updated_at=excluded.updated_at
    """, (supervisor.strip(), _hash_pw(password), datetime.utcnow().isoformat(timespec="seconds")))
    con.commit()
    con.close()

def delete_supervisor_password(supervisor: str):
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM audit_supervisors WHERE supervisor = ?", (supervisor.strip(),))
    con.commit()
    con.close()

def check_supervisor_password(supervisor: str, password: str) -> bool:
    con = sqlite3.connect(HIST_DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT pass_hash FROM audit_supervisors WHERE supervisor = ?", (supervisor.strip(),))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    return row[0] == _hash_pw(password or "")

def get_month_trips_with_assignment(year: int, month: int, supervisor: Optional[str] = None) -> pd.DataFrame:
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
