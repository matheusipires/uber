from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
import streamlit as st
import altair as alt

CURRENCY_SYMBOL = {"BRL": "R$", "USD": "$", "EUR": "€", "R$": "R$", "$": "$", "€": "€"}

def money_fmt(v: float, currency: str = "R$") -> str:
    symbol = CURRENCY_SYMBOL.get(str(currency).strip().upper(), currency or "R$")
    try:
        f = float(v or 0)
    except Exception:
        f = 0.0
    s = f"{symbol} {f:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

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

def pretty_table(df: pd.DataFrame, money_cols: Iterable[str] = (), km_cols: Iterable[str] = (), datetime_cols: Iterable[str] = ()):
    if df.empty:
        st.info("Sem dados para exibir.")
        return
    d = df.copy()
    for c in money_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: money_fmt(v))
    for c in km_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: f"{x:,.2f} km".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(x) else "—")
    for c in datetime_cols:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
    st.dataframe(d, use_container_width=True, hide_index=True)

def monthly_line_with_labels(df: pd.DataFrame, date_col: str, value_col: str, title: str, y_title: str, fmt=",.2f"):
    if df.empty:
        st.info("Sem dados para o gráfico.")
        return
    d = df.copy()
    month_idx = pd.to_datetime(d[date_col], errors="coerce").dt.to_period("M")
    d["month_start"] = month_idx.dt.to_timestamp()
    d = d.groupby("month_start", as_index=False)[value_col].sum().sort_values("month_start")
    line = alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("month_start:T", title=None),
        y=alt.Y(f"{value_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip("month_start:T", title="Mês", format="%b %Y"),
            alt.Tooltip(f"{value_col}:Q", format=fmt, title=y_title),
        ],
    ).properties(height=320, title=title)
    labels = alt.Chart(d).mark_text(dy=-10).encode(
        x="month_start:T", y=f"{value_col}:Q", text=alt.Text(f"{value_col}:Q", format=fmt)
    )
    st.altair_chart(line + labels, use_container_width=True)

def month_year_selector(label_prefix: str = "Período") -> Tuple[int, int]:
    cols = st.columns([1, 1, 2])
    with cols[0]:
        from datetime import datetime as _dt
        year = st.number_input(f"{label_prefix} — Ano", min_value=2018, max_value=2100, value=_dt.now().year, step=1)
    with cols[1]:
        from datetime import datetime as _dt
        month = st.selectbox(f"{label_prefix} — Mês", list(range(1,13)), index=(_dt.now().month-1))
    with cols[2]:
        st.caption("Selecione o mês/ano de referência.")
    return int(year), int(month)

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
    from .ui import money_fmt  # safe import
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

def render_trip_card_report(r: pd.Series):
    from .ui import money_fmt  # safe import
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
