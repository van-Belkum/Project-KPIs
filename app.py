import os
import io
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional forecasting libs (Prophet → ETS → naive)
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

st.set_page_config(page_title="KPI Project Report", layout="wide")

# =============================
# Config / Column Mapping
# =============================
DEFAULT_MILESTONES_URL = "https://5367938.app.netsuite.com/app/site/hosting/scriptlet.nl?script=2850&deploy=1&compid=5367938&wsId=426"
DEFAULT_DEPENDENCIES_URL = "https://5367938.app.netsuite.com/app/site/hosting/scriptlet.nl?script=2850&deploy=1&compid=5367938&wsId=618"

DEFAULT_MILESTONES = ["GA", "DD", "MSV", "Legal Access", "MS6"]

ID_COL = "Internal ID"
TEMPLATE_COL = "Template"
STATUS_COL = "Status"
MILESTONE_COL = "Milestone"
DATE_COL = "Date"
FORECAST_FLAG_COL = "Is Forecast"
DESIGNER_COL = "Designer"
DEPENDENCY_COL = "Dependency"

DEFAULT_COMPLETION_STATUSES = ["Accepted", "Completed", "Actual"]

# =============================
# Helpers
# =============================
@st.cache_data(ttl=15 * 60)
def fetch_netsuite_tabular(url: str, headers: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    r = requests.get(url, headers=headers or {}, timeout=60)
    r.raise_for_status()
    data = r.content
    for loader in (
        lambda b: pd.read_csv(io.BytesIO(b)),
        lambda b: pd.read_excel(io.BytesIO(b)),
        lambda b: pd.read_csv(io.BytesIO(b), sep="\t"),
    ):
        try:
            return loader(data)
        except Exception:
            continue
    return pd.DataFrame()


def coerce_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def week_floor(d: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(d):
        return d
    return (d - pd.to_timedelta(d.weekday(), unit="D")).normalize()


def latest_completion(sub: pd.DataFrame, completion_statuses: List[str]) -> Optional[pd.Timestamp]:
    if sub.empty:
        return pd.NaT
    mask_actual = (sub.get(FORECAST_FLAG_COL, False) == False)
    if STATUS_COL in sub.columns:
        mask_actual &= sub[STATUS_COL].astype(str).isin(completion_statuses)
    done = sub[mask_actual]
    if done.empty or DATE_COL not in done.columns:
        return pd.NaT
    return done[DATE_COL].max()


def next_forecast(sub: pd.DataFrame) -> Optional[pd.Timestamp]:
    if sub.empty or DATE_COL not in sub.columns:
        return pd.NaT
    future = sub[(sub.get(FORECAST_FLAG_COL, True) == True) & (sub[DATE_COL] >= pd.Timestamp.today().normalize())]
    if future.empty:
        return pd.NaT
    return future[DATE_COL].min()


def weekly_targets_vs_actuals(df: pd.DataFrame, milestone: str, completion_statuses: List[str]) -> pd.DataFrame:
    s = df[df[MILESTONE_COL] == milestone].copy()
    s["week"] = s[DATE_COL].apply(week_floor)

    targets = (
        s[s.get(FORECAST_FLAG_COL, True) == True]
        .groupby("week")[ID_COL]
        .nunique()
        .rename("Target")
        .reset_index()
    ) if not s.empty else pd.DataFrame(columns=["week", "Target"]) 

    actual_mask = s.get(FORECAST_FLAG_COL, False) == False
    if STATUS_COL in s.columns:
        actual_mask &= s[STATUS_COL].astype(str).isin(completion_statuses)
    actuals = (
        s[actual_mask]
        .groupby("week")[ID_COL]
        .nunique()
        .rename("Actual")
        .reset_index()
    ) if not s.empty else pd.DataFrame(columns=["week", "Actual"]) 

    out = pd.merge(targets, actuals, on="week", how="outer").sort_values("week").fillna(0)
    out.insert(1, "Milestone", milestone)
    return out


def forecast_weekly(series_df: pd.DataFrame, value_col: str = "Actual", periods: int = 8) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame(columns=["week", "Forecast", "yhat_lower", "yhat_upper"]) 
    df = series_df.rename(columns={"week": "ds", value_col: "y"})[["ds", "y"]].dropna()
    if len(df) < 2:
        future_index = pd.date_range(pd.Timestamp.today().normalize(), periods=periods, freq="W-MON")
        return pd.DataFrame({"week": future_index, "Forecast": float(df["y"].iloc[-1]) if len(df) else 0.0})

    if PROPHET_OK and len(df) >= 4:
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods, freq="W-MON")
        fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return fcst.rename(columns={"ds": "week", "yhat": "Forecast"})

    if STATSMODELS_OK and len(df) >= 4:
        df2 = df.set_index("ds").asfreq("W-MON").fillna(method="ffill")
        model = ExponentialSmoothing(df2["y"], trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        future_index = pd.date_range(df2.index.max(), periods=periods+1, freq="W-MON")[1:]
        forecast = fit.forecast(len(future_index))
        return pd.DataFrame({"week": future_index, "Forecast": forecast.values})

    last = float(df["y"].iloc[-1])
    future_index = pd.date_range(pd.Timestamp.today().normalize(), periods=periods, freq="W-MON")
    return pd.DataFrame({"week": future_index, "Forecast": last})


def progress_table(df: pd.DataFrame, milestones_order: List[str], completion_statuses: List[str]) -> pd.DataFrame:
    rows = []
    for site, g in df.groupby(ID_COL):
        entry = {ID_COL: site}
        completed_count = 0
        next_due_date = pd.NaT
        next_due_name = ""
        current_stage = "Not Started"

        for m in milestones_order:
            sg = g[g[MILESTONE_COL] == m]
            a = latest_completion(sg, completion_statuses)
            f = next_forecast(sg)
            entry[f"{m} Actual"] = a
            entry[f"{m} Forecast"] = f
            if pd.notna(a):
                completed_count += 1
                current_stage = m
            elif pd.isna(next_due_date) and pd.notna(f):
                next_due_date, next_due_name = f, m
        entry["% Complete"] = round(100 * completed_count / max(1, len(milestones_order)), 1)
        entry["Current Stage"] = current_stage
        entry["Next Due"] = next_due_name
        entry["Next Due Date"] = next_due_date
        rows.append(entry)
    out = pd.DataFrame(rows)
    if "Template" in df.columns:
        tmpl = df.groupby(ID_COL)["Template"].agg(lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan)
        out = out.merge(tmpl.rename("Template"), on=ID_COL, how="left")
    cols = [ID_COL, "Template"] if "Template" in out.columns else [ID_COL]
    for m in milestones_order:
        cols += [f"{m} Actual", f"{m} Forecast"]
    cols += ["% Complete", "Current Stage", "Next Due", "Next Due Date"]
    return out[cols].sort_values(["% Complete", "Next Due Date"], ascending=[False, True])


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.title("KPI Project Report")
    st.caption("Live from NetSuite – no Excel needed.")

    default_auth = st.secrets.get("auth", {}).get("header", "")
    milestones_url = st.text_input("Milestones URL", value=DEFAULT_MILESTONES_URL)
    dependencies_url = st.text_input("Dependencies URL", value=DEFAULT_DEPENDENCIES_URL)
    auth_header = st.text_input("Authorization header (optional)", value=default_auth, type="password")

    completion_statuses = st.multiselect("Completion statuses count as ACTUAL", options=DEFAULT_COMPLETION_STATUSES, default=DEFAULT_COMPLETION_STATUSES)

    st.markdown("---")
    st.markdown("**Column Mapping**")
    ID_COL = st.text_input("Internal ID column", ID_COL)
    TEMPLATE_COL = st.text_input("Template column", TEMPLATE_COL)
    STATUS_COL = st.text_input("Status column", STATUS_COL)
    MILESTONE_COL = st.text_input("Milestone column", MILESTONE_COL)
    DATE_COL = st.text_input("Date column", DATE_COL)
    FORECAST_FLAG_COL = st.text_input("Is Forecast column", FORECAST_FLAG_COL)
    DESIGNER_COL = st.text_input("Designer column", DESIGNER_COL)
    DEPENDENCY_COL = st.text_input("Dependency column", DEPENDENCY_COL)


# =============================
# Data Load
# =============================
headers = {"Authorization": auth_header} if auth_header else None
with st.spinner("Fetching data from NetSuite..."):
    ms_df = fetch_netsuite_tabular(milestones_url, headers=headers)
    dep_df = fetch_netsuite_tabular(dependencies_url, headers=headers)

for df in (ms_df, dep_df):
    if not df.empty and DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# =============================
# Layout (Tabs)
# =============================
st.title("📊 KPI Project Report – Pro")

if ms_df.empty:
    st.error("No milestone data loaded. Check your URL/auth/columns.")
else:
    total_sites = ms_df[ID_COL].nunique() if ID_COL in ms_df.columns else len(ms_df)
    total_designers = ms_df[DESIGNER_COL].nunique() if DESIGNER_COL in ms_df.columns else np.nan
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Sites", total_sites)
    kpi_cols[1].metric("Designers", total_designers)
    kpi_cols[2].metric("Rows", len(ms_df))
    kpi_cols[3].metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M"))

    tabs = st.tabs(["Overview", "Progress", "Milestones", "Dependencies", "Forecasts", "Settings"]) 

    with tabs[0]:
        st.subheader("Designer KPIs")
        if DESIGNER_COL in ms_df.columns and DATE_COL in ms_df.columns:
            actuals = ms_df[(ms_df.get(FORECAST_FLAG_COL, False) == False)]
            actuals = actuals.assign(week=actuals[DATE_COL].apply(week_floor))
            kpi = actuals.groupby([DESIGNER_COL, "week"]).size().reset_index(name="Actual completions")
            st.dataframe(kpi, use_container_width=True)
        else:
            st.caption("Need Designer & Date columns for KPIs.")

    with tabs[1]:
        st.subheader("Status – Progress by Site")
        milestones_order = [m for m in ["GA","DD","MSV","Legal Access","MS6"] if m in ms_df[MILESTONE_COL].unique().tolist()] or ["GA","DD","MSV","Legal Access","MS6"]
        prog = progress_table(ms_df, milestones_order, completion_statuses)
        st.dataframe(prog, use_container_width=True)

    with tabs[2]:
        st.subheader("Weekly Targets vs Actuals")
        pick = st.multiselect("Choose milestones", options=sorted(ms_df[MILESTONE_COL].dropna().unique()), default=[m for m in ["GA","DD","Legal Access","MS6"] if m in ms_df[MILESTONE_COL].unique()])
        for m in pick:
            with st.expander(f"{m}"):
                taa = weekly_targets_vs_actuals(ms_df, m, completion_statuses)
                st.dataframe(taa, use_container_width=True)
                if not taa.empty:
                    chart_df = taa.set_index("week")[ [c for c in ["Target","Actual"] if c in taa.columns] ]
                    st.line_chart(chart_df)

    with tabs[3]:
        st.subheader("Dependencies – Completed / Forecasted (Weekly)")
        if dep_df.empty:
            st.caption("No dependencies feed loaded.")
        else:
            if "Date" in dep_df.columns:
                dep_df["week"] = pd.to_datetime(dep_df["Date"], errors="coerce").apply(week_floor)
            pivot = dep_df.pivot_table(index=["week", DEPENDENCY_COL], columns=STATUS_COL, values=ID_COL, aggfunc="nunique").reset_index()
            st.dataframe(pivot.rename_axis(None, axis=1), use_container_width=True)

    with tabs[4]:
        st.subheader("Predictive Forecast – next 8 weeks")
        picked = st.multiselect("Milestones to forecast", options=sorted(ms_df[MILESTONE_COL].dropna().unique()), default=[m for m in ["GA","DD","Legal Access","MS6"] if m in ms_df[MILESTONE_COL].unique()])
        for m in picked:
            with st.expander(f"{m}"):
                taa = weekly_targets_vs_actuals(ms_df, m, completion_statuses)
                fc = forecast_weekly(taa[["week", "Actual"]].dropna(), periods=8)
                st.markdown("**History (Actuals)**")
                st.dataframe(taa, use_container_width=True)
                st.markdown("**Forecast (next 8 weeks)**")
                st.dataframe(fc, use_container_width=True)
                if not fc.empty and "Forecast" in fc.columns:
                    st.line_chart(fc.set_index("week")["Forecast"])

    with tabs[5]:
        st.subheader("Settings & Help")
        st.markdown(
            """
**Auth header**  
Set `auth.header = "Bearer <token>"` (or Cookie string) in Secrets for NetSuite endpoints. The sidebar reads it automatically.

**Completion logic**  
Use the sidebar multiselect to define which Status values mean a milestone is **completed**.

**Forecasting**  
The app uses Prophet → ETS → naive fallback depending on installed packages and available history.
            """
        )
