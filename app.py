import os
import io
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta

# Optional: forecasting models (best → fallback → naive)
try:
    from prophet import Prophet  # pip install prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # pip install statsmodels
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="KPI Project Report", layout="wide")

# Expected column names (you can remap in the UI if yours differ)
ID_COL = "Internal ID"
TEMPLATE_COL = "Template"
STATUS_COL = "Status"  # e.g. Accepted/Rejected/Completed/Forecasted
MILESTONE_COL = "Milestone"  # GA, DD, Legal Access, MS6
DATE_COL = "Date"
FORECAST_FLAG_COL = "Is Forecast"  # True for plan/forecast, False for actual
DESIGNER_COL = "Designer"
DEPENDENCY_COL = "Dependency"
DEFAULT_MILESTONES = ["GA", "DD", "Legal Access", "MS6"]

# Default NetSuite endpoints (yours). You can override in the sidebar.
DEFAULT_MILESTONES_URL = "https://5367938.app.netsuite.com/app/site/hosting/scriptlet.nl?script=2850&deploy=1&compid=5367938&wsId=426"
DEFAULT_DEPENDENCIES_URL = "https://5367938.app.netsuite.com/app/site/hosting/scriptlet.nl?script=2850&deploy=1&compid=5367938&wsId=618"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=15 * 60, show_spinner=True)
def fetch_netsuite_tabular(url: str, headers: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Fetch CSV/Excel/TSV from NetSuite scriptlet endpoint.
    Uses a 15‑minute TTL cache by default. Click "Refresh Data" to bypass cache.
    """
    r = requests.get(url, headers=headers or {}, timeout=60)
    r.raise_for_status()
    data = r.content

    # Try CSV → Excel → TSV
    for loader in (
        lambda b: pd.read_csv(io.BytesIO(b)),
        lambda b: pd.read_excel(io.BytesIO(b)),
        lambda b: pd.read_csv(io.BytesIO(b), sep="\\t"),
    ):
        try:
            df = loader(data)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def coerce_dates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def weekly_bucket(date: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(date):
        return date
    return (date - pd.to_timedelta(date.weekday(), unit="D")).normalize()


def compute_designer_kpis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or DESIGNER_COL not in df.columns:
        return pd.DataFrame()
    actuals = df[df.get(FORECAST_FLAG_COL, False) == False].copy()
    actuals["week"] = actuals[DATE_COL].apply(weekly_bucket)
    kpi = actuals.groupby([DESIGNER_COL, "week"]).size().reset_index(name="Actual completions")
    future = df[(df.get(FORECAST_FLAG_COL, True) == True) & (df[DATE_COL] >= pd.Timestamp.today().normalize())].copy()
    if not future.empty:
        future["week"] = future[DATE_COL].apply(weekly_bucket)
        load = future.groupby([DESIGNER_COL, "week"]).size().reset_index(name="Forecast load")
        kpi = kpi.merge(load, on=[DESIGNER_COL, "week"], how="outer").fillna(0)
    return kpi.sort_values([DESIGNER_COL, "week"]).reset_index(drop=True)


def compute_remaining_work(df: pd.DataFrame, milestone: str) -> pd.DataFrame:
    if MILESTONE_COL not in df.columns:
        return pd.DataFrame()
    sub = df[df[MILESTONE_COL] == milestone].copy()
    actuals = sub[sub.get(FORECAST_FLAG_COL, False) == False]
    planned = sub
    total_planned = planned.groupby(ID_COL).size().rename("Planned").reset_index()
    total_actual = actuals.groupby(ID_COL).size().rename("Completed").reset_index()
    out = total_planned.merge(total_actual, on=ID_COL, how="left").fillna({"Completed": 0})
    out["Remaining"] = out["Planned"] - out["Completed"]
    out.insert(1, "Milestone", milestone)
    return out


def weekly_targets_vs_actuals(df: pd.DataFrame, milestone: str) -> pd.DataFrame:
    sub = df[df[MILESTONE_COL] == milestone].copy()
    sub["week"] = sub[DATE_COL].apply(weekly_bucket)
    targets = (
        sub[sub.get(FORECAST_FLAG_COL, True) == True]
        .groupby("week")[ID_COL]
        .nunique()
        .rename("Target")
        .reset_index()
    ) if not sub.empty else pd.DataFrame(columns=["week", "Target"]) 

    actuals = (
        sub[sub.get(FORECAST_FLAG_COL, False) == False]
        .groupby("week")[ID_COL]
        .nunique()
        .rename("Actual")
        .reset_index()
    ) if not sub.empty else pd.DataFrame(columns=["week", "Actual"]) 

    out = pd.merge(targets, actuals, on="week", how="outer").sort_values("week").fillna(0)
    out.insert(1, "Milestone", milestone)
    return out


def forecast_weekly(series_df: pd.DataFrame, value_col: str = "Actual", periods: int = 6) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame(columns=["week", "Forecast", "yhat_lower", "yhat_upper"]) 
    df = series_df.rename(columns={"week": "ds", value_col: "y"})[["ds", "y"]].dropna()

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

    # Naive fallback
    last = df["y"].iloc[-1] if not df.empty else 0
    future_index = pd.date_range(pd.Timestamp.today().normalize(), periods=periods, freq="W-MON")
    return pd.DataFrame({"week": future_index, "Forecast": last})


def dependencies_weekly(df_dep: pd.DataFrame) -> pd.DataFrame:
    if df_dep.empty:
        return pd.DataFrame()
    if DATE_COL in df_dep.columns:
        df_dep["week"] = df_dep[DATE_COL].apply(weekly_bucket)
    pivot = df_dep.pivot_table(index=["week", DEPENDENCY_COL], columns=STATUS_COL, values=ID_COL, aggfunc="nunique").reset_index()
    return pivot.rename_axis(None, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("KPI Project Report")
    st.caption("Live from NetSuite – no Excel needed.")

    # Secrets-based auth header is preferred in production
    default_auth = st.secrets.get("auth", {}).get("header", "")

    milestones_url = st.text_input("Milestones URL", value=DEFAULT_MILESTONES_URL)
    dependencies_url = st.text_input("Dependencies URL", value=DEFAULT_DEPENDENCIES_URL)
    auth_header = st.text_input("Authorization header (optional)", value=default_auth, type="password")

    colA, colB = st.columns([1,1])
    with colA:
        refresh = st.button("🔄 Refresh Data", use_container_width=True)
    with colB:
        ignore_cache = st.checkbox("Ignore cache this run", value=False)

    st.markdown("---")
    st.markdown("**Filters**")
    designer_filter = st.text_input("Designer contains", value="")
    milestones_pick = st.multiselect("Milestones", DEFAULT_MILESTONES, default=DEFAULT_MILESTONES)

    st.markdown("---")
    st.markdown("**Column Mapping**")
    # Simple mapping UI so you don't have to edit code when headers change.
    map_cols = {}
    for label, default in [
        ("Internal ID", ID_COL),
        ("Template", TEMPLATE_COL),
        ("Status", STATUS_COL),
        ("Milestone", MILESTONE_COL),
        ("Date", DATE_COL),
        ("Is Forecast", FORECAST_FLAG_COL),
        ("Designer", DESIGNER_COL),
        ("Dependency", DEPENDENCY_COL),
    ]:
        map_cols[label] = st.text_input(f"{label} column", value=default, key=f"map_{label}")

# Adopt mapped names
ID_COL = map_cols["Internal ID"]
TEMPLATE_COL = map_cols["Template"]
STATUS_COL = map_cols["Status"]
MILESTONE_COL = map_cols["Milestone"]
DATE_COL = map_cols["Date"]
FORECAST_FLAG_COL = map_cols["Is Forecast"]
DESIGNER_COL = map_cols["Designer"]
DEPENDENCY_COL = map_cols["Dependency"]

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ──────────────────────────────────────────────────────────────────────────────
headers = {"Authorization": auth_header} if auth_header else None

if refresh and "_fetch_cache" in st.session_state:
    # clear cache explicitly when Refresh pressed
    fetch_netsuite_tabular.clear()

with st.spinner("Loading data from NetSuite..."):
    milestones_df = fetch_netsuite_tabular(milestones_url, headers=headers)
    dependencies_df = fetch_netsuite_tabular(dependencies_url, headers=headers)

if ignore_cache:
    fetch_netsuite_tabular.clear()
    milestones_df = fetch_netsuite_tabular(milestones_url, headers=headers)
    dependencies_df = fetch_netsuite_tabular(dependencies_url, headers=headers)

# Basic cleaning & type coercion
for df in (milestones_df, dependencies_df):
    if not df.empty:
        # try to detect a likely date column if mapping doesn't exist
        if DATE_COL not in df.columns:
            guess = next((c for c in df.columns if "date" in c.lower()), None)
            if guess:
                df.rename(columns={guess: DATE_COL}, inplace=True)
        coerce_dates(df, [DATE_COL])

# Filters
if designer_filter and not milestones_df.empty and DESIGNER_COL in milestones_df.columns:
    milestones_df = milestones_df[milestones_df[DESIGNER_COL].astype(str).str.contains(designer_filter, case=False, na=False)]

# ──────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)
if not milestones_df.empty:
    total_sites = milestones_df[ID_COL].nunique() if ID_COL in milestones_df.columns else len(milestones_df)
    total_designers = milestones_df[DESIGNER_COL].nunique() if DESIGNER_COL in milestones_df.columns else np.nan
    total_rows = len(milestones_df)
    total_actual_records = int((milestones_df.get(FORECAST_FLAG_COL, False) == False).sum())
    col1.metric("Total Sites", value=total_sites)
    col2.metric("Designers", value=total_designers)
    col3.metric("Rows", value=total_rows)
    col4.metric("Actual Records", value=total_actual_records)
else:
    st.info("No milestone data returned. Check URL/auth and column mappings.")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# Designer KPIs
st.subheader("Designer KPIs")
if not milestones_df.empty and DESIGNER_COL in milestones_df.columns and DATE_COL in milestones_df.columns:
    kpi = compute_designer_kpis(milestones_df)
    st.dataframe(kpi, use_container_width=True)
else:
    st.caption("Provide milestones with Designer & Date columns to view KPIs.")

st.divider()

# Milestones sections
st.subheader("Milestones – Remaining, Targets & Actuals, Forecasts")
if milestones_df.empty:
    st.caption("No milestones loaded.")
else:
    for m in milestones_pick:
        with st.expander(f"{m}", expanded=False):
            rem = compute_remaining_work(milestones_df, m)
            taa = weekly_targets_vs_actuals(milestones_df, m)
            st.markdown("**Remaining (by Internal ID)**")
            st.dataframe(rem, use_container_width=True)

            st.markdown("**Weekly Targets vs Actuals**")
            st.dataframe(taa, use_container_width=True)
            if not taa.empty:
                # Simple charts
                chart_df = taa.set_index("week")[ [c for c in ["Target", "Actual"] if c in taa.columns] ]
                st.line_chart(chart_df)

            st.markdown("**Recommended Forecast (next 6 weeks)**")
            fc = forecast_weekly(taa[["week", "Actual"]].dropna(), value_col="Actual", periods=6)
            st.dataframe(fc, use_container_width=True)
            if not fc.empty and "Forecast" in fc.columns:
                st.line_chart(fc.set_index("week")["Forecast"])

st.divider()

# Dependencies
st.subheader("Dependencies – Completed / Forecasted (Weekly)")
if dependencies_df.empty:
    st.caption("No dependencies loaded.")
else:
    dep = dependencies_weekly(dependencies_df)
    st.dataframe(dep, use_container_width=True)

st.divider()

with st.expander("Settings & Notes", expanded=False):
    st.markdown(
        """
**Authentication**  
Use Streamlit secrets for production: create `.streamlit/secrets.toml`:

```toml
[auth]
header = "Bearer YOUR_TOKEN_OR_COOKIE"
```

In this app, the sidebar reads `st.secrets['auth']['header']` automatically if present.

**Caching & Refresh**  
Responses are cached for 15 minutes. Click **Refresh Data** to force a new request, or tick **Ignore cache this run**.

**Column Mapping**  
Use the mapping controls in the sidebar to align NetSuite field names with the app's expectations, no code edits required.

**Forecasting**  
The app uses Prophet → ETS → Naive in that order, depending on installed packages and data history length.
        """
    )

st.success("Live NetSuite dashboard ready. Paste your URLs + auth and you’re off — no Excel required.")
