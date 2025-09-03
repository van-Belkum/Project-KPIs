# KPI Project Report – Pro (NetSuite, Password-Protected)

A professional Streamlit dashboard that connects directly to NetSuite (no Excel), shows **Status-style progress**, **Designer KPIs**, **Weekly Targets vs Actuals**, **Dependencies**, and **predictive forecasts** for each milestone.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud (public app with password)
1. Go to **App → Settings → Secrets**, add:
```toml
[app]
# sha256 of your chosen password
password_sha256 = "<paste_hash_here>"

[auth]
# NetSuite token or Cookie header, if required by your endpoints
header = "Bearer YOUR_TOKEN_OR_COOKIE"
```
2. To generate the password hash locally:
```python
import hashlib; print(hashlib.sha256(b"YOUR_PASSWORD").hexdigest())
```

## Configure endpoints in the app sidebar
- Milestones URL (prefilled with your wsId=426)
- Dependencies URL (prefilled with your wsId=618)
- Authorization header auto-reads from secrets; you can override/paste it in the sidebar.

## Completion logic
Choose which **Status** values count as a completed **Actual** (defaults: Accepted, Completed, Actual).

## Forecasting
Uses **Prophet → ETS → Naive** in that order depending on installed packages and history length.

## Notes
- Responses are cached for 15 minutes.
- No Excel required; you can still extend with a file uploader if needed.
