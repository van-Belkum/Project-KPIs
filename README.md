# KPI Project Report (NetSuite Live)

A Streamlit dashboard that reads **directly from NetSuite** (no Excel) to show:
- Designer KPIs
- Remaining GA/DD/Legal Access/MS6
- Weekly Targets vs Actuals
- Dependencies (Completed / Forecasted)
- 6-week **forecast recommendations** (Prophet → ETS → Naive)

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Configure NetSuite
The app sidebar has fields for your NetSuite scriptlet URLs. If auth is required, use a token or cookie in the Authorization header.

### Secrets (recommended)
Create `.streamlit/secrets.toml`:
```toml
[auth]
header = "Bearer YOUR_TOKEN_OR_COOKIE"
```

> Do **not** commit secrets. Use Streamlit Cloud's Secrets manager or environment variables in your host.

## Secure sharing options
- **Streamlit Community Cloud (private)**: invite specific emails. Add your auth header in Secrets.
- **Self‑host behind SSO**: run on a VM/container and protect with Cloudflare Access, Auth0, Okta, Azure AD, or Google IAP.
- **Reverse proxy & TLS**: terminate HTTPS at Nginx/Caddy; only allow your office IPs or VPN if desired.
- **Principle of least privilege**: use a NetSuite token scoped to read‑only for these endpoints; rotate regularly.

## Environment variables (alternative to secrets)
Set `AUTH_HEADER` and read it in `app.py` with `os.environ.get("AUTH_HEADER")` if you prefer env vars.

## Notes
- Caching: responses cached for 15 min; use 'Refresh Data' to bypass.
- Forecasting requires history. If Prophet can't fit, ETS or a naive forecast will be used.
