# Deploying to Streamlit Community Cloud

Follow these quick steps to publish this app while keeping developer tooling hidden:

1. Push your latest changes to GitHub (any public or private repo works as long as Streamlit Cloud can access it).
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **New app** → connect the GitHub repo.
3. Choose the main branch and set the entry point to `ui/streamlit_app.py`.
4. In **Advanced settings** (Secrets / Environment variables), add at minimum:
   - `INVEST_AI_ENV=production`
   - `INVEST_AI_DEMO=1` *(forces cached/offline data so the app boots without external API keys)*
   - Optional: provider keys such as `TIINGO_API_KEY` and `FRED_API_KEY` if you want live updates instead of the offline demo snapshots.
5. Confirm the runtime uses the repo’s `requirements.txt` (Python 3.11 works best).
6. Click **Deploy**.

Notes:
- Locally you should **not** set `INVEST_AI_ENV`, so all diagnostics and dev tooling stay visible; keep `INVEST_AI_DEMO=0` to use live data.
- On Streamlit Cloud the combination of `INVEST_AI_ENV=production` and `INVEST_AI_DEMO=1` hides diagnostics and guarantees no external API calls unless you explicitly provide keys.
- The offline snapshots live under `data/cache/` and `data/macro/`; refresh them locally before deploying if you want a more recent demo experience.
