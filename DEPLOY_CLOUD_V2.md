# Cloud Deployment (V2)

This repo targets **Python 3.11.x** everywhere. Platforms such as Streamlit Community Cloud, Render, and Heroku-style builders read `runtime.txt` to decide which interpreter to pre-provision. If the builder ignores that file, explicitly select Python 3.11 in its UI to avoid pandas / SciPy compilation attempts under Python 3.13.

## How cloud builds work

1. **Dependency resolution vs installation**
   - Cloud logs that say _"Resolved 81 packages"_ are the resolver planning step. Pip still needs matching **binary wheels** for the target CPU/OS + Python version.
   - If a wheel exists (e.g., pandas for Linux + Python 3.11), pip downloads it and moves on. If no wheel exists (common for brand-new CPython releases like 3.13), pip falls back to source tarballs and compiles C/C++/Fortran code.
2. **Why builds fail on Python 3.13**
   - Heavy packages here (`pandas`, `numpy`, `scipy`, `cvxpy`, solvers such as `scs`, `osqp`, `clarabel`) still ship CPython 3.13 wheels slowly. When a platform insists on 3.13, the build image must compile them from source and often lacks compilers or BLAS headers, leading to failures while "building wheel for pandas" or "building wheel for scipy".
3. **Canonical runtime**
   - `runtime.txt` now pins `python-3.11.8`. Treat 3.11 as the supported interpreter for both local dev and cloud deploys. Python 3.12/3.13 may work later once upstream wheels stabilize.

## Local pre-deploy checklist

Run every time before pushing:

```bash
./deploy_checks.sh
```

The script performs:
1. `pip install -r requirements.txt`
2. `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`
3. `PYTHONPATH=. .venv/bin/python -c "import ui.streamlit_app; print('streamlit_app import OK')"`

On success you can `git push origin main` and trigger your cloud deploy. On failure, fix the issue locally before retrying.

## Streamlit Community Cloud

1. Push the branch (usually `main`) with the updated files.
2. In the Streamlit Cloud dashboard, click **New app** and point it to this repo + branch.
3. Set the entry point to `ui/streamlit_app.py`.
4. Open **Advanced settings → Secrets** and define at least:
   - `INVEST_AI_ENV=production`
   - `INVEST_AI_DEMO=1` (forces cached demo datasets; set to `0` only if you provide live API keys)
   - Optional provider keys: `TIINGO_API_KEY`, `FRED_API_KEY`, `POLYGON_API_KEY`
5. Confirm the **Python version** dropdown is 3.11 (if editable) so it matches `runtime.txt`.
6. Deploy. A healthy log will mention `Using Python 3.11.x` followed by cached wheels for pandas/numpy/scipy instead of C compilation.
7. After the first boot, open the app and verify: navigation sidebar, macro panel, and at least one per-ticker receipt under the debug expander.

## Render Web Service (free tier)

1. Create a **Web Service** in Render pointing to this repo/branch.
2. Environment → select **Python 3.11** (or let Render read `runtime.txt`).
3. Set **Build Command**:
   - `pip install -r requirements.txt`
4. Set **Start Command**:
   - `streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Configure the same environment variables/secrets as Streamlit Cloud: `INVEST_AI_ENV=production`, `INVEST_AI_DEMO=1`, optional provider keys.
6. Trigger a deploy. Render tail logs should show dependency installation without compilation errors and a final `Streamlit Running on http://0.0.0.0:$PORT` message.

## FAQ

**What does “Resolved 81 packages” mean?**  
Pip completed dependency graph calculation but has not downloaded wheels yet. It still needs to fetch or build each package for the selected Python version.

**Why do pandas/scipy sometimes fail to build on Python 3.13 but work on 3.11?**  
Because upstream teams have released prebuilt wheels for 3.11 on Linux, so pip downloads them instantly. For 3.13, many wheels are missing; pip tries to compile from source, which requires system compilers and BLAS/Fortran libraries that lightweight cloud builders often omit.

**Do I need to change application code to fix this?**  
No. Keep the business logic as-is. Just pin the runtime to Python 3.11, run `./deploy_checks.sh`, and ensure your cloud platform honors `requirements.txt` + `runtime.txt`.
