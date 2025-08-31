
# Telco Customer Churn — Streamlit

## Quick start (local)

1. Ensure Python 3.9+ is installed.
2. Create a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Put your dataset file in the same folder, e.g.: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
5. Run the app:
   ```bash
   streamlit run app.py
   ```
6. Your browser will open at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (via GitHub)

1. Create a new GitHub repo and push at least these files:
   - `app.py`
   - `requirements.txt`
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv` (or host it elsewhere and modify the code)
2. Go to https://streamlit.io/cloud, sign in, and **New app** → pick your repo/branch/file `app.py`.
3. Click **Deploy**. Subsequent pushes to GitHub will auto-redeploy.

## Notes

- The app expects the label/target column to be `Churn` by default. Change it in the sidebar if yours differs.
- If your notebook has custom feature engineering or models, port those steps into `build_pipeline(...)`.
