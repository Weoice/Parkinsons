# Parkinson's Spiral Prediction

Research demo: draw a spiral in the browser; a kinematic XGBoost model returns a probability score.

**Not a medical device.**

## Deploy (Vercel)

- `app.py` — FastAPI entrypoint (`app`)
- `public/` — static UI (CDN)
- `artifacts/` — trained model + scaler
- `requirements.txt` — runtime deps (Python 3.12 on Vercel)

```bash
vercel
```

## Local API check

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `public/index.html` via `vercel dev` so `/api/predict` works end-to-end.

## Retrain artifacts

```bash
python scripts/export_artifacts.py
```

Requires the UCI hw_dataset path configured in that script.
