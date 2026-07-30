"""Vercel FastAPI entrypoint. Static UI is served from public/ by the CDN."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from predict_core import predict_from_points

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

app = FastAPI(title="Parkinson Spiral Predictor")


@app.get("/")
def root():
    return FileResponse(ROOT / "public" / "index.html")


class PredictRequest(BaseModel):
    points: list = Field(default_factory=list)


@app.post("/api/predict")
def predict(body: PredictRequest):
    try:
        return predict_from_points(body.points, ARTIFACT_DIR)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        return JSONResponse(
            {"error": "Prediction failed. Try a longer continuous trace."},
            status_code=500,
        )


@app.get("/api/health")
def health():
    return {"ok": True}
