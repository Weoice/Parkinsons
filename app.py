"""Vercel FastAPI entrypoint. Static assets live next to this file (not in
public/) and are served through explicit routes so the same code path works
both under `uvicorn` locally and as a Vercel Function - a public/ directory
is promoted to a separate CDN layer on Vercel that isn't reliably readable
from the running Python process, and doesn't exist at all under plain
uvicorn."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from predict_core import predict_from_points

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

app = FastAPI(title="Parkinson Spiral Predictor")


def _serve_asset(path: str, media_type: str) -> FileResponse:
    return FileResponse(ROOT / path, media_type=media_type)


@app.get("/")
@app.get("/index.html")
def index():
    return _serve_asset("index.html", "text/html; charset=utf-8")


@app.get("/styles.css")
def styles():
    return _serve_asset("styles.css", "text/css; charset=utf-8")


@app.get("/app.js")
def script():
    return _serve_asset("app.js", "application/javascript; charset=utf-8")


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
