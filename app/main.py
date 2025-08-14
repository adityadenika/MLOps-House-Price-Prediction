import time, os, joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from app.schemas import PredictRequest, PredictResponse
from src.utils.logger import get_logger

ARTIFACT_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

log = get_logger("api")
app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(ARTIFACT_PATH):
            raise FileNotFoundError(f"Model artifact not found: {ARTIFACT_PATH}")
        _model = joblib.load(ARTIFACT_PATH)
        log.info("Model loaded", extra={"extra":{"path": ARTIFACT_PATH}})
    return _model

@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    start = time.time()
    try:
        model = load_model()
        # Model menerima DataFrame dengan 1 baris fitur mentah
        import pandas as pd
        X = pd.DataFrame([req.features])
        y_pred = float(model.predict(X)[0])
        latency_ms = round((time.time() - start) * 1000, 2)
        log.info("Prediction served", extra={"extra":{
            "client": request.client.host if request.client else None,
            "latency_ms": latency_ms
        }})
        return PredictResponse(prediction=y_pred)
    except Exception as e:
        log.error("Prediction error", extra={"extra":{"error": str(e)}})
        raise HTTPException(status_code=400, detail=str(e))
