from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class PredictRequest(BaseModel):
    # Agar fleksibel, terima dict bebas (raw features) — cocok dengan pipeline yang handle missing/unknown
    features: Dict[str, Any] = Field(..., description="Key-value pasangan kolom raw seperti pada data training")

class PredictResponse(BaseModel):
    prediction: float
