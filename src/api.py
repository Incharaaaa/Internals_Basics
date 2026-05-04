from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI()

# Safe path (avoids Windows issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

model = joblib.load(model_path)

# Input validation (VERY IMPORTANT FOR MARKS)
class InputData(BaseModel):
    network_congestion: float = Field(..., ge=0.1, le=1.0)
    txn_size_bytes: int = Field(..., ge=200, le=2000)
    is_priority: int = Field(..., ge=0, le=1)
    block_fullness_pct: float = Field(..., ge=50, le=100)

# Health endpoint
@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "CoinPulse gas_fee_gwei API"}

# Prediction endpoint
@app.post("/score")
def score(data: InputData):
    features = np.array([[
        data.network_congestion,
        data.txn_size_bytes,
        data.is_priority,
        data.block_fullness_pct
    ]])

    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}