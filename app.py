from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Digits Prediction API", version="1.0.0")

MODEL_PATH = "model/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé avec succès")
    else:
        print("Modèle non trouvé, entraînez d'abord le modèle")

class DigitsInput(BaseModel):
    pixels: list[float]  # 64 valeurs (image 8x8)

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: DigitsInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(data.pixels) != 64:
        raise HTTPException(status_code=400, detail="Il faut exactement 64 valeurs de pixels (image 8x8)")

    features = np.array(data.pixels).reshape(1, -1)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction])

    return PredictionOutput(
        prediction=int(prediction),
        confidence=round(confidence, 4)
    )
