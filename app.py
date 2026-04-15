from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Prediction API", version="1.0.0")

MODEL_PATH = "model/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé avec succès")
    else:
        print("⚠️  Modèle non trouvé, entraînez d'abord le modèle")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: int
    label: str
    confidence: float

LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction])

    return PredictionOutput(
        prediction=int(prediction),
        label=LABELS[prediction],
        confidence=round(confidence, 4)
    )
