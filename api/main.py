"""
API REST para inferência de diabetes.
Rode com: uvicorn api.main:app --reload
Acesse: http://localhost:8000/docs
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Diabetes Prediction API",
    description="API para previsao de diabetes usando pipeline Kedro - PADS Insper",
    version="1.0.0",
)

MODEL_PATH = "data/06_models/best_model.pkl"
SCALER_PATH = "data/06_models/scaler.pkl"
COLUMNS_PATH = "data/06_models/feature_columns.pkl"

_model = None
_scaler = None
_feature_columns = None


def load_artifacts():
    global _model, _scaler, _feature_columns
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            _feature_columns = joblib.load(COLUMNS_PATH)
        except FileNotFoundError:
            raise RuntimeError("Execute 'kedro run' antes de subir a API.")


class PatientData(BaseModel):
    Pregnancies: float = Field(..., example=2)
    Glucose: float = Field(..., example=120.0)
    BloodPressure: float = Field(..., example=70.0)
    SkinThickness: float = Field(..., example=20.0)
    Insulin: float = Field(..., example=80.0)
    BMI: float = Field(..., example=28.5)
    DiabetesPedigreeFunction: float = Field(..., example=0.5)
    Age: float = Field(..., example=33)


class PredictionResult(BaseModel):
    predicted_label: int
    predicted_proba: float
    interpretation: str


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientData):
    load_artifacts()

    from diabetes_prediction.pipelines.inference.nodes import preprocess_inference

    df = pd.DataFrame([patient.dict()])
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    try:
        df_processed = preprocess_inference(df, _scaler, _feature_columns, zero_cols)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    proba = float(_model.predict_proba(df_processed)[:, 1][0])
    label = int(proba >= 0.5)
    interpretation = (
        "Alta probabilidade de diabetes" if proba >= 0.7
        else "Risco moderado" if proba >= 0.4
        else "Baixa probabilidade de diabetes"
    )

    return PredictionResult(
        predicted_label=label,
        predicted_proba=round(proba, 4),
        interpretation=interpretation,
    )
