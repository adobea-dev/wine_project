"""
Wine Quality Prediction API
---------------------------
Serves a trained multiclass model for wine quality prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
from pathlib import Path
import json
import logging

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FastAPI App ===
app = FastAPI(
    title="Wine Quality Classification API",
    description="Predicts wine quality category (Low / Medium / High) using the champion ML model.",
    version="1.1.0"
)

# === Paths ===
MODEL_PATH = Path("models/champion.joblib")          # Best (champion) model
METADATA_PATH = Path("models/champion_metadata.json") 

# === Global Variables ===
model = None
metadata = {
      "champion_model_name": best_model_name,
    "champion_model_class": champion_model.__class__.__name__,
    "training_timestamp": str(pd.Timestamp.now()),
    "selected_metric": selected_metric,  
    "metrics": evaluation_results[best_model_name],
    "feature_count": X_train.shape[1],
    "label_map": {0: "Low", 1: "Medium", 2: "High"}
}

# === Load model function ===
def load_model():
    global model, metadata
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Loaded champion model: {MODEL_PATH.name}")
        if METADATA_PATH.exists():
            with open("models/champion_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                metadata = json.load(f)
                logger.info("Loaded model metadata.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

# Load model on startup
load_model()

# === Input schemas ===
class WineFeatures(BaseModel):
    """Schema for single wine sample input."""
    features: Dict[str, float]

class BatchInput(BaseModel):
    """Schema for batch predictions."""
    records: List[WineFeatures]

# === Label mapping ===
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

# === Health check ===
@app.get("/")
def root():
    return {
        "message": "Wine Quality Prediction API is running",
        "model_loaded": model is not None,
        "metadata": metadata if metadata else "No metadata found"
    }

# === Reload model endpoint (optional) ===
@app.post("/reload_model")
def reload_model():
    """Reloads the champion model after retraining."""
    load_model()
    if model:
        return {"status": "success", "message": "Model reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model.")

# === Predict single ===
@app.post("/predict")
def predict_single(input_data: WineFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        df = pd.DataFrame([input_data.features])
        pred = int(model.predict(df)[0])
        label = LABEL_MAP.get(pred, "Unknown")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0].tolist()
            proba_dict = {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(proba)}
        else:
            proba_dict = None

        return {
            "prediction": label,
            "probabilities": proba_dict,
            "raw_class": pred
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# === Predict batch ===
@app.post("/predict_batch")
def predict_batch(input_batch: BatchInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        df = pd.DataFrame([rec.features for rec in input_batch.records])
        preds = model.predict(df).tolist()
        labels = [LABEL_MAP.get(int(p), "Unknown") for p in preds]

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(df).tolist()
            probas_named = [
                {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(prob_row)}
                for prob_row in probas
            ]
        else:
            probas_named = None

        return {
            "predictions": labels,
            "probabilities": probas_named,
            "raw_classes": preds
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
