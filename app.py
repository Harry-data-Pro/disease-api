from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# ---------- load artefacts ----------
model = joblib.load("model/disease_model.pkl")
label = joblib.load("model/label_encoder.pkl")
FEATURES = list(model.feature_name_)       # 132 symptom names

# ---------- request schema ----------
class SymptomInput(BaseModel):
    symptoms: dict   # e.g. {"itching":1, "skin_rash":0}

# ---------- app ----------
app = FastAPI(title="Disease-Predictor")

@app.post("/predict")
def predict(item: SymptomInput):
    # build 0/1 feature vector
    row = pd.Series(0, index=FEATURES, dtype=int)
    for k, v in item.symptoms.items():
        if k not in FEATURES:
            raise HTTPException(400, f"Unknown symptom: {k}")
        row[k] = int(bool(v))
    prob = model.predict_proba([row])[0]
    idx  = prob.argmax()
    return {
        "disease": label.inverse_transform([idx])[0],
        "probability": float(prob[idx])
    }

# local dev:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
