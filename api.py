from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import time
from datetime import datetime

# ======================================================
# APP
# ======================================================

app = FastAPI(
    title="House Price Prediction API",
    description="Production Ready House Price Prediction API",
    version="1.0"
)

# ======================================================
# LOAD MODELS
# ======================================================

BEST_MODEL_PATH = "models/best_model.pkl"
ALL_MODELS_PATH = "models/all_models.pkl"

try:
    best_model = joblib.load(BEST_MODEL_PATH)
except:
    best_model = None

try:
    all_models = joblib.load(ALL_MODELS_PATH)
except:
    all_models = {}

selected_model = "Best Model"

# ======================================================
# INPUT SCHEMA
# ======================================================

class PredictionInput(BaseModel):
    features: Dict

# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/health")

def health():

    return {
        "status": "Healthy",
        "time": datetime.now(),
        "model_loaded": best_model is not None
    }

# ======================================================
# AVAILABLE MODELS
# ======================================================

@app.get("/models")

def models():

    return {
        "Available Models": list(all_models.keys())
    }

# ======================================================
# MODEL INFORMATION
# ======================================================

@app.get("/model-info")

def model_info():

    if best_model is None:
        raise HTTPException(
            status_code=404,
            detail="No model loaded."
        )

    return {

        "Current Model": selected_model,

        "Model Type": str(type(best_model)),

        "Loaded": True

    }

# ======================================================
# CHANGE MODEL
# ======================================================

@app.post("/select-model")

def select_model(model_name:str):

    global best_model
    global selected_model

    if model_name not in all_models:

        raise HTTPException(
            status_code=404,
            detail="Model not found."
        )

    best_model = all_models[model_name]

    selected_model = model_name

    return {

        "message":"Model changed successfully.",

        "Current Model":selected_model

    }

# ======================================================
# SINGLE PREDICTION
# ======================================================

@app.post("/predict")

def predict(data:PredictionInput):

    global best_model

    if best_model is None:

        raise HTTPException(
            status_code=404,
            detail="Model not loaded."
        )

    start = time.time()

    df = pd.DataFrame([data.features])

    prediction = best_model.predict(df)

    latency = round((time.time()-start)*1000,2)

    return{

        "Predicted Price":float(prediction[0]),

        "Model":selected_model,

        "Latency(ms)":latency

    }

# ======================================================
# BATCH PREDICTION
# ======================================================

@app.post("/batch-predict")

def batch_predict(data:list):

    global best_model

    if best_model is None:

        raise HTTPException(
            status_code=404,
            detail="Model not loaded."
        )

    df=pd.DataFrame(data)

    predictions=best_model.predict(df)

    return{

        "Total Predictions":len(predictions),

        "Predictions":[float(i) for i in predictions]

    }

# ======================================================
# RETRAIN MODEL
# ======================================================

@app.post("/retrain")

def retrain():

    subprocess.Popen(

        ["python",".streamlit/train_models.py"]

    )

    return{

        "Status":"Retraining Started"

    }

# ======================================================
# ROOT
# ======================================================

@app.get("/")

def home():

    return{

        "Project":"House Price Prediction",

        "Framework":"FastAPI",

        "Status":"Running"

    }