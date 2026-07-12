from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import numpy as np
import joblib
import os
import csv
import time
import json
import subprocess
import threading
import time
from datetime import datetime
from schemas import HouseInput
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

sample_df = pd.read_csv("train.csv")
default_features= sample_df.drop("SalePrice", axis=1).iloc[0].to_dict()

BEST_MODEL_PATH = "best_model.pkl"
ALL_MODELS_PATH = "all_models.pkl"
LOG_FILE = "prediction_logs.csv"

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

from schemas import HouseInput

# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/Health")

def health():

    return {
        "status": "Healthy",
        "time": datetime.now(),
        "model_loaded": best_model is not None
    }

# ======================================================
# AVAILABLE MODELS
# ======================================================

@app.get("/Models")

def models():

    return {
        "Available Models": list(all_models.keys())
    }

# ======================================================
# MODEL INFORMATION
# ======================================================

@app.get("/Model-Info")

def model_info():

    if best_model is None:
        raise HTTPException(
            status_code=404,
            detail="No model loaded."
        )

    return {

        "Model Name": "Random Forest Regressor",

        "Model Type": str(type(best_model)),

        "Loaded": "True",
       
        "Framework": "Regression",
 
        "Status": "Loaded"

    }

# ======================================================
# CHANGE MODEL
# ======================================================

from pydantic import BaseModel

class ModelSelection(BaseModel):
      model_name: str

# ======================================================
# SINGLE PREDICTION
# ======================================================

@app.post("/Predict")

def predict(data: HouseInput):

    global best_model

    if best_model is None:

        raise HTTPException(
            status_code=404,
            detail="Model not loaded."
        )

    start = time.time()

    features = default_features.copy()
    features.update(data.features)
    pipeline = joblib.load(BEST_MODEL_PATH)
    
    input_df = pd.DataFrame([features])

    input_df["TotalSF"] = (input_df["TotalBsmtSF"] + input_df["1stFlrSF"] + input_df["2ndFlrSF"])

    input_df["TotalBathrooms"] = (input_df["FullBath"] + 0.5 * input_df["HalfBath"] + input_df["BsmtFullBath"] + 0.5 * input_df["BsmtHalfBath"])

    input_df["HouseAge"] = (input_df["YrSold"] - input_df["YearBuilt"])

    input_df["RemodAge"] = (input_df["YrSold"] - input_df["YearRemodAdd"])

    input_df["TotalPorchSF"] = (input_df["OpenPorchSF"] + input_df["EnclosedPorch"] + input_df["3SsnPorch"] + input_df["ScreenPorch"])

    prediction = best_model.predict(input_df)

    latency = round((time.time()-start)*1000,2)

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
         writer = csv.writer(f)
         if not file_exists:
  
                writer.writerow(["Timestamp", "LotArea", "OverallQual", "TotalBsmtSF", "YearBuilt", "Prediction", "Latency(ms)"])
       
         writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), features["LotArea"], features["OverallQual"], features["TotalBsmtSF"], features["YearBuilt"], float(prediction[0]), round(latency,2)])
    
    return{

        "Predicted Price":float(prediction[0]),

        "Model":selected_model,

        "Latency(ms)":latency

    }

# ======================================================
# LOGS
# ======================================================

@app.get("/Logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
           raise HTTPException(status_code=404, detail="No logs found.")

    logs = []

    with open(LOG_FILE, "r") as f:
         reader = csv.DictReader(f)

         for row in reader:
             logs.append(row)

    return {
         "Total Logs": len(logs),
         "Logs": logs
    }

# ======================================================
# BATCH PREDICTION
# ======================================================

@app.post("/Batch-Predict")

def batch_predict(data:list[HouseInput]):

    global best_model

    if best_model is None:

        raise HTTPException(
            status_code=404,
            detail="Model not loaded."
        )
    
    try:
        rows = []
        for item in data:
            features = default_features.copy()
            features.update(item.features)
            rows.append(features)

        df=pd.DataFrame(rows)

        df["TotalSF"] = (df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"])

        df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"])

        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"])

        df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"])

        predictions=best_model.predict(df)

        return{

               "Total Predictions":len(predictions),

               "Predictions":[float(x) for x in predictions]

               }

    except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# MODEL METRICS
# ======================================================

@app.get("/Metrics")
def metrics():

    if not os.path.exists("metrics.json"):
           raise HTTPException(status_code=404, details="Metrics file not found.")
 
    with open("metrics.json", "r") as f:
         metrics = json.load(f)

    return metrics

# ======================================================
# DATA DRIFT REPORT
# ======================================================

@app.get("/Drift")
def drift_report():

    if not os.path.exists("drift_report.json"):
           raise HTTPException(status_code=404, detail="Drift report not found.")

    with open("drift_report.json", "r") as f:
         report = json.load(f)

    return report

# ======================================================
# MONITORING DASHBOARD
# ======================================================

@app.get("/Monitor")
def monitor():
    global best_model

    if not os.path.exists(LOG_FILE):
           raise HTTPException(status_code=404, detail = "Prediction log file not found.")

    logs = pd.read_csv(LOG_FILE)

    total_predictions = len(logs)

    if total_predictions == 0:
       return {
            "Status": "Running", 
            "Model Loaded": best_model is not None,
            "Total Predictions": 0,
            "Average Prediction": 0,
            "Average Latency (ms)": 0,
            "Maximum Prediction": 0,
            "Minimum Prediction": 0
            }

    avg_prediction = logs["Prediction"].mean()
    max_prediction = logs["Prediction"].max()
    min_prediction = logs["Prediction"].min()
    avg_latency = logs["Latency(ms)"].mean()

    return {
            "Status": "Running",
          
            "Model Loaded": best_model is not None,

            "Model": "Random Forest Regressor",
 
            "Total Predictions": int(total_predictions),
 
            "Average Predictions": round(avg_prediction, 2),
 
            "Maximum Prediction": round(min_prediction, 2),

            "Minimum Prediction": round(min_prediction, 2),

            "Average Latency (ms)": round(avg_latency, 2)
           }

# ======================================================
# RETRAIN MODEL
# ======================================================

def run_training():
 
    try:
         subprocess.run(["python","train_models.py"], check=True)
 
         print("Model Retrained successfully.")

    except Exception as e:

           print(f"Retraining failed: {e}")

@app.post("/Retrain")
def retrain_model():

    thread = threading.Thread(target=run_training)
    thread.start()

    return {
            "Status": "Retraining Started",

            "Time": datetime.now().strftime("%Y-%m-%d %H:%H:%S"),

            "Messege": "Model retraining is running in the background."
            }

# ======================================================
# ROOT
# ======================================================

@app.get("/Home")

def home():

    return{

        "Project":"House Price Prediction",

        "Framework":"FastAPI",

        "Status":"Running"

    }