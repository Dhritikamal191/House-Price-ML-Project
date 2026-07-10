import pandas as pd
import json

# Reference (training) data
reference = pd.read_csv("Housing.csv")

# Current production data
current = pd.read_csv("prediction_logs.csv")

# Example drift calculation
drift_score = abs(
    reference["LotArea"].mean()
    - current["lot_area"].mean()
)

if drift_score > 1000:
    status = "Retraining Recommended"
    drift = "Yes"
else:
    status = "Stable"
    drift = "No"

drift_result = {
    "Data Drift": drift,
    "Drift Score": round(float(drift_score), 2),
    "Status": status
}

with open("drift_report.json", "w") as f:
    json.dump(drift_result, f, indent=4)

print("drift_report.json created successfully.")