import json
import subprocess

with open("data/drift_report.json", "r") as f:
     drift = json.load(f)

if drift["Data Drift"] == "Yes":
   print("Drift detected. Retraining model...")
   subprocess.run(["python","train_models.py"])

else:
     print("No drift detected. Retraining not required.")
