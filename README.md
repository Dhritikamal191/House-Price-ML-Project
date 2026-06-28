[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://github.com/Dhritikamal191/House-Price-ML-Project/blob/main/Dockerfile)

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)](https://house-price-ml-project-olkeyehfarfpdwo86p7iwk.streamlit.app/)

[![Supabase](https://img.shields.io/badge/Supabase-Prediction%20Logging-green)](https://supabase.com/dashboard/project/mfoauekyjbkppdyxwzfu/editor/17588)

# 🏡 House Price Prediction using Machine Learning & MLOps

### Overview

This project is an end-to-end Machine Learning and MLOps application that predicts residential house prices using the Ames Housing dataset. It combines advanced feature engineering, multiple regression models, experiment tracking, prediction logging, and an interactive Streamlit dashboard to simulate a production-ready ML system.

---

### Features

- Interactive Streamlit Web Application
- Multiple Machine Learning Models
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
- Automatic Feature Engineering
- MLflow Experiment Tracking
- Prediction Logging using Supabase
- Docker Containerization
- GitHub Actions Continuous Integration
- Real-time Monitoring Dashboard
- CSV Export of Prediction Logs

---

### Tech Stack

#### Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

#### Visualization

- Plotly
- Matplotlib
- Streamlit

#### MLOps

- MLflow
- Docker
- GitHub Actions
- Supabase

---

### Dataset

Ames Housing Dataset

- 1,460 Houses
- 80+ Features
- Numerical and Categorical Variables
- Extensive Feature Engineering

---

### Feature Engineering

Additional engineered features include:

- TotalSF
- TotalBathrooms
- HouseAge
- RemodAge
- TotalPorchSF

These engineered features improve model performance and capture important property characteristics.

---

### Machine Learning Models

Model| Purpose
Linear Regression| Baseline Model
Ridge Regression| Regularized Linear Model
Lasso Regression| Feature Selection & Regularization
Random Forest| Ensemble Tree-Based Model

The best-performing model is automatically saved for deployment.

---

### MLflow Experiment Tracking

Each training run logs:

- Model Parameters
- Training Metrics
- RMSE
- MAE
- R² Score
- Model Artifacts

---

### Prediction Monitoring

Every prediction is automatically stored in Supabase with:

- Timestamp
- Selected Model
- Predicted Price
- Lot Area
- Year Built
- House Age
- Overall Quality
- Garage Capacity
- Ground Living Area
- Basement Area
- Prediction Latency
- Prediction Status
- Error Message (if any)

---

### Monitoring Dashboard

The Streamlit dashboard includes:

- Total Predictions
- Average Predicted Price
- Highest Prediction
- Lowest Prediction
- Predictions Today
- Predictions This Week
- Average Lot Area
- Average House Age
- Most Frequently Used Model
- Failed Predictions
- Average Prediction Latency
- Prediction Trend
- Model Usage Analysis
- Download Prediction Logs (CSV)

---

### Project Structure

House-Price-Prediction/
│
├── data/
├── models/
├── app.py
├── train_models.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .streamlit/
├── .github/
│   └── workflows/
├── mlruns/
├── mlflow.db
└── prediction_logs.csv

---

### Installation

git clone <repository-url>

cd House-Price-Prediction

pip install -r requirements.txt

streamlit run app.py

---

### Docker

docker build -t house-price-app .

docker run -p 8501:8501 house-price-app

---

### CI/CD

GitHub Actions automatically:

- Installs dependencies
- Validates Python files
- Ensures application integrity

---

### Future Improvements

- FastAPI REST API
- Cloud MLflow Deployment
- Kubernetes Deployment
- Model Drift Monitoring
- Automated Retraining Pipeline
- Cloud Storage Integration

---

### Author

Dhritikamal Das

M.Sc. MACS | Data Analyst | Machine Learning Enthusiast

---

### License

This project is intended for educational and portfolio purposes.