# ==========================================
# IMPORT LIBRARIES
# ==========================================
import os
import logging
import warnings
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATASET
# ==========================================
with open("config.yaml","r") as f:
     config = yaml.safe_load(f)

print(config)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Training Started")

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv(config["data"]["train_path"])

RANDOM_STATE = config["model"]["random_state"]

TEST_SIZE = config["model"]["test_size"]

LASSO_ALPHA = config["lasso"]["alpha"]
LASSO_MAX_ITER=  config["lasso"]["max_iter"]

RF_N_ESTIMATORS = config["random_forest"]["n_estimators"]
RF_N_JOBS = config["random_forest"]["n_jobs"]

RIDGE_ALPHA = config["ridge"]["alpha"]

LINEAR_N_JOBS = config["linear"]["n_jobs"]

print("\nDataset Loaded Successfully!")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# DATA EXPLORATION
# ==========================================

print("\n" + "=" * 50)
print("DATA EXPLORATION")
print("=" * 50)

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values (Top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# ==========================================
# TARGET VARIABLE ANALYSIS
# ==========================================

print("\nTarget Variable: SalePrice")

print("\nSalePrice Statistics:")
print(df["SalePrice"].describe())

print("\nSkewness:",df["SalePrice"].skew())

print("Kurtosis:",df["SalePrice"].kurt())

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# Total Square Feet
df["TotalSF"] = (
    df["TotalBsmtSF"]
    + df["1stFlrSF"]
    + df["2ndFlrSF"]
)

# Total Bathrooms
df["TotalBathrooms"] = (
    df["FullBath"]
    + (0.5 * df["HalfBath"])
    + df["BsmtFullBath"]
    + (0.5 * df["BsmtHalfBath"])
)

# House Age
df["HouseAge"] = (
    df["YrSold"]
    - df["YearBuilt"]
)

# Remodeling Age
df["RemodAge"] = (
    df["YrSold"]
    - df["YearRemodAdd"]
)

# Total Porch Area
df["TotalPorchSF"] = (
    df["OpenPorchSF"]
    + df["EnclosedPorch"]
    + df["3SsnPorch"]
    + df["ScreenPorch"]
)

print("\nFeature Engineering Completed!")

print("\nNew Features Added:")
print([
    "TotalSF",
    "TotalBathrooms",
    "HouseAge",
    "RemodAge",
    "TotalPorchSF"
])

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= TEST_SIZE,random_state= RANDOM_STATE)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================
numeric_defaults = X_train.select_dtypes(include="number").median()

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns

categorical_features = X_train.select_dtypes(include=["object"]).columns

print("\nNumber of Numerical Features:",len(numeric_features))

print("Number of Categorical Features:",len(categorical_features))

joblib.dump(numeric_defaults,config["data"]["numeric_default"])

# ==========================================
# PREPROCESSING PIPELINES
# ==========================================

numeric_transformer = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler", StandardScaler())])

categorical_transformer = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

preprocessor = ColumnTransformer([("num",numeric_transformer,numeric_features),("cat",categorical_transformer,categorical_features)])

print("\nPreprocessing Pipeline Created Successfully!")

# ==========================================
# IMPORT MODELS AND METRICS
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

# ==========================================
# DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(n_jobs=LINEAR_N_JOBS),

    "Ridge Regression": Ridge(alpha=RIDGE_ALPHA,random_state= RANDOM_STATE),

    "Lasso Regression": Lasso(alpha= LASSO_ALPHA,random_state=RANDOM_STATE,max_iter= LASSO_MAX_ITER),

    "Random Forest": RandomForestRegressor(n_estimators= RF_N_ESTIMATORS,random_state= RANDOM_STATE,n_jobs= RF_N_JOBS)
}

# ==========================================
# MODEL TRAINING & EVALUATION
# ==========================================

print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

results = {}

trained_models = {}

best_model_name = None
best_pipeline = None
best_r2 = -np.inf

mlflow.set_experiment("Housing Price Prediction")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

         print(f"\nTraining {name}...")

         pipeline = Pipeline([("preprocessor", preprocessor),("model", model)])

         pipeline.fit(X_train, y_train)

         predictions = pipeline.predict(X_test)

         mae = mean_absolute_error(y_test,predictions)

         mse = mean_squared_error(y_test,predictions)

         rmse = np.sqrt(mean_squared_error(y_test,predictions))

         r2 = r2_score(y_test,predictions)

         results[name] = { "MAE": mae,"MSE": mse,"RMSE": rmse,"R2": r2}

         trained_models[name] = pipeline

         print(f"MSE  : {mse:,.2f}")
         print(f"MAE  : {mae:,.2f}")
         print(f"RMSE : {rmse:,.2f}")
         print(f"R²   : {r2:.4f}")

         print("-" * 40)

         if r2 > best_r2:
           best_r2 = r2
           best_pipeline = pipeline
           best_model_name = name
      
         logging.info(f"{name} | R2={r2:.4f} | RMSE={rmse:.2f} | MSE={mse:.2f} | MAE={mae:.2f}")

         mlflow.log_metric("MAE", mae)
         mlflow.log_metric("MSE", mse)
         mlflow.log_metric("RMSE", rmse)
         mlflow.log_metric("R2", r2)
         mlflow.sklearn.log_model(sk_model=pipeline, name=name, serialization_format="cloudpickle")

# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(by="R2",ascending=False)

results_df.to_csv("model_comparison_before_tuning.csv")

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(results_df)

# ==========================================
# SAVE COMPARISON RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

mlflow.log_artifact(config["data"]["model_comparison_before"])

print("\nModel comparison saved as:")
print("model_comparison_before_tuning.csv")

# ==========================================
# BEST MODEL BEFORE TUNING
# ==========================================

print("\nBest Model Before Tuning:")
print(best_model_name)

print(f"Best R² Score: {best_r2:.4f}")

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import RandomizedSearchCV

print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)

# Dictionary to store tuned models
tuned_models = {}

# ==========================================
# RIDGE REGRESSION TUNING
# ==========================================

if mlflow.active_run():
   mlflow.end_run()

with mlflow.start_run(run_name="Ridge Regression"):
     print("\nTuning Ridge Regression...")

     ridge_pipeline = Pipeline([("preprocessor", preprocessor),("model", Ridge(random_state= RANDOM_STATE))])

     ridge_params = {"model__alpha": config["ridge"]["alpha_values"]}

     ridge_search = RandomizedSearchCV(estimator= ridge_pipeline,param_distributions=ridge_params,n_iter=config["lasso"]["n_iter"],cv= config["random_forest"]["cv"],scoring="r2",random_state= RANDOM_STATE,n_jobs= LINEAR_N_JOBS)

     ridge_search.fit(X_train, y_train)

     tuned_models["Ridge Regression"] = ridge_search

     best_ridge=ridge_search.best_estimator_

     mlflow.log_params(ridge_search.best_params_)

     print("Best Parameters:", ridge_search.best_params_)
     print("Best CV R²:", ridge_search.best_score_)

# ==========================================
# LASSO REGRESSION TUNING
# ==========================================

if mlflow.active_run():
   mlflow.end_run()

with mlflow.start_run(run_name="Lasso Regression"):
     print("\nTuning Lasso Regression...")

     lasso_pipeline = Pipeline([("preprocessor", preprocessor),("model", Lasso(random_state= RANDOM_STATE,max_iter=LASSO_MAX_ITER))])

     lasso_params = {"model__alpha": config["lasso"]["alpha_values"]}

     lasso_search = RandomizedSearchCV(estimator=lasso_pipeline,param_distributions=lasso_params,n_iter= config["lasso"]["n_iter"],cv= config["random_forest"]["cv"],scoring="r2",random_state= RANDOM_STATE,n_jobs= LINEAR_N_JOBS)

     lasso_search.fit(X_train, y_train)

     tuned_models["Lasso Regression"] = lasso_search

     best_lasso=lasso_search.best_estimator_

     mlflow.log_params(lasso_search.best_params_)

     print("Best Parameters:", lasso_search.best_params_)
     print("Best CV R²:", lasso_search.best_score_)

# ==========================================
# RANDOM FOREST TUNING
# ==========================================

if mlflow.active_run():
   mlflow.end_run()

with mlflow.start_run(run_name="Random Forest Regression"):
     print("\nTuning Random Forest...")

     rf_pipeline = Pipeline([("preprocessor", preprocessor),("model", RandomForestRegressor(random_state= RANDOM_STATE,n_jobs= LINEAR_N_JOBS))])

     rf_params = {"model__n_estimators": config["random_forest_tuned"]["n_estimators"],"model__max_depth": config["random_forest_tuned"]["max_depth"],"model__min_samples_split": config["random_forest_tuned"]["min_samples_split"],"model__min_samples_leaf": config["random_forest_tuned"]["min_samples_leaf"]}

     rf_search = RandomizedSearchCV(estimator=rf_pipeline,param_distributions=rf_params,n_iter= config["random_forest"]["n_iter"],cv= config["random_forest"]["cv"],scoring="r2",random_state= RANDOM_STATE,n_jobs= LINEAR_N_JOBS)

     rf_search.fit(X_train, y_train)

     tuned_models["Random Forest"] = rf_search

     best_rf =rf_search.best_estimator_

     mlflow.log_param("n_estimators",best_rf.named_steps["model"].n_estimators)
     mlflow.log_param("max_depth",best_rf.named_steps["model"].max_depth)
     mlflow.log_param("min_samples_split", best_rf.named_steps["model"].min_samples_split)
     mlflow.log_param("min_samples_leaf", best_rf.named_steps["model"].min_samples_leaf)

     print("Best Parameters:", rf_search.best_params_)
     print("Best CV R²:", rf_search.best_score_)

# ==========================================
# EVALUATE TUNED MODELS
# ==========================================

print("\n" + "=" * 50)
print("EVALUATING TUNED MODELS")
print("=" * 50)

tuned_results = {}

best_final_model = None
best_final_name = None
best_final_r2 = -np.inf

mlflow.set_experiment("Housing Price Prediction")

for name, search in tuned_models.items():

    with mlflow.start_run(run_name=name):
         model = search.best_estimator_

         predictions = model.predict(X_test)

         mae = mean_absolute_error(y_test,predictions)

         mse = mean_squared_error(y_test,predictions)

         rmse = np.sqrt(mean_squared_error(y_test,predictions))

         r2 = r2_score(y_test,predictions)

         tuned_results[name] = {"MAE": mae,"MSE": mse,"RMSE": rmse,"R2": r2}

         print(f"\n{name}")
         print(f"MSE  : {mse:,.2f}")
         print(f"MAE  : {mae:,.2f}")
         print(f"RMSE : {rmse:,.2f}")
         print(f"R²   : {r2:.4f}")

         if r2 > best_final_r2:
            best_final_r2 = r2
            best_final_model = model
            best_final_name = name

         logging.info(f"{name} | R2={r2:.4f} | RMSE={rmse:.2f} | MSE={mse:.2f} | MAE={mae:.2f}")
           
         mlflow.log_metric("MAE", mae)
         mlflow.log_metric("MSE", mse)
         mlflow.log_metric("RMSE", rmse)
         mlflow.log_metric("R2", r2)
         mlflow.sklearn.log_model(sk_model=pipeline, name=name, serialization_format="cloudpickle")

# ==========================================
# COMPARISON AFTER TUNING
# ==========================================

tuned_results_df = pd.DataFrame(tuned_results).T

tuned_results_df = tuned_results_df.sort_values(by="R2",ascending=False)

tuned_results_df.to_csv("model_comparison_after_tuning.csv")

print("\n" + "=" * 50)
print("MODEL COMPARISON AFTER TUNING")
print("=" * 50)

print(tuned_results_df)

mlflow.log_artifact(config["data"]["logs"])
mlflow.log_artifact(config["data"]["model_comparison_after"])

print("\nSaved:")
print("model_comparison_after_tuning.csv")

joblib.dump(X.columns.tolist(),config["data"]["feature_columns"])

best_model_path = config["data"]["best_model"]

if os.path.exists(best_model_path):

    old_pipeline = joblib.load(best_model_path)

    old_predictions = old_pipeline.predict(X_test)
    old_r2 = r2_score(y_test, old_predictions)

    if best_final_r2 > old_r2:

        joblib.dump(best_final_model, best_model_path)

        print("✅ Better model found. Best model updated.")

    else:

        print("ℹ Existing best model retained.")

else:

    joblib.dump(best_final_model, best_model_path)

    print("✅ No previous model found. Best model saved.")

category_mapping = {}

for col in categorical_features:
    category_mapping[col] = sorted(X[col].dropna().unique().tolist())

joblib.dump(category_mapping, config["data"]["category_mapping"])

# ==========================================
# SAVE FINAL BEST MODEL
# ==========================================

best_model_path = config["data"]["best_model"]

if os.path.exists(best_model_path):

    old_model = joblib.load(best_model_path)

    old_predictions = old_model.predict(X_test)
    old_r2 = r2_score(y_test, old_predictions)

    if best_final_r2 > old_r2:

        joblib.dump(best_final_model, best_model_path)

        mlflow.log_param("Best Model", best_final_name)
        mlflow.log_metric("Best R2", best_final_r2)
        mlflow.set_tag("Model Updated", "Yes")

        print("Better model found. Model updated.")

    else:

         mlflow.log_param("Best Model", best_final_name)
         mlflow.log_metric("Best R2", best_final_r2)
         mlflow.set_tag("Model Updated", "No")
 
         print("Existing model retained.")

         print("✅ Better model found. best_model.pkl updated.")
         
else:

    joblib.dump(best_final_model, best_model_path)

    mlflow.log_param("Best Model", best_final_name)
    mlflow.log_metric("Best R2", best_final_r2)
    mlflow.set_tag("Model Updated", "First Model")

    print("✅ First model saved.")

all_models= {"Linear Regression": trained_models["Linear Regression"], "Ridge Regression": ridge_search.best_estimator_, "Lasso Regression": lasso_search.best_estimator_, "Random Forest": rf_search.best_estimator_}

joblib.dump(all_models, config["data"]["all_models"])
    
print("all_models.pkl saved successfully")

print("\n" + "=" * 50)
print("FINAL MODEL")
print("=" * 50)

print("Best Model :", best_final_name)

print(
    f"Final R² Score: {best_final_r2:.4f}"
)

print("\nSaved Files:")
print("✔ best_model.pkl")
print("✔ model_comparison_before_tuning.csv")
print("✔ model_comparison_after_tuning.csv")