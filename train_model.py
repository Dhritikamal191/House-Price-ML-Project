# ==========================================
# IMPORT LIBRARIES
# ==========================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv("data/train.csv")

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
print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(20)
)


# ==========================================
# TARGET VARIABLE ANALYSIS
# ==========================================

print("\nTarget Variable: SalePrice")

print("\nSalePrice Statistics:")
print(df["SalePrice"].describe())

print("\nSkewness:",
      df["SalePrice"].skew())

print("Kurtosis:",
      df["SalePrice"].kurt())


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns


print("\nNumber of Numerical Features:",
      len(numeric_features))

print("Number of Categorical Features:",
      len(categorical_features))


# ==========================================
# PREPROCESSING PIPELINES
# ==========================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])


print("\nPreprocessing Pipeline Created Successfully!")

# ==========================================
# IMPORT MODELS AND METRICS
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(
        alpha=1.0,
        random_state=42
    ),

    "Lasso Regression": Lasso(
        alpha=0.001,
        random_state=42,
        max_iter=10000
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
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


for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = pipeline

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    print("-" * 40)

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_pipeline = pipeline
        best_model_name = name


# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results).T

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(results_df)


# ==========================================
# SAVE COMPARISON RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "models/model_comparison_before_tuning.csv"
)

print(
    "\nModel comparison saved as:"
)

print(
    "models/model_comparison_before_tuning.csv"
)


# ==========================================
# BEST MODEL BEFORE TUNING
# ==========================================

print("\nBest Model Before Tuning:")
print(best_model_name)

print(
    f"Best R² Score: {best_r2:.4f}"
)
