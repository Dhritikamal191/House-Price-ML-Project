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