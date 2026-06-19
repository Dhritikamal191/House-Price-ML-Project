
import joblib
import numpy as np
import pandas as pd

best_model = joblib.load("models/best_model.pkl")

test_df = pd.read_csv("test.csv")

print(test_df.shape)

test_df["TotalSF"] = (
    test_df["TotalBsmtSF"]
    + test_df["1stFlrSF"]
    + test_df["2ndFlrSF"]
)

test_df["TotalBathrooms"] = (
    test_df["FullBath"]
    + (0.5 * test_df["HalfBath"])
    + test_df["BsmtFullBath"]
    + (0.5 * test_df["BsmtHalfBath"])
)

test_df["HouseAge"] = (
    test_df["YrSold"]
    - test_df["YearBuilt"]
)

test_df["RemodAge"] = (
    test_df["YrSold"]
    - test_df["YearRemodAdd"]
)

test_df["TotalPorchSF"] = (
    test_df["OpenPorchSF"]
    + test_df["EnclosedPorch"]
    + test_df["3SsnPorch"]
    + test_df["ScreenPorch"]
)

predictions = best_model.predict(test_df)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print(submission.head())
