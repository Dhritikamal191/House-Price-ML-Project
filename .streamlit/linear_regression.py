import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])

categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

lr_model = Pipeline([('preprocessor', preprocessor),('model', LinearRegression())])

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mse=mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

linear_results =pd.DataFrame({"Model": ["Linear Regression"],"MAE":[mae], "RMSE":[rmse],"R2":[r2]})
linear_results.to_csv("linear_regression_results.csv", index=False)
print("CSV saved successfully!")