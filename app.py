import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==================================
# LOAD FILES
# ==================================

model = joblib.load("models/best_model.pkl")

feature_columns = joblib.load(
    "feature_columns.pkl"
)

category_mapping = joblib.load(
    "category_mapping.pkl"
)

# ==================================
# TITLE
# ==================================

st.title("🏠 Housing Price Prediction")

st.markdown(
    "Predict house prices using Machine Learning"
)

# ==================================
# INPUTS
# ==================================

user_data = {}

st.sidebar.header("Property Information")

for col in feature_columns:

    if col in category_mapping:

        user_data[col] = st.sidebar.selectbox(
            col,
            category_mapping[col]
        )

    else:

        if col == "Id":
            user_data[col] = 1

        else:
            user_data[col] = st.sidebar.number_input(
                col,
                value=0.0
            )

# ==================================
# DATAFRAME
# ==================================

input_df = pd.DataFrame(
    [user_data]
)

# ==================================
# FEATURE ENGINEERING
# ==================================

try:

    input_df["TotalSF"] = (
        input_df["TotalBsmtSF"]
        + input_df["1stFlrSF"]
        + input_df["2ndFlrSF"]
    )

    input_df["TotalBathrooms"] = (
        input_df["FullBath"]
        + (0.5 * input_df["HalfBath"])
        + input_df["BsmtFullBath"]
        + (0.5 * input_df["BsmtHalfBath"])
    )

    input_df["HouseAge"] = (
        input_df["YrSold"]
        - input_df["YearBuilt"]
    )

    input_df["RemodAge"] = (
        input_df["YrSold"]
        - input_df["YearRemodAdd"]
    )

    input_df["TotalPorchSF"] = (
        input_df["OpenPorchSF"]
        + input_df["EnclosedPorch"]
        + input_df["3SsnPorch"]
        + input_df["ScreenPorch"]
    )

except:
    pass

# ==================================
# PREDICT
# ==================================

if st.button("Predict House Price"):

    prediction = model.predict(
        input_df
    )[0]

    st.success(
        f"Estimated House Price: ${prediction:,.0f}"
    )

    st.metric(
        "Predicted Price",
        f"${prediction:,.0f}"
    )

# ==================================
# SHOW INPUTS
# ==================================

with st.expander("View Inputs"):

    st.dataframe(input_df)