import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# LOAD FILES
# ==========================================

model = joblib.load("best_model.pkl")

feature_columns = joblib.load(
    "feature_columns.pkl"
)

category_mapping = joblib.load(
    "category_mapping.pkl"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#2563eb;
}

.metric-card{
    background-color:#f8fafc;
    padding:15px;
    border-radius:10px;
    border:1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<p class="main-title">🏠 Housing Price Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown("""
Predict residential property prices using Machine Learning.

Models Used:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
""")

# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Features",
        len(feature_columns)
    )

with col2:
    st.metric(
        "Categorical Features",
        len(category_mapping)
    )

with col3:
    st.metric(
        "Model",
        "Best Model"
    )

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================

st.sidebar.header("🏡 Property Information")

user_data = {}

for col in feature_columns:

    if col in [
        "TotalSF",
        "TotalBathrooms",
        "HouseAge",
        "RemodAge",
        "TotalPorchSF"
    ]:
        continue

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

# ==========================================
# DATAFRAME
# ==========================================

input_df = pd.DataFrame([user_data])

# ==========================================
# FEATURE ENGINEERING
# ==========================================

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

# ==========================================
# PREDICTION BUTTON
# ==========================================

predict = st.button(
    "🔍 Predict House Price",
    use_container_width=True
)

# ==========================================
# PREDICTION RESULT
# ==========================================

if predict:

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated House Price: ${prediction:,.0f}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Price",
            f"${prediction:,.0f}"
        )

    with col2:

        st.metric(
            "Price (Thousands)",
            f"${prediction/1000:.1f}K"
        )

    # ======================================
    # GAUGE CHART
    # ======================================

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Predicted Price"},
    gauge={"axis": {"range": [0, max(prediction*1.5, 500000)]}}))

    st.plotly_chart(
        fig,
        use_container_width=True)

    report = pd.DataFrame({

    "PredictedPrice":[prediction]})

    st.download_button(
    "📥 Download Prediction Report",
    report.to_csv(index=False),
    file_name="prediction_report.csv",
    mime="text/csv")

# ==========================================
# INPUT SUMMARY
# ==========================================

with st.expander("📋 View Input Data"):

    st.dataframe(
        input_df,
        use_container_width=True
    )

# ==========================================
# MODEL INFORMATION
# ==========================================

st.subheader("📌 Project Information")

st.info("""
This Housing Price Prediction System was trained using:

• Linear Regression

• Ridge Regression

• Lasso Regression

• Random Forest Regressor

Feature Engineering:

• TotalSF

• TotalBathrooms

• HouseAge

• RemodAge

• TotalPorchSF

Evaluation Metrics:

• MAE

• RMSE

• R² Score
""")

# ==========================================
# MODEL COMPARISON
# ==========================================

try:

    comparison = pd.read_csv(
        "model_comparison_after_tuning.csv"
    )

    st.subheader("📊 Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

except:
    pass

st.subheader("🏡 Property Summary")

summary = pd.DataFrame({

    "Feature":[
        "Overall Quality",
        "Living Area",
        "Garage Capacity",
        "Bedrooms",
        "Bathrooms"
    ],

    "Value":[
        input_df.get("OverallQual",[0])[0],
        input_df.get("GrLivArea",[0])[0],
        input_df.get("GarageCars",[0])[0],
        input_df.get("BedroomAbvGr",[0])[0],
        input_df.get("FullBath",[0])[0]
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

st.subheader("⚙️ Engineered Features")

engineered_df = pd.DataFrame({

    "Feature":[
        "TotalSF",
        "TotalBathrooms",
        "HouseAge",
        "RemodAge",
        "TotalPorchSF"
    ],

    "Value":[
        input_df["TotalSF"].iloc[0],
        input_df["TotalBathrooms"].iloc[0],
        input_df["HouseAge"].iloc[0],
        input_df["RemodAge"].iloc[0],
        input_df["TotalPorchSF"].iloc[0]
    ]
})

st.dataframe(
    engineered_df,
    use_container_width=True
)
