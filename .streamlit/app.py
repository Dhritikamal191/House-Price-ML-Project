import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import plotly.graph_objects as go
from pathlib import Path
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
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR/"config.yaml","r") as f:
    config = yaml.safe_load(f)
    
best_model = joblib.load(config["models"]["best_model"])

feature_columns = joblib.load(config["models"]["feature_columns"])

category_mapping = joblib.load(config["models"]["category_mapping"])

numeric_defaults = joblib.load(config["models"]["numeric_default"])

models = joblib.load(config["models"]["all_models"])

# ==========================================
# HEADER
# ==========================================

st.title("🏠 Housing Price Prediction System")

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

comparison = pd.read_csv(config["data"]["model_comparison_after"])

best_model_name = comparison.iloc[0,0]

col1, col2, col3 = st.columns(3)

with col1:
     st.metric("Features",len(feature_columns))

with col2:
     st.metric("Categorical Features",len(category_mapping))

with col3:
     st.metric("Best Model",best_model_name)

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================

st.sidebar.header("🤖 Model Selection")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys()),
    index=3
)

if selected_model == "Linear Regression":

    st.warning(
        """
        • Linear Regression is included as a baseline model.

        • Due to one-hot encoding and extensive feature engineering,
        the model may produce unstable predictions.
        """
    )

elif selected_model == "Lasso Regression":

    st.warning("""
     Lasso Regression

    • Best performing model based on R² score.

    • Automatically removes less important features.

    • Can produce unstable predictions for unusual feature combinations.

    • Negative house price predictions may occur for inputs outside the training distribution.
    """)

elif selected_model == "Ridge Regression":

    st.info("""
     Ridge Regression

    • Uses L2 regularization to reduce overfitting.

    • More stable than Linear Regression.

    • Handles multicollinearity effectively.

    • Provides consistent and reliable predictions across most property types.
    """)

elif selected_model == "Random Forest":

    st.success("""
     Random Forest

    • Ensemble tree-based model.

    • Captures complex non-linear relationships.

    • Produces highly stable predictions.

    • Less sensitive to extreme user inputs.

    • Recommended model for deployment and real-world usage.
    """)

st.sidebar.header("🏡 Property Information")

st.sidebar.info(
    f"Total Features: {len(feature_columns)}"
)

st.sidebar.info(
    f"Most features are pre-filled using dataset medians."
)

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
        
    default_value = float(numeric_defaults.get(col,0))
        
    if col in category_mapping:
       user_data[col] = st.sidebar.selectbox(col,category_mapping[col])
    else:
         if col == "Id":
            user_data[col] = 1
         else:              
              user_data[col] = st.sidebar.number_input(col,value= default_value)

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

    model = models[selected_model]
  
    prediction = model.predict(input_df)[0]
    
    predictions = np.clip(prediction, a_min=0, a_max=None)

    st.success(
        f"Estimated House Price: ${predictions:,.0f}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
         st.metric(
            "Predicted Price",
            f"${predictions:,.0f}"
         )

    with col2:
         st.metric(
            "Price (Thousands)",
            f"${predictions/1000:.1f}K"
         )

    with col3:
         if predictions < 150000:
            category = "Budget Home 🏠"

         elif predictions < 300000:
              category = "Mid-Range Home 🏡"

         else:
              category = "Premium Home 🏰"

         st.metric("Property Category", category)

    # ======================================
    # GAUGE CHART
    # ======================================

    fig = go.Figure()

    fig.add_trace(
    go.Indicator(
        mode="gauge+number+delta",
        value=predictions,

        number={
            "prefix":"$",
            "valueformat":",.0f"
        },

        delta={
            "reference":180000,
            "relative":True,
            "valueformat":".1%"
        },

        title={
            "text":"🏠 Predicted House Price",
            "font":{"size":24}
        },

        gauge={

            "axis":{
                "range":[0,max(predictions*1.5,500000)],
                "tickwidth":2
            },

            "bar":{
                "thickness":0.35
            },

            "steps":[

                {
                    "range":[0,150000],
                    "color":"#dbeafe"
                },

                {
                    "range":[150000,300000],
                    "color":"#93c5fd"
                },

                {
                    "range":[300000,500000],
                    "color":"#2563eb"
                }

            ],

            "threshold":{

                "line":{
                    "color":"red",
                    "width":4
                },

                "thickness":0.8,

                "value":predictions
            }
        }
      )
    )

    fig.update_layout(
    height=450,
    margin=dict(
        l=30,
        r=30,
        t=80,
        b=20
       )
    )

    st.plotly_chart(
    fig,
    use_container_width=True
    )    

    report = pd.DataFrame({

    "PredictedPrice":[predictions]})

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
with st.expander("📌 Project Information"):
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

     • MSE

     • RMSE

     • R² Score
     """)

# ==========================================
# MODEL COMPARISON
# ==========================================

with st.expander("📊 Model Comparison"):
     try:
         comparison = pd.read_csv(config["data"]["model_comparison_before"])

         st.subheader("📊 Model Comparison Before Tuning")

         st.dataframe(comparison,use_container_width=True)

     except:
            pass
     
     st.warning("""
     Linear Regression was included as a baseline model.

     The model performed well on the original dataset (R² ≈ 0.87). However, after feature engineering, severe multicollinearity caused unstable coefficient estimates, resulting in extremely large MSE and R2 values.

     Therefore, Ridge Regression, Lasso Regression, and Random Forest were preferred for final model selection.
     """)

     linear = pd.read_csv(config["data"]["linear_regression_results"]) 

     st.subheader("Linear Regression Baseline Performance")

     st.dataframe(linear, use_container_width=True)
    
     try:
         st.subheader("Model Comparison After Tuning")
         st.dataframe(comparison, use_container_width=True)

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
