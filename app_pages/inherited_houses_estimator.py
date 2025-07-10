import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to sys.path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.machine_learning.data_management import load_pkl_file, load_inherited_houses

@st.cache_resource
def load_pipeline():
    version = 'v1'
    model_path = f"outputs/ml_pipeline/predict_sale_price/{version}/extra_trees_regressor_pipeline.pkl"
    return load_pkl_file(model_path)

def inherited_houses_estimator_body():

    pipeline = load_pipeline()

    st.write("## Inherited Houses Value Estimator")
    st.info("Lydia inherited 4 houses. Let's estimate their total market value using the trained model.")

    df = load_inherited_houses()
    
    st.write("### Inherited Houses - Input Features")
    st.dataframe(df)

    try:
        # Predict prices using pipeline
        predicted_prices = pipeline.predict(df)
        df["Predicted SalePrice"] = predicted_prices.astype(int)

        st.write("### Predicted Sale Prices")
        st.dataframe(df[["Predicted SalePrice"]])

        total = df["Predicted SalePrice"].sum()
        st.success(f"### Estimated Total Sale Value for All 4 Houses: **${total:,.0f}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

inherited_houses_estimator_body()