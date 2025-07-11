import streamlit as st
import pandas as pd

from src.machine_learning.data_management import (
    load_pkl_file,
    load_inherited_houses,
)


@st.cache_resource
def load_pipeline():
    """
    Load and cache the trained machine learning pipeline
    for predicting house sale prices.
    """
    version = 'v1'
    model_path = (
        f"outputs/ml_pipeline/predict_sale_price/"
        f"{version}/extra_trees_regressor_pipeline.pkl"
    )
    return load_pkl_file(model_path)


def inherited_houses_estimator_body():
    """
    Streamlit page that estimates the total market value of 4 inherited houses.

    - Loads input data representing house features
    - Uses a trained model to predict sale prices
    - Displays individual and total estimated sale values
    """
    pipeline = load_pipeline()

    st.write("## Inherited Houses Value Estimator")
    st.info(
        "Lydia inherited 4 houses. Let's estimate their total "
        "market value using the trained prediction model."
    )

    # Load house features for inherited homes
    df = load_inherited_houses()

    st.write("### Inherited Houses - Input Features")
    st.dataframe(df)

    try:
        # Predict sale prices using the loaded pipeline
        predicted_prices = pipeline.predict(df)
        df["Predicted SalePrice"] = predicted_prices.astype(int)

        # Show individual predictions
        st.write("### Predicted Sale Prices")
        st.dataframe(df[["Predicted SalePrice"]])

        # Display total estimated value
        total = df["Predicted SalePrice"].sum()
        st.success(
            f"### Estimated Total Sale Value for All 4 Houses: "
            f"**${total:,.0f}**"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
