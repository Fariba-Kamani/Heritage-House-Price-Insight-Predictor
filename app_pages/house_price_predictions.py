import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.machine_learning.data_management import load_pkl_file

# Mapping of feature names to user-friendly labels/descriptions
feature_descriptions = {
    "1stFlrSF": "First Floor square footage",
    "2ndFlrSF": "Second Floor square footage",
    "BedroomAbvGr": "Number of bedrooms above ground level",
    "BsmtExposure": (
        "Walkout or garden level basement walls exposure:\n"
        "(Gd: Good Exposure, Av: Average Exposure, Mn: Minimum Exposure\n"
        "No: No Exposure, None: No Basement)"
    ),
    "BsmtFinSF1": "Finished area in the basement",
    "BsmtFinType1": (
        "Rating of basement finished area:\n"
        "(GLQ: Good Living Quarters, ALQ: Average Living Quarters\n"
        "BLQ: Below Average Living Quarters, Rec: Average Rec Room\n"
        "LwQ: Low Quality, Unf: Unfinished, None: No Basement)"
    ),
    "BsmtUnfSF": "Unfinished basement area",
    "EnclosedPorch": "Enclosed porch area (in square feet)",
    "GarageFinish": (
        "Interior finish of the garage:\n"
        "(Fin: Finished, RFn: Rough Finished, Unf: Unfinished, None: No Garage)"
    ),
    "GarageYrBlt": "Year garage was built",
    "KitchenQual": (
        "Kitchen quality:\n"
        "(Ex: Excellent, Gd: Good, TA: Typical/Average, Fa: Fair, Po: Poor)"
    ),
    "LotArea": "Lot size in square feet",
    "LotFrontage": "Linear feet of street connected to property",
    "MasVnrArea": "Masonry veneer area in square feet",
    "OpenPorchSF": "Open porch area in square feet",
    "OverallCond": "Overall condition rating (1 = Very Poor, 10 = Excellent)",
    "WoodDeckSF": "Wood deck area in square feet"
}


def house_price_predict_body():
    st.title("Predict House Sale Price")

    version = 'v1'
    model = load_pkl_file(f"outputs/ml_pipeline/predict_sale_price/{version}/extra_trees_regressor_pipeline.pkl")
    X_train = pd.read_csv(f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    trained_columns = X_train.columns.tolist()

    df_stats = X_train.copy()
    stat_defaults = {
        col: (df_stats[col].median() if is_numeric_dtype(df_stats[col]) else df_stats[col].mode()[0])
        for col in trained_columns
    }

    st.subheader("Top Features")
    top_features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'GarageArea']
    X_live = pd.DataFrame(columns=trained_columns)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        X_live["OverallQual"] = [st.slider(
            "OverallQual", 1, 10, 5,
            help="Overall material and finish quality (1 = Very Poor, 10 = Excellent). Range: 1 to 10"
        )]
    with col2:
        X_live["GrLivArea"] = [st.number_input(
            "GrLivArea", 300, 5000, 1500,
            help="Above-ground living area (in square feet). Range: 300 to 5000"
        )]
    with col3:
        X_live["YearBuilt"] = [st.number_input(
            "YearBuilt", 1870, 2025, 2000,
            help="Original construction year of the house. Range: 1870 to 2025"
        )]
    with col4:
        X_live["TotalBsmtSF"] = [st.number_input(
            "TotalBsmtSF", 0, 3000, 800,
            help="Total area of the basement (in square feet). Range: 0 to 3000"
        )]
    with col5:
        X_live["GarageArea"] = [st.number_input(
            "GarageArea", 0, 1500, 400,
            help="Size of the garage (in square feet). Range: 0 to 1500"
        )]

    more_input = st.checkbox("Provide more details (optional)", value=False)
    user_inputs = {}
    autofilled_fields = []

    if more_input:
        st.subheader("Additional Features")
        for col in trained_columns:
            if col not in top_features:
                # Get friendly description
                help_text = feature_descriptions.get(col, f"Feature: {col}")

                if not is_numeric_dtype(df_stats[col]):
                    default_val = stat_defaults[col]
                    unique_vals = df_stats[col].dropna().unique()
                    default_index = list(unique_vals).index(default_val) if default_val in unique_vals else 0
                    user_val = st.selectbox(
                        label=col,
                        options=unique_vals,
                        index=default_index,
                        help=help_text
                    )
                else:
                    user_val = st.number_input(
                        label=col,
                        min_value=float(df_stats[col].min()),
                        max_value=float(df_stats[col].max()),
                        value=float(stat_defaults[col]),
                        help=f"{help_text}. Range: {int(df_stats[col].min())} to {int(df_stats[col].max())}"
                    )

                X_live[col] = [user_val]
                if user_val != stat_defaults[col]:
                    user_inputs[col] = user_val

    if st.button("Predict Sale Price"):
        try:
            user_provided_fields = list(user_inputs.keys())

            autofilled_fields = []
            for col in trained_columns:
                if col not in X_live.columns or pd.isnull(X_live[col].values[0]):
                    X_live[col] = [stat_defaults[col]]
                    autofilled_fields.append(col)
                elif col not in user_provided_fields:
                    autofilled_fields.append(col)

            X_live = X_live[trained_columns]

            prediction = model.predict(X_live)[0]
            st.success(f"Predicted Sale Price: **${int(prediction):,}**")

            if autofilled_fields:
                st.warning(
                    f"The following features were filled with default values: {', '.join(autofilled_fields)}. "
                    "Providing more details may improve the prediction."
                )

            st.write("Prediction Input Overview:")
            st.dataframe(X_live)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

house_price_predict_body()
