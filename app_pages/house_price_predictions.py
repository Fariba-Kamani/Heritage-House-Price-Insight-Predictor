import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.machine_learning.data_management import load_pkl_file

# Dictionary mapping feature names to user-friendly descriptions for tooltips
feature_descriptions = {
    "1stFlrSF": "First Floor square footage",
    "2ndFlrSF": "Second Floor square footage",
    "BedroomAbvGr": "Number of bedrooms above ground level",
    "BsmtExposure": (
        "Walkout or garden level basement walls exposure:\n"
        "(Gd: Good Exposure, Av: Average Exposure, Mn: Minimum Exposure,\n"
        "No: No Exposure, None: No Basement)"
    ),
    "BsmtFinSF1": "Finished area in the basement",
    "BsmtFinType1": (
        "Rating of basement finished area:\n"
        "(GLQ: Good Living Quarters, ALQ: Average Living Quarters,\n"
        "BLQ: Below Average Living Quarters, Rec: Average Rec Room,\n"
        "LwQ: Low Quality, Unf: Unfinished, None: No Basement)"
    ),
    "BsmtUnfSF": "Unfinished basement area",
    "EnclosedPorch": "Enclosed porch area (in square feet)",
    "GarageFinish": (
        "Interior finish of the garage:\n"
        "(Fin: Finished, RFn: Rough Finished, Unf: Unfinished, "
        "None: No Garage)"
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
    """
    Streamlit page for predicting house sale price.
    - Loads a trained ML model and expected features.
    - Allows users to input top features and optionally customize
      additional ones.
    - Uses statistical defaults for any missing features.
    - Displays predicted price and warns about autofilled fields.
    """

    st.title("Predict House Sale Price")

    # Load trained pipeline and training columns
    version = 'v1'
    model = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/"
        "extra_trees_regressor_pipeline.pkl"
        )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    trained_columns = X_train.columns.tolist()

    # Compute statistical defaults (median for numeric, mode for categorical)
    df_stats = X_train.copy()
    stat_defaults = {
        col: (
            df_stats[col].median()
            if is_numeric_dtype(df_stats[col])
            else df_stats[col].mode()[0]
        )
        for col in trained_columns
    }

    # Top 5 features based on model feature importance (Extra Trees Regressor)
    st.subheader("Top Features")
    top_features = ['OverallQual', 'GrLivArea',
                    'YearBuilt', 'TotalBsmtSF', 'GarageArea']
    X_live = pd.DataFrame(columns=trained_columns)

    # Input fields for top features
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        X_live["OverallQual"] = [st.slider(
                                "OverallQual", 1, 10, 5,
                                help=(
                                 "Overall material and finish quality "
                                 "(1 = Very Poor, 10 = Excellent)"
                                 )
        )]
    with col2:
        X_live["GrLivArea"] = [st.number_input(
            "GrLivArea", 300, 5000, 1500,
            help="Above-ground living area (in square feet)"
        )]
    with col3:
        X_live["YearBuilt"] = [st.number_input(
            "YearBuilt", 1870, 2025, 2000,
            help="Year the house was originally built"
        )]
    with col4:
        X_live["TotalBsmtSF"] = [st.number_input(
            "TotalBsmtSF", 0, 3000, 800,
            help="Total basement area in square feet"
        )]
    with col5:
        X_live["GarageArea"] = [st.number_input(
            "GarageArea", 0, 1500, 400,
            help="Garage area in square feet"
        )]

    # More detailed input for additional features - Optional
    more_input = st.checkbox("Provide more details (optional)", value=False)
    user_inputs = {}
    autofilled_fields = []

    if more_input:
        st.subheader("Additional Features")
        for col in trained_columns:
            if col not in top_features:
                help_text = feature_descriptions.get(col, f"Feature: {col}")

                if not is_numeric_dtype(df_stats[col]):
                    default_val = stat_defaults[col]
                    unique_vals = df_stats[col].dropna().unique()
                    default_index = (
                        list(unique_vals).index(default_val)
                        if default_val in unique_vals
                        else 0
                        )
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
                        help=(
                            f"{help_text}. "
                            f"Range: {int(df_stats[col].min())}"
                            f"to {int(df_stats[col].max())}"
                        )
                    )

                X_live[col] = [user_val]

                # Track if user has modified the default
                if user_val != stat_defaults[col]:
                    user_inputs[col] = user_val

    # Predict button logic
    if st.button("Predict Sale Price"):
        try:
            user_provided_fields = list(user_inputs.keys())

            # Fill in any missing or null fields with statistical defaults
            for col in trained_columns:
                if (
                    col not in X_live.columns
                    or pd.isnull(X_live[col].values[0])
                ):
                    X_live[col] = [stat_defaults[col]]
                    autofilled_fields.append(col)
                elif col not in user_provided_fields:
                    autofilled_fields.append(col)

            # Ensure correct feature order
            X_live = X_live[trained_columns]

            # Generate prediction
            prediction = model.predict(X_live)[0]
            st.success(f"Predicted Sale Price: **${int(prediction):,}**")

            # Warn user if default values were used
            if autofilled_fields:
                st.warning(
                    f"The following features were filled with "
                    f"default values: {', '.join(autofilled_fields)}. "
                    "Providing more details may improve the prediction."
                )

            # Show full input used for prediction
            st.write("Prediction Input Overview:")
            st.dataframe(X_live)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
