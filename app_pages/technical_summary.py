import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # Used for future optional visuals
from sklearn import set_config
import streamlit.components.v1 as components

from src.machine_learning.data_management import (
    load_house_price_data,
    load_pkl_file
)
from src.machine_learning.evaluate_performance import (
    regression_performance,
    regression_evaluation_plots
)


def technical_summary_body():
    """
    Streamlit page that summarizes the technical details
    of the machine learning pipeline.
    Includes model selection rationale, visual pipeline,
    feature importance, and performance evaluation.
    """

    # Load final pipeline, feature importance chart, and train/test data
    version = 'v1'
    # Load trained ExtraTreesRegressor pipeline
    extra_trees_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/"
        f"{version}/extra_trees_regressor_pipeline.pkl"
        )

    # Load feature importance image
    extra_trees_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/"
        f"{version}/features_importance.png"
        )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/"
        f"{version}/X_test.csv"
        )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv"
        ).squeeze()
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv"
        ).squeeze()

    # Model selection summary
    st.write("### ML Pipeline: Predict Sale Price")
    st.info(
        "After evaluating multiple models that "
        "best matched the project's goals, "
        "the ExtraTreesRegressor was selected. "
        "It achieved the highest mean cross-validation "
        "R² score (0.8264), outperforming:\n"
        "- Lasso: 0.8147\n"
        "- Ridge: 0.7762\n"
        "- Random Forest: 0.7853\n\n"
        "It also had a low standard deviation (~0.0345),"
        " showing consistent performance. "
        "Extra Trees was chosen for its balance of high "
        "accuracy and resistance to overfitting."
    )
    st.write("---")

    # Pipeline structure
    st.write("### ML Pipeline Steps")
    st.write(extra_trees_pipe)
    st.write("---")

    # Visual pipeline flow
    st.write("### Visual Pipeline Flow")
    set_config(display='diagram')
    pipeline_html = extra_trees_pipe._repr_html_()
    components.html(pipeline_html, height=600, scrolling=True)
    st.write("---")

    # Display training features and feature importance
    st.write("### Features Used for Training")
    st.write(X_train.columns.to_list())

    st.write("### Top 5 Most Important Features")
    st.image(extra_trees_feat_importance)
    st.write("---")

    # Performance on training and testing sets
    st.header("Pipeline Performance")
    regression_performance(X_train, y_train, X_test, y_test, extra_trees_pipe)
    regression_evaluation_plots(X_train, y_train,
                                X_test, y_test, extra_trees_pipe)

    # Final summary
    st.header("Conclusion")
    st.write(
        "The ExtraTreesRegressor highlights the top "
        "predictors of house price as: "
        "**Overall Quality**, **Above-Ground Living Area**,"
        "**Year Built**, "
        "**Basement Size**, and **Garage Area**. \n\n"
        "To maximize returns, Lydia should focus on properties"
        " with modern finishes, "
        "larger living spaces, and functional additions like "
        "garages and basements — "
        "features that significantly influence buyer perception"
        " and sale price in Ames."
    )
