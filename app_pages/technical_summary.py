import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.machine_learning.data_management import load_house_price_data, load_pkl_file
from src.machine_learning.evaluate_performance import regression_performance, regression_evaluation_plots
from sklearn import set_config
import streamlit.components.v1 as components


def technical_summary_body():
    # Load SalePrice analysis files and pipeline
    version = 'v1'
    extra_trees_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/extra_trees_regressor_pipeline.pkl")
    extra_trees_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv").squeeze()
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv").squeeze()
    
    st.write("### ML Pipeline: Predict Sale Price")
    # display pipeline training summary conclusions
    st.info(
        f"After evaluating multiple models that were considered "
        f"the most suitable algorithms based on hyperparameter "
        f"optimization and the project's performance requirement "
        f"(an R² score of at least 0.75 on both the train and test "
        f"sets), the ExtraTreesRegressor was selected as the final "
        f"estimator. It achieved the highest mean cross-validation "
        f"R² score of 0.8264, compared to:\n"
        f"  * Lasso: 0.8147\n"
        f"  * Ridge: 0.7762\n"
        f"  * RandomForestRegressor: 0.7853\n\n"
        f"It also maintained a low standard deviation (≈ 0.0345), "
        f"indicating stable performance across all folds.\n"
        f"ExtraTreesRegressor was chosen for its strong predictive "
        f"performance, low variance, and robustness to overfitting."
        f" It offered the best trade-off between bias and variance,"
        f" making it an ideal choice for the regression task of "
        f"predicting Ames house prices."
        )
    st.write("---")

    # show pipeline steps
    st.write("### ML pipeline to predict Sale Price.")
    st.write(extra_trees_pipe)
    st.write("---")

    # Visual flow
    st.write("### Visual Pipeline Flow")
    set_config(display='diagram')
    pipeline_html = extra_trees_pipe._repr_html_()
    components.html(pipeline_html, height=600, scrolling=True)
    st.write("---")

    # show best features
    st.write("### The features the model was trained on")
    st.write(X_train.columns.to_list())
    st.write("### Top 5 important features in descending order:")
    st.image(extra_trees_feat_importance)
    st.write("---")

    # evaluate performance on both sets
    st.write("# Pipeline Performance")
    regression_performance(X_train, y_train, X_test, y_test, extra_trees_pipe)
    regression_evaluation_plots(X_train, y_train, X_test, y_test, extra_trees_pipe)

    st.write("# Conclusion")
    st.write(f"Based on the feature importance from the "
             f"ExtraTreesRegressor model, the top five predictors"
            f" of house price are **Overall Quality, Above-Ground "
            f"Living Area, Year Built, Basement Size, and Garage Area**."
            f" To maximize value, Lydia should prioritize homes with "
            f"modern finishes, spacious layouts, and functional "
            f"extras like large garages and basements. "
            f"These features have the greatest impact on buyer "
            f"perception and sale price in the Ames housing market.")

technical_summary_body()