import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# Set global plotting style
sns.set_style("whitegrid")


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Display regression performance metrics on both the train and test sets.
    Parameters:
    - X_train, y_train: Training data and targets
    - X_test, y_test: Test data and targets
    - pipeline: Trained sklearn pipeline or model
    """
    st.write("## Model Evaluation")

    st.write("### Train Set Performance")
    regression_evaluation(X_train, y_train, pipeline)

    st.write("### Test Set Performance")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Evaluate model performance on a dataset and print R², MAE, MSE, and RMSE.
    Parameters:
    - X: Features
    - y: Target values
    - pipeline: Trained model/pipeline
    """
    prediction = pipeline.predict(X)
    r2 = r2_score(y, prediction).round(3)
    mae = mean_absolute_error(y, prediction).round(3)
    mse = mean_squared_error(y, prediction).round(3)
    rmse = np.sqrt(mse).round(3)

    st.write(f"- R² Score: **{r2}**")
    st.write(f"- Mean Absolute Error (MAE): **{mae}**")
    st.write(f"- Mean Squared Error (MSE): **{mse}**")
    st.write(f"- Root Mean Squared Error (RMSE): **{rmse}**")


def regression_evaluation_plots(
    X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5
):
    """
    Generate scatter plots comparing actual vs. predicted values
    for both train and test sets.
    Parameters:
    - X_train, y_train: Training data and targets
    - X_test, y_test: Test data and targets
    - pipeline: Trained model
    - alpha_scatter: Transparency level for scatter plots
    """
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Train set plot
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual Sale Price")
    axes[0].set_ylabel("Predicted Sale Price")
    axes[0].set_title("Train Set")

    # Test set plot
    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual Sale Price")
    axes[1].set_ylabel("Predicted Sale Price")
    axes[1].set_title("Test Set")

    st.pyplot(fig)
