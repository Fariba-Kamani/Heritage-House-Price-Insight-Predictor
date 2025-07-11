"""
Main application script for the Heritage House Price Estimator.
This script uses Streamlit's multipage app structure to allow
navigation between different analysis and prediction modules.
"""

import streamlit as st
from app_pages.multipage import MultiPage

# Import page content functions
from app_pages.project_summary import project_summary_body
from app_pages.correlation_insights import correlation_insights_body
from app_pages.project_hypothesis import project_hypothesis_body
from app_pages.technical_summary import technical_summary_body
from app_pages.inherited_houses_estimator import (
    inherited_houses_estimator_body
)
from app_pages.house_price_predictions import house_price_predict_body

# Set Streamlit page configuration
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏘️",
    layout="centered"
)

# Initialize app
app = MultiPage(app_name="House Price Estimator")

# Register app pages
app.add_page("Project Summary", project_summary_body)
app.add_page("Correlation Insights", correlation_insights_body)
app.add_page("Hypothesis", project_hypothesis_body)
app.add_page("Technical Summary", technical_summary_body)
app.add_page("Inherited Houses Estimator", inherited_houses_estimator_body)
app.add_page("House Price Predictions", house_price_predict_body)

# Run the app
app.run()
