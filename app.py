import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.project_summary import project_summary_body
from app_pages.correlation_insights import correlation_insights_body
from app_pages.house_price_predictions import house_price_predictions_body
from app_pages.custom_input_prediction import custom_input_prediction_body
from app_pages.technical_summary import technical_summary_body

app = MultiPage(app_name= "Churnometer") # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Project Summary", project_summary_body)
app.add_page("Correlation Insights", correlation_insights_body)
app.add_page("House Price Predictions", house_price_predictions_body)
app.add_page("Custom Input Prediction", custom_input_prediction_body)
app.add_page("Technical Summary", technical_summary_body)

app.run() # Run the  app