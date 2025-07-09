import streamlit as st

def project_summary_body():

    st.write("### Heritage Housing - Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"## Project Terms & Jargon\n"
        f"- **Client**: The fictional individual (Lydia Doe)"
        f" who inherited four houses and seeks pricing insights.\n"
        f"- **Property**: A house located in Ames, Iowa, included in the dataset.\n"
        f"- **Sale Price**: The amount a house was sold for."
        f" This is what we aim to predict.\n"
        f"- **Attribute (or Feature)**: A characteristic of a house, such as its size,"
        f" number of bedrooms, or the quality of the kitchen.\n"
        f"- **Prediction**: An estimate of a house’s sale"
        f" price based on its attributes."
        )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Fariba-Kamani/Heritage-House-Price-Insight-Predictor).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in discovering how"
        f" house attributes correlate with sale prices."
        f" Therefore, the client expects data visualizations"
        f" of the correlated variables against the sale price.\n"
        f"* 2 - The client is interested in predicting the house"
        f" sale prices from her 4 inherited houses, and any"
        f" other houses in Ames, Iowa."
        )
    
project_summary_body()