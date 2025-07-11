import streamlit as st


def project_summary_body():
    """
    Streamlit page that presents a high-level overview
    of the Heritage Housing project.

    Includes:
    - Key terminology and project jargon
    - A link to the full project README
    - A summary of the client's business requirements
    """
    st.write("### Heritage Housing - Project Summary")

    # Definitions of key terms used throughout the project
    st.info(
        "## Project Terms & Jargon\n"
        "- **Client**: The fictional individual (Lydia Doe)"
        "who inherited four houses and seeks pricing insights.\n"
        "- **Property**: A house located in Ames, Iowa, "
        "included in the dataset.\n"
        "- **Sale Price**: The amount a house was sold for. "
        "This is what we aim to predict.\n"
        "- **Attribute (or Feature)**: A characteristic of a house,"
        "such as its size, number of bedrooms,"
        "or the quality of the kitchen.\n"
        "- **Prediction**: An estimate of a house’s sale price"
        "based on its attributes."
    )

    # Link to the full project README documentation on GitHub
    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file]"
        "("
        "https://github.com/Fariba-Kamani/"
        "Heritage-House-Price-Insight-Predictor"
        ")."
    )

    # Overview of the client's key business requirements
    st.success(
        "The project has two primary business requirements:\n\n"
        "* 1 - The client is interested in discovering "
        "how house attributes correlate with sale prices. "
        "Therefore, the client expects data visualizations "
        "of the correlated variables against the sale price.\n"
        "* 2 - The client is interested in predicting the house"
        "sale prices from her 4 inherited houses, and any "
        "other houses in Ames, Iowa."
    )
