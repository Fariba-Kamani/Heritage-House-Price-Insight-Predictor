import streamlit as st


def project_hypothesis_body():
    """
    Streamlit page that presents project hypotheses and their validation status
    based on insights from exploratory data analysis (EDA) and model findings.

    Each hypothesis is introduced with context, followed by a conclusion
    reflecting its support based on correlation coefficients and
    model importance.
    """
    st.write("### Project Hypothesis and Validation")

    # Hypothesis 1: Larger living area (GrLivArea) leads to higher sale prices
    st.subheader("H1: Houses with greater total living area"
                 " (GrLivArea) are more expensive.")
    st.info(
        "Exploratory data analysis shows that GrLivArea "
        "(above-ground living area) is right-skewed, "
        "with most houses between 1000 and 2000 sq ft."
        "Prices generally increase with size, particularly "
        "for houses under 4000 sq ft. Beyond this, outliers appear."
    )
    st.success(
        "This hypothesis is **strongly supported** by the data. "
        "GrLivArea shows a strong positive correlation "
        "with SalePrice (Spearman: 0.70, Pearson: 0.708) and "
        "is among the top 5 predictive features in the model."
    )

    # Hypothesis 2: Better overall quality leads to higher sale prices
    st.subheader("H2: Higher overall quality (OverallQual) is "
                 "associated with higher sale prices.")
    st.info(
        "Most houses in the dataset are rated between 5 and 7 "
        "for overall quality. There is a clear exponential "
        "increase in sale price with higher quality ratings."
        "Houses rated 8 and above show particularly sharp price increases."
    )
    st.success(
        "This hypothesis is **very strongly supported**. OverallQual has"
        "the highest correlation with SalePrice "
        "(Spearman: 0.809, Pearson: 0.790) and ranks as the most"
        "important feature in the Extra Trees model."
    )

    # Hypothesis 3: Presence and size of garage influences sale price
    st.subheader("H3: Houses with a garage (GarageArea > 0) "
                 "sell for higher prices than those without.")
    st.info(
        "GarageArea shows a positive correlation with SalePrice,"
        "especially in the 400–800 sq ft range. "
        "However, some homes with very small garages "
        "(or none at all) also sell for high prices — indicating "
        "that other features can compensate."
    )
    st.warning(
        "This hypothesis is **partially supported**. While garage"
        "size contributes to sale price (Spearman: 0.64, "
        "Pearson: 0.62), it is not as strong a predictor as living"
        "area or quality. Some high-value homes lack garages entirely."
    )
