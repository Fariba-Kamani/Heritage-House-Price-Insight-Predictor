import plotly.graph_objects as go
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.machine_learning.data_management import load_house_price_data
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style globally
sns.set_style("whitegrid")


def correlation_insights_body():
    """
    Streamlit page function that presents correlation
    insights between house attributes and SalePrice.

    Features included:
    - Data inspection
    - Correlation analysis summary (Pearson & Spearman)
    - Correlation interpretation from EDA
    - Univariate distribution plots
    - Correlation plots showing relationship to SalePrice
    - Custom parallel categories plot for multivariate exploration
    """

    # load data
    df = load_house_price_data()

    # Variables selected based on Pearson and Spearman
    # correlation with SalePrice
    vars_to_study = ['1stFlrSF',
                     'GarageArea',
                     'GrLivArea',
                     'OverallQual',
                     'TotalBsmtSF',
                     'YearBuilt']

    st.write("### Correlation Insights")
    st.info(
        f"* The client is interested in discovering "
        f"how the house attributes correlate with the "
        f"sale price. Therefore, the client expects "
        f"data visualisations of the correlated variables "
        f"against the sale price to show that."
        )

    # inspect data
    if st.checkbox("Inspect Sale Records"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in"
        f"the notebook to better understand how "
        f"the variables are correlated to SalePrice. \n"
        f"The most correlated variables are: **{vars_to_study}**"
    )

    # Interpretation summary based on correlation plots
    st.info(
        "The correlation scores and visualizations support "
        "the following interpretations:\n\n"
        "- **1st Floor Area (1stFlrSF):** Larger first-floor"
        "areas generally correlate with higher sale prices. "
        "However, the wide spread in sale prices for a given "
        "floor size suggests that other factors also influence"
        "the final price.\n\n"
        "- **Garage Area:** A positive correlation exists "
        "between GarageArea and SalePrice, particularly "
        "within the 400–800 sq ft range. "
        "Despite this trend, there's noticeable variability"
        "— some small or garage-less homes still achieve high"
        "prices, likely due to other strong contributing features.\n\n"
        "- **Above-Ground Living Area (GrLivArea):** There is a strong "
        "positive linear relationship between GrLivArea and SalePrice up"
        "to around 4000 sq ft. "
        "Homes larger than 2000 sq ft tend to sell for significantly more."
        "This appears to be one of the strongest predictors of sale price.\n\n"
        "- **Overall Quality (OverallQual):** Sale price increases "
        "exponentially with higher overall quality ratings. "
        "The average sale price nearly doubles for each step above "
        "quality level 6, making this one of the most powerful "
        "indicators of home value.\n\n"
        "- **Basement Area (TotalBsmtSF):** A positive linear trend "
        "exists between basement area and sale price up to"
        "approximately 3000 sq ft. "
        "However, very large basements do not always translate"
        "to significantly higher value.\n\n"
        "- **Year Built:** Newer houses tend to sell at higher prices."
        "Properties built after 2000 show more consistent high sale prices, "
        "while older homes generally have lower prices and "
        "greater variability."
        )

    # Code copied from "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Explore how each selected variable relates to SalePrice
    if st.checkbox("Explore feature distribution and relation to SalePrice"):
        sale_price_per_variable(df_eda)

    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.info(
            "**How to Read This Parallel Categories Plot:**\n\n"
            "- Each vertical axis represents a feature that is "
            "related to house prices.\n"
            "- The values for each feature are grouped into ranges"
            "(bins), e.g., `GarageArea: 600 to 900`.\n"
            "- Each line (path) running across the axes represents"
            "a group of houses sharing those feature combinations.\n"
            "- The thickness and brightness of each path reflect how"
            "many houses follow that path — the brighter and thicker,"
            "the more frequent.\n"
            "- You can hover over a section to see the count of houses"
            "in that category range.\n\n"
            "**Tip:** Look for paths that lead to the higher end of the"
            "SalePrice axis — these often show what feature combinations"
            "are associated with more expensive homes."
        )
        parallel_plot_sale_price(df_eda)


def sale_price_per_variable(df_eda):
    """
    Plots the distribution of and relationship
    between SalePrice and selected features.
    Uses barplot for ordinal variables like OverallQual,
    and histplot/scatter for continuous variables.
    """
    target_var = 'SalePrice'
    force_barplot = ['OverallQual']
    st.subheader("Distribution (left) and SalePrice correlation (right)")

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        st.write(f"#### {col}")
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var, force_barplot)
        st.write("---")


def plot_categorical(df, col, target_var):
    """
    Plots the distribution of a categorical variable using seaborn countplot.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=20)
    plt.xticks(rotation=90)
    st.pyplot(fig)


def plot_numerical(df, col, target_var, force_barplot=['OverallQual']):
    """
    For numerical variables, plots:
    - Left: Distribution (histplot or countplot if bar-forced)
    - Right: Relationship with SalePrice (scatter or bar)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Distribution of the feature
    if col in force_barplot:
        sns.countplot(data=df, x=col,
                      order=sorted(df[col].unique()),
                      ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")
    else:
        sns.histplot(data=df, x=col, kde=True, element="step", ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")

    # Right: Relationship with SalePrice
    if col in force_barplot:
        sns.barplot(data=df, x=col,
                    y=target_var,
                    order=sorted(df[col].unique()),
                    ax=axes[1])
        axes[1].set_ylabel("Average Sale Price")
        axes[1].set_title(f"{col} vs {target_var} (Mean)")
    else:
        sns.scatterplot(data=df, x=col, y=target_var, ax=axes[1])
        axes[1].set_title(f"{col} vs {target_var}")

    plt.tight_layout()
    st.pyplot(fig)


def parallel_plot_sale_price(df_eda):
    """
    Creates a custom Parallel Categories Plot using Plotly.
    - Discretizes numerical variables into bins
    - Labels them for clarity
    - Displays categories colored by SalePrice
    """
    # Define binning
    binning_dict = {
        'OverallQual': [-np.Inf, 4, 6, 8, np.Inf],
        'GrLivArea': [-np.Inf, 1200, 1800, 2500, np.Inf],
        'GarageArea': [-np.Inf, 300, 600, 900, np.Inf],
        '1stFlrSF': [-np.Inf, 1000, 1400, 1800, np.Inf],
        'TotalBsmtSF': [-np.Inf, 800, 1200, 1600, np.Inf],
        'YearBuilt': [-np.Inf, 1945, 1970, 2000, np.Inf],
        'SalePrice': [-np.Inf, 150000, 200000, 300000, np.Inf]
    }
    # Apply binning
    disc = ArbitraryDiscretiser(binning_dict=binning_dict)
    df_binned = disc.fit_transform(df_eda)

    # Replace bin numbers with readable labels
    for col, bins in disc.binner_dict_.items():
        label_map = {}
        for i in range(len(bins) - 1):
            if i == 0:
                label_map[i] = f"<{int(bins[1])}"
            elif i == len(bins) - 2:
                label_map[i] = f">{int(bins[-2])}"
            else:
                label_map[i] = f"{int(bins[i])} to {int(bins[i + 1])}"
        df_binned[col] = df_binned[col].replace(label_map)
    # Prepare dimensions for Plotly parallel categories
    dimensions = [
        dict(label=col, values=df_binned[col])
        for col in ['OverallQual', 'GrLivArea', 'GarageArea',
                    '1stFlrSF', 'TotalBsmtSF', 'YearBuilt', 'SalePrice']]
    fig = go.Figure(data=[
        go.Parcats(
            dimensions=dimensions,
            line=dict(color=df_eda['SalePrice'], colorscale='Plasma'),
            labelfont=dict(size=14, color='black'),  # control label font
            tickfont=dict(size=12, color='black'),   # control tick font
        )
    ])

    fig.update_layout(
        title="Custom Parallel Categories Plot",
        width=1050,
        height=500,
        font=dict(color="black", size=14),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig)
