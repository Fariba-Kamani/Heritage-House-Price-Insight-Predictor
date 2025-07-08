import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import plotly.graph_objects as go

import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.machine_learning.data_management import load_house_price_data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def correlation_insights_body():

    # load data
    df = load_house_price_data()

    # hard copied from EDA, Correlation and PPS Study notebook
    vars_to_study =['1stFlrSF',
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
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to SalePrice. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "EDA, Correlation and PPS Study notebook - Insight from the plots" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* larger first-floor areas tend to correlate with higher sale prices. "
        f"The high spread in SalePrice for a given 1stFlrSF indicates that "
        f"other factors likely influence the price as well.\n"
        f"* A positive correlation exists between GarageArea and SalePrice,"
        f" especially noticeable in the 400-800 sq ft range."
        f" However, there is significant spread — houses with the same "
        f"garage size can vary widely in price."
        f" Some houses with tiny garages also have high prices "
        f"— possibly indicating no garage but other strong contributing features.\n"
        f"* Strong positive linear trend up to around 4000 sq ft between "
        f"GrLivArea and sale price, beyond which outliers appear. Houses "
        f"larger than 2000 sq ft tend to fetch significantly higher prices."
        f" Data suggests larger living space directly correlates with increased sale price "
        f"— one of the strongest relationships seen so far.\n"
        f"* Clear exponential increase in average sale price with higher quality."
        f" The mean sale price nearly doubles with each increment above "
        f"quality level 6. This is a very strong predictor of price \n"
        f"* A positive linear trend exists between basement area "
        f"and sale price up to ~3000 sq ft. Large basements not "
        f"always translating to higher value. \n"
        f"* newer houses tend to sell at higher prices. "
        f"Houses built after 2000 show a more consistent "
        f"concentration of high sale prices. Older homes have "
        f"more price variability and are generally lower priced.\n"
    )

    # Code copied from "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Sale Price per Variable"):
        sale_price_per_variable(df_eda)
        
    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Information in white indicates the profile from a churned customer")
        parallel_plot_sale_price(df_eda)

def sale_price_per_variable(df_eda):
    target_var = 'SalePrice'
    force_barplot = ['OverallQual']
    st.subheader("Visualize SalePrice per Variable")

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        st.write(f"#### {col}")
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var, force_barplot)
        st.write("---")

def plot_categorical(df, col, target_var):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Distribution of {col}", fontsize=20)
    plt.xticks(rotation=90)
    st.pyplot(fig)

def plot_numerical(df, col, target_var, force_barplot=['OverallQual']):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Distribution of the feature
    if col in force_barplot:
        sns.countplot(data=df, x=col, order=sorted(df[col].unique()), ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")
    else:
        sns.histplot(data=df, x=col, kde=True, element="step", ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")

    # Right: Relationship with SalePrice
    if col in force_barplot:
        sns.barplot(data=df, x=col, y=target_var, order=sorted(df[col].unique()), ax=axes[1])
        axes[1].set_ylabel("Average Sale Price")
        axes[1].set_title(f"{col} vs {target_var} (Mean)")
    else:
        sns.scatterplot(data=df, x=col, y=target_var, ax=axes[1])
        axes[1].set_title(f"{col} vs {target_var}")

    plt.tight_layout()
    st.pyplot(fig)

def parallel_plot_sale_price(df_eda):
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

    disc = ArbitraryDiscretiser(binning_dict=binning_dict)
    df_binned = disc.fit_transform(df_eda)

    # Create readable bin labels
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

    dimensions = [
        dict(label=col, values=df_binned[col])
        for col in ['OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF', 'TotalBsmtSF','YearBuilt', 'SalePrice']
    ]

    fig = go.Figure(data=[
        go.Parcats(
            dimensions=dimensions,
            line=dict(color=df_eda['SalePrice'], colorscale='Plasma'),
            labelfont=dict(size=14, color='black'),  # <-- control label font
            tickfont=dict(size=12, color='black'),   # <-- control tick font
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

# Call the main function to render the page
correlation_insights_body()
