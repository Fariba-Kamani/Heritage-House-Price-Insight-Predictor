import streamlit as st
import pandas as pd
import joblib


@st.cache_data
def load_house_price_data():
    """
    Loads the house price records dataset from disk.

    Returns:
        pd.DataFrame: DataFrame containing the house sale records.
    """
    df = pd.read_csv("outputs/datasets/collection/house_prices_records.csv")
    return df


def load_pkl_file(file_path):
    """
    Loads a pickled object (e.g., ML model or pipeline)
    from the given file path.

    Args:
        file_path (str): Path to the pickle (.pkl) file.

    Returns:
        object: The deserialized Python object.
    """
    return joblib.load(filename=file_path)


@st.cache_data
def load_inherited_houses():
    """
    Loads the dataset containing the inherited houses' features.

    Returns:
        pd.DataFrame: DataFrame with features
        for the inherited properties.
    """
    df = pd.read_csv("outputs/datasets/collection/inherited_houses.csv")
    return df
