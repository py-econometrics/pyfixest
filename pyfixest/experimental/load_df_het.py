import pandas as pd


def load_df_het():
    """
    Load the df_het.csv dataset.

    Returns
    -------
    df_het : pd.DataFrame
        The df_het dataset.
    """

    df_het = pd.read_csv("./pyfixest/experimental/data/df_het.csv")
    return df_het
