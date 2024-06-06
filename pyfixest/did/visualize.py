from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def panelview(
    data: pd.DataFrame,
    unit: str,
    time: str,
    treat: str,
    subsamp: Optional[int] = None,
    title: Optional[str] = None,
) -> None:
    """
    Generate a panel view of the treatment variable over time for each unit.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing the data.
    unit : str
        The column name representing the unit identifier.
    time : str
        The column name representing the time identifier.
    treat : str
        The column name representing the treatment variable.
    subsamp : int, optional
        The number of samples to draw from data set for display (default is None).
    title : str, optional
        The title for the plot. Default is None, in which case no title is displayed.

    Returns
    -------
    None
        This function does not return any value. It displays a matrix plot.

    Examples
    --------
    ```python
    import pandas as pd
    import numpy as np

    df_het = pd.read_csv("pd.read_csv("pyfixest/did/data/df_het.csv")
    panelview(
        data = df_het,
        unit = "unit",
        time = "year",
        treat = "treat",
        subsamp = 50,
        title = "Treatment Assignment"
    )
    ```
    """
    treatment_quilt = data.pivot(index=unit, columns=time, values=treat)
    if subsamp:
        plt.matshow(treatment_quilt.sample(subsamp))
    else:
        plt.matshow(treatment_quilt)
    plt.xlabel(time)
    plt.ylabel(unit)
    if title:
        plt.title(f"Panel view of {title}")
