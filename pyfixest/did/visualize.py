from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def panelview(
    data: pd.DataFrame,
    unit: str,
    time: str,
    treat: str,
    type: Optional[str] = None,
    outcome: Optional[str] = None,
    collapse_to_cohort: Optional[bool] = False,
    subsamp: Optional[int] = None,
    sort_by_timing: Optional[bool] = False,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    figsize: Optional[tuple] = (11, 3),  # Default plot size
    noticks: Optional[bool] = False,
    title: Optional[str] = None,
    legend: Optional[bool] = False,
    ax: Optional[plt.Axes] = None,
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
    type : str, optional
        Optional type of plot. Currently supported: 'outcome'.
    outcome : str, optional
        The column name representing the outcome variable. Used when `type` is 'outcome'.
    collapse_to_cohort : bool, optional
        Whether to collapse units into treatment cohorts.
    subsamp : int, optional
        The number of samples to draw from data set for display (default is None).
    sort_by_timing : bool, optional
        Whether to sort the treatment cohorts by the number of treated periods.
    xlab : str, optional
        The label for the x-axis. Default is None, in which case default labels are used.
    ylab : str, optional
        The label for the y-axis. Default is None, in which case default labels are used.
    figsize : tuple, optional
        The figure size for the outcome plot. Default is (11, 3).
    noticks : bool, optional
        Whether to display ticks on the plot. Default is False.
    title : str, optional
        The title for the plot. Default is None, in which case no title is displayed.
    legend : bool, optional
        Whether to display a legend. Default is False (since binary treatments are
        self-explanatory).
    ax : matplotlib.pyplot.Axes, optional
        The axes on which to draw the plot. Default is None, in which case a new figure
        is created.

    Returns
    -------
    ax : matplotlib.pyplot.Axes

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
    if type == "outcome" and outcome:
        if subsamp:
            unique_units = data[unit].unique()
            sampled_units = np.random.choice(unique_units, size=subsamp, replace=False)
            data = data[data[unit].isin(sampled_units)]
        if collapse_to_cohort:
            treatment_starts = data.groupby(unit).apply(lambda x: x[x[treat] == True][time].min())
            treatment_starts = treatment_starts.reset_index(name='treatment_start')
            treatment_starts = treatment_starts.drop_duplicates(subset='treatment_start')
            data = data[data[unit].isin(treatment_starts[unit])]
        if not ax:
            f, ax = plt.subplots(figsize=figsize, dpi = 300)
        for unit_id in data[unit].unique():
            unit_data = data[data[unit] == unit_id]
            treatment_times = unit_data[unit_data[treat] == True][time]
            
            # If the unit never receives treatment, plot the line in grey
            if treatment_times.empty:
                ax.plot(unit_data[time], unit_data[outcome], color="#999999", linewidth=0.5, alpha = 0.5)
            else:
                treatment_start = treatment_times.min()

                # Plot the entire line with the initial color (orange), then change to red after treatment
                ax.plot(
                    unit_data[time],
                    unit_data[outcome],
                    color="#FF8343",
                    linewidth=0.5,  # Thinner line
                    label=f"Unit {unit_id}" if legend else None,
                    alpha = 0.5
                )
                ax.plot(
                    unit_data[unit_data[time] >= treatment_start][time],
                    unit_data[unit_data[time] >= treatment_start][outcome],
                    color="#ff0000",
                    linewidth=0.9,  
                    alpha = 0.5
                )

        ax.set_xlabel(xlab if xlab else time)
        ax.set_ylabel(ylab if ylab else outcome)
        ax.set_title(title if title else "Outcome over Time with Treatment Effect", fontweight='bold')  # Bold title
        ax.grid(True, color="#e0e0e0", linewidth=0.3, linestyle='-')
        if legend:
            custom_lines = [
                plt.Line2D([0], [0], color="#999999", lw=1.5),
                plt.Line2D([0], [0], color="#FF8343", lw=1.5),
                plt.Line2D([0], [0], color="#ff0000", lw=1.5)
            ]
            ax.legend(custom_lines, ['Control', 'Treatment (Pre)', 'Treatment (Post)'],
                      loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    else:
        treatment_quilt = data.pivot(index=unit, columns=time, values=treat)
        treatment_quilt = treatment_quilt.sample(subsamp) if subsamp else treatment_quilt
        if collapse_to_cohort:
            treatment_quilt = treatment_quilt.drop_duplicates()
        if sort_by_timing:
            treatment_quilt = treatment_quilt.loc[
                treatment_quilt.sum(axis=1).sort_values().index
            ]
        if not ax:
            f, ax = plt.subplots()
        cax = ax.matshow(treatment_quilt, cmap="viridis", aspect="auto")
        f.colorbar(cax) if legend else None
        ax.set_xlabel(xlab) if xlab else None
        ax.set_ylabel(ylab) if ylab else None

        if noticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if title:
            ax.set_title(title)
    return ax
