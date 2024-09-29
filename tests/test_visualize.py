import matplotlib.pyplot as plt
import pandas as pd

from pyfixest.did.visualize import panelview
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.report.visualize import coefplot, iplot
from pyfixest.utils.utils import get_data


def test_visualize():
    data = get_data()
    fit1 = feols("Y ~ X1 + X2 | f1", data=data)
    coefplot(fit1)

    data_pois = get_data(model="Fepois")
    fit2 = fepois("Y ~ X1 + X2 + f2 | f1", data=data_pois, vcov={"CRV1": "f1+f2"})
    coefplot(fit2)

    coefplot([fit1, fit2])

    # FixestMulti
    fit3 = feols("Y + Y2 ~ X1 + X2 | f1", data=data)
    coefplot(fit3.to_list())

    fit3.coefplot()

    # iplot
    fit5 = feols("Y ~ i(f1)", data=data)
    iplot(fit5)

    # FixestMulti
    fit6 = feols("Y + Y2 ~ X1 + X2 | f1", data=data)
    fit6.coefplot()


def test_panelview():
    df_het = pd.read_csv("pyfixest/did/data/df_het.csv")

    # Test basic functionality
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        subsamp=50,
        title="Treatment Assignment",
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with basic functionality for outcome plot
    ax = panelview(
        data=df_het,
        outcome="dep_var",
        unit="unit",
        time="year",
        treat="treat",
        subsamp=50,
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with collapse_to_cohort
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        collapse_to_cohort=True,
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with collapse_to_cohort for outcome plot
    ax = panelview(
        data=df_het,
        type="outcome",
        outcome="dep_var",
        unit="unit",
        time="year",
        treat="treat",
        collapse_to_cohort=True,
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with units_to_plot for outcome plot
    ax = panelview(
        data=df_het,
        type="outcome",
        outcome="dep_var",
        unit="unit",
        time="year",
        treat="treat",
        units_to_plot=[1, 2, 3, 4],
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with sort_by_timing
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        sort_by_timing=True,
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test with custom labels and no ticks
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        xlab="Custom X",
        ylab="Custom Y",
        noticks=True,
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Custom X"
    assert ax.get_ylabel() == "Custom Y"
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    plt.close()

    # Test with legend
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        legend=True,
    )
    assert isinstance(ax, plt.Axes)
    assert len(ax.figure.axes) > 1  # Check if colorbar (legend) is present
    plt.close()

    # Test with provided axes
    fig, ax = plt.subplots()
    result_ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        ax=ax,
    )
    assert result_ax is ax
    plt.close()

    # Test with all options enabled
    ax = panelview(
        data=df_het,
        unit="unit",
        time="year",
        treat="treat",
        collapse_to_cohort=True,
        subsamp=30,
        sort_by_timing=True,
        xlab="Years",
        ylab="Units",
        noticks=True,
        title="Full Test",
        legend=True,
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Full Test"
    plt.close()
