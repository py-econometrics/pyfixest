from unittest.mock import patch

import matplotlib
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.report.visualize import _HAS_LETS_PLOT, coefplot, iplot, set_figsize
from pyfixest.utils.utils import get_data

matplotlib.use("Agg")  # Use a non-interactive backend


@pytest.fixture
def data():
    data = get_data()
    data["f2"] = pd.Categorical(data["f2"])
    return data


@pytest.fixture
def fit1(data):
    return feols(fml="Y ~ i(f2, X1) | f1", data=data, vcov="iid")


@pytest.fixture
def fit2(data):
    return feols(fml="Y ~ i(f2, X1) | f2", data=data, vcov="iid")


@pytest.fixture
def fit3(data):
    return feols(fml="Y ~ i(f2, X1, ref=1.0)", data=data, vcov="iid")


@pytest.fixture
def fit4(data):
    return feols(fml="Y ~ i(f2)", data=data, vcov="iid")


@pytest.fixture
def fit_multi(data):
    return feols(fml="Y + Y2 ~ i(f2, X1)", data=data)


@pytest.mark.extended
@pytest.mark.parametrize(
    argnames="figsize",
    argvalues=[(10, 6), None],
)
@pytest.mark.parametrize(
    argnames="plot_backend",
    argvalues=["lets_plot", "matplotlib"],
    ids=["lets_plot", "matplotlib"],
)
def test_set_figsize(figsize, plot_backend):
    if plot_backend == "lets_plot" and not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    figsize_not_none = set_figsize(figsize, plot_backend)

    if figsize is None:
        if plot_backend == "lets_plot":
            assert figsize_not_none == (500, 300)
        elif plot_backend == "matplotlib":
            assert figsize_not_none == (10, 6)
    else:
        assert figsize_not_none == figsize


@pytest.mark.extended
def test_set_figsize_not_none_bad_backend():
    figsize_not_none = set_figsize((10, 6), "bad_backend")
    assert figsize_not_none == (10, 6)


@pytest.mark.extended
def test_set_figsize_none_bad_backend():
    with pytest.raises(
        ValueError, match=r"plot_backend must be either 'lets_plot' or 'matplotlib'\."
    ):
        set_figsize(None, "bad_backend")


@pytest.mark.parametrize(
    argnames="plot_backend",
    argvalues=["lets_plot", "matplotlib"],
    ids=["lets_plot", "matplotlib"],
)
@pytest.mark.parametrize(
    argnames="figsize", argvalues=[(10, 6), None], ids=["figsize", "no_figsize"]
)
@pytest.mark.parametrize(
    argnames="yintercept", argvalues=[1.0, None], ids=["yintercept", "no_yintercept"]
)
@pytest.mark.parametrize(
    argnames="xintercept", argvalues=[2.0, None], ids=["xintercept", "no_xintercept"]
)
@pytest.mark.parametrize(
    argnames="drop", argvalues=[None, "T.12"], ids=["drop", "no_drop"]
)
@pytest.mark.parametrize(
    argnames="title", argvalues=[None, "Title"], ids=["no_title", "title"]
)
@pytest.mark.parametrize(
    argnames="coord_flip", argvalues=[True, False], ids=["coord_flip", "no_coord_flip"]
)
@pytest.mark.parametrize(
    argnames="labels",
    argvalues=[None, {"f2": "F2", "X1": "1x"}],
    ids=["no_labels", "labels"],
)
@pytest.mark.extended
def test_iplot(
    fit1,
    fit2,
    fit3,
    fit4,
    fit_multi,
    plot_backend,
    figsize,
    yintercept,
    xintercept,
    drop,
    title,
    coord_flip,
    labels,
):
    if plot_backend == "lets_plot" and not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    plot_kwargs = {
        "plot_backend": plot_backend,
        "figsize": figsize,
        "yintercept": yintercept,
        "xintercept": xintercept,
        "drop": drop,
        "title": title,
        "coord_flip": coord_flip,
        "labels": labels,
    }

    fit1.iplot(**plot_kwargs)
    fit2.iplot(**plot_kwargs)
    fit3.iplot(**plot_kwargs)
    fit4.iplot(**plot_kwargs)
    fit_multi.iplot(**plot_kwargs)

    iplot(fit1, **plot_kwargs)
    iplot([fit1, fit2], **plot_kwargs)


@pytest.mark.extended
def test_iplot_error(data):
    with pytest.raises(ValueError):
        fit4 = feols(fml="Y ~ X1", data=data, vcov="iid")
        fit4.iplot()
        iplot(fit4)


@pytest.mark.parametrize(
    argnames="plot_backend",
    argvalues=["lets_plot", "matplotlib"],
    ids=["lets_plot", "matplotlib"],
)
@pytest.mark.parametrize(
    argnames="figsize", argvalues=[(10, 6), None], ids=["figsize", "no_figsize"]
)
@pytest.mark.parametrize(
    argnames="yintercept", argvalues=[1.0, None], ids=["yintercept", "no_yintercept"]
)
@pytest.mark.parametrize(
    argnames="xintercept", argvalues=[2.0, None], ids=["xintercept", "no_xintercept"]
)
@pytest.mark.parametrize(
    argnames="keep", argvalues=[None, "X"], ids=["keep", "no_keep"]
)
@pytest.mark.parametrize(
    argnames="drop", argvalues=[None, "X"], ids=["drop", "no_drop"]
)
@pytest.mark.parametrize(
    argnames="title", argvalues=[None, "Title"], ids=["no_title", "title"]
)
@pytest.mark.parametrize(
    argnames="coord_flip", argvalues=[True, False], ids=["coord_flip", "no_coord_flip"]
)
@pytest.mark.parametrize(
    argnames="labels",
    argvalues=[None, {"f2": "F2", "X1": "1x"}],
    ids=["no_labels", "labels"],
)
@pytest.mark.extended
def test_coefplot(
    fit1,
    fit2,
    fit3,
    fit4,
    fit_multi,
    plot_backend,
    figsize,
    yintercept,
    xintercept,
    keep,
    drop,
    title,
    coord_flip,
    labels,
):
    if plot_backend == "lets_plot" and not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    plot_kwargs = {
        "plot_backend": plot_backend,
        "figsize": figsize,
        "yintercept": yintercept,
        "xintercept": xintercept,
        "keep": keep,
        "drop": drop,
        "title": title,
        "coord_flip": coord_flip,
        "labels": labels,
    }

    fit1.coefplot(**plot_kwargs)
    fit2.coefplot(**plot_kwargs)
    fit3.coefplot(**plot_kwargs)
    fit4.coefplot(**plot_kwargs)
    coefplot(fit1, **plot_kwargs)
    coefplot([fit1, fit2], **plot_kwargs)
    fit_multi.coefplot(**plot_kwargs)


@pytest.mark.extended
@patch("pyfixest.report.visualize._coefplot_matplotlib")
def test_coefplot_default_figsize_matplotlib(_coefplot_matplotlib_mock, fit1, data):
    coefplot(fit1, plot_backend="matplotlib")
    _, kwargs = _coefplot_matplotlib_mock.call_args
    assert kwargs.get("figsize") == (10, 6)


@pytest.mark.extended
@patch("pyfixest.report.visualize._coefplot_matplotlib")
def test_coefplot_non_default_figsize_matplotlib(_coefplot_matplotlib_mock, fit1, data):
    coefplot(fit1, figsize=(12, 7), plot_backend="matplotlib")
    _, kwargs = _coefplot_matplotlib_mock.call_args
    assert kwargs.get("figsize") == (12, 7)


@pytest.mark.extended
@patch("pyfixest.report.visualize._coefplot_lets_plot")
def test_coefplot_default_figsize_lets_plot(_coefplot_lets_plot_mock, fit1, data):
    if not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    coefplot(fit1, plot_backend="lets_plot")
    _, kwargs = _coefplot_lets_plot_mock.call_args
    assert kwargs.get("figsize") == (500, 300)


@pytest.mark.extended
@patch("pyfixest.report.visualize._coefplot_lets_plot")
def test_coefplot_non_default_figsize_lets_plot(_coefplot_lets_plot_mock, fit1, data):
    if not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    coefplot(fit1, figsize=(600, 400), plot_backend="lets_plot")
    _, kwargs = _coefplot_lets_plot_mock.call_args
    assert kwargs.get("figsize") == (600, 400)


def test_rename_models():
    # Skip the entire test if lets-plot is not installed
    # This is because the default backend is lets-plot if available
    if not _HAS_LETS_PLOT:
        pytest.skip("lets-plot is not installed")

    df = pf.get_data()
    fit1 = pf.feols("Y ~ i(f1)", data=df)
    fit2 = pf.feols("Y ~ i(f1) + f2", data=df)

    pf.iplot(
        models=[fit1, fit2],
        rename_models={"Y~i(f1)": "Model 1", "Y~i(f1)+f2": "Model 2"},
    )

    pf.coefplot(
        models=[fit1, fit2],
        rename_models={"Y~i(f1)": "Model 1", "Y~i(f1)+f2": "Model 2"},
    )

    fit_multi = pf.feols("Y ~ sw(f1, f2)", data=df)
    fit_multi.coefplot(rename_models={"Y~f1": "Model 1", "Y~f2": "Model 2"})

    pf.coefplot(models=[fit1], rename_models={"Y~i(f1)": "Model 1"}, joint=True)

    pf.coefplot(models=[fit1], rename_models={"Y~i(f1)": "Model 1"}, joint="both")

    with pytest.warns(
        UserWarning,
        match="The following model names specified in rename_models are not found in the models",
    ):
        coefplot(models=[fit1], rename_models={"Y~a": "Model 1"}, joint="bad")


def test_rename_models_matplotlib():
    """Test rename_models functionality with matplotlib backend."""
    df = pf.get_data()
    fit1 = pf.feols("Y ~ i(f1)", data=df)
    fit2 = pf.feols("Y ~ i(f1) + f2", data=df)

    pf.iplot(
        models=[fit1, fit2],
        rename_models={"Y~i(f1)": "Model 1", "Y~i(f1)+f2": "Model 2"},
        plot_backend="matplotlib",
    )

    pf.coefplot(
        models=[fit1, fit2],
        rename_models={"Y~i(f1)": "Model 1", "Y~i(f1)+f2": "Model 2"},
        plot_backend="matplotlib",
    )

    fit_multi = pf.feols("Y ~ sw(f1, f2)", data=df)
    fit_multi.coefplot(
        rename_models={"Y~f1": "Model 1", "Y~f2": "Model 2"},
        plot_backend="matplotlib",
    )

    pf.coefplot(
        models=[fit1],
        rename_models={"Y~i(f1)": "Model 1"},
        joint=True,
        plot_backend="matplotlib",
    )

    pf.coefplot(
        models=[fit1],
        rename_models={"Y~i(f1)": "Model 1"},
        joint="both",
        plot_backend="matplotlib",
    )
