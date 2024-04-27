import pandas as pd
import pytest

from pyfixest.estimation.estimation import feols
from pyfixest.report.visualize import coefplot, iplot
from pyfixest.utils.utils import get_data


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
def fit_multi(data):
    return feols(fml="Y + Y2 ~ i(f2, X1)", data=data)


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
def test_iplot(
    fit1,
    fit2,
    fit3,
    fit_multi,
    plot_backend,
    figsize,
    yintercept,
    xintercept,
    drop,
    title,
    coord_flip,
):
    plot_kwargs = {
        "plot_backend": plot_backend,
        "figsize": figsize,
        "yintercept": yintercept,
        "xintercept": xintercept,
        "drop": drop,
        "title": title,
        "coord_flip": coord_flip,
    }

    fit1.iplot(**plot_kwargs)
    fit2.iplot(**plot_kwargs)
    fit3.iplot(**plot_kwargs)
    fit_multi.iplot(**plot_kwargs)

    iplot(fit1, **plot_kwargs)
    iplot([fit1, fit2], **plot_kwargs)
    iplot([fit1, fit2], drop="T.12")


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
def test_coefplot(
    fit1,
    fit2,
    fit3,
    fit_multi,
    plot_backend,
    figsize,
    yintercept,
    xintercept,
    keep,
    drop,
    title,
    coord_flip,
):
    plot_kwargs = {
        "plot_backend": plot_backend,
        "figsize": figsize,
        "yintercept": yintercept,
        "xintercept": xintercept,
        "keep": keep,
        "drop": drop,
        "title": title,
        "coord_flip": coord_flip,
    }

    fit1.coefplot(**plot_kwargs)
    fit2.coefplot(**plot_kwargs)
    fit3.coefplot(**plot_kwargs)
    coefplot(fit1, **plot_kwargs)
    coefplot([fit1, fit2], **plot_kwargs)
    fit_multi.coefplot(**plot_kwargs)
