import warnings
from typing import Optional, Union

import pyfixest
from pyfixest.utils._exceptions import find_stack_level


def iplot(
    models,
    alpha: float = 0.05,
    figsize: tuple = (500, 300),
    yintercept: Union[int, str, None] = None,
    xintercept: Union[int, str, None] = None,
    rotate_xticks: int = 0,
    title: Optional[str] = None,
    coord_flip: Optional[bool] = True,
):
    """
    Plot model coefficients.

    Plot model coefficients for variables interacted via "i()" syntax, with
    confidence intervals.

        'pyfixest.visualize.iplot' is deprecated and will be removed in a future
        version. Please use 'pyfixest.iplot' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

    Parameters
    ----------
    models : list or object
        A list of fitted models of type `Feols` or
        `Fepois`, or just a single model.
    figsize : tuple
        The size of the figure.
    alpha : float
        The significance level for the confidence intervals.
    yintercept : int or None, optional
        The value at which to draw a horizontal line on the plot.
    xintercept : int or None, optional
        The value at which to draw a vertical line on the plot.
    rotate_xticks : float, optional
        The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    title : str, optional
        The title of the plot.
    coord_flip : bool, optional
        Whether to flip the coordinates of the plot. Default is True.

    Returns
    -------
    object
        A lets-plot figure.

    Examples
    --------
    ```{python}
    from pyfixest.utils import get_data
    from pyfixest.estimation import feols
    from pyfixest.visualize import iplot

    df = get_data()
    fit1 = feols("Y ~ i(f1)", data = df)
    fit2 = feols("Y ~ i(f1) + X2", data = df)
    fit3 = feols("Y ~ i(f1) + X2 | f2", data = df)

    iplot([fit1, fit2, fit3])
    ```
    """
    warnings.warn(
        "'pyfixest.visualize.iplot' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.iplot' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return pyfixest.iplot(
        models=models,
        alpha=alpha,
        figsize=figsize,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        title=title,
        coord_flip=coord_flip,
    )


def coefplot(
    models: list,
    alpha: int = 0.05,
    figsize: tuple = (500, 300),
    yintercept: float = 0,
    xintercept: float = None,
    rotate_xticks: int = 0,
    coefficients: Optional[list[str]] = None,
    title: Optional[str] = None,
    coord_flip: Optional[bool] = True,
):
    """
    Plot model coefficients with confidence intervals.

        'pyfixest.visualize.coefplot' is deprecated and will be removed in a future
        version. Please use 'pyfixest.coefplot' instead. You may refer the updated
        documentation at: https://s3alfisc.github.io/pyfixest/quickstart.html

    Parameters
    ----------
    models : list or object
        A list of fitted models of type `Feols` or `Fepois`, or just a single model.
    figsize : tuple
        The size of the figure.
    alpha : float
        The significance level for the confidence intervals.
    yintercept : float or None, optional
        The value at which to draw a horizontal line on the plot. Default is 0.
    xintercept : float or None, optional
        The value at which to draw a vertical line on the plot. Default is None.
    rotate_xticks : float, optional
        The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    coefficients : list, optional
        A list of coefficients to plot. If None, all coefficients are plotted.
    title : str, optional
        The title of the plot.
    coord_flip : bool, optional
        Whether to flip the coordinates of the plot. Default is True.

    Returns
    -------
    object
        A lets-plot figure.

    Examples
    --------
    ```{python}
    from pyfixest.utils import get_data
    from pyfixest.estimation import feols
    from pyfixest.visualize import coefplot

    df = get_data()
    fit1 = feols("Y ~ X1", data = df)
    fit2 = feols("Y ~ X1 + X2", data = df)
    fit3 = feols("Y ~ X1 + X2 | f1", data = df)

    coefplot([fit1, fit2, fit3])
    ```
    """
    warnings.warn(
        "'pyfixest.visualize.coefplot' is deprecated and "
        "will be removed in a future version.\n"
        "Please use 'pyfixest.coefplot' instead. "
        "You may refer the updated documentation at: "
        "https://s3alfisc.github.io/pyfixest/quickstart.html",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    return pyfixest.coefplot(
        models=models,
        alpha=alpha,
        figsize=figsize,
        yintercept=yintercept,
        xintercept=xintercept,
        rotate_xticks=rotate_xticks,
        coefficients=coefficients,
        title=title,
        coord_flip=coord_flip,
    )
