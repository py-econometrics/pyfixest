import warnings
from typing import Optional, Union

import pyfixest
from pyfixest.utils._exceptions import find_stack_level
from pyfixest.utils.dev_utils import docstring_from


@docstring_from(pyfixest.iplot)
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


@docstring_from(pyfixest.coefplot)
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
