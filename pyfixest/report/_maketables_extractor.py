"""Teach `maketables` about every pyfixest result class.

`maketables` recognizes a pyfixest model by testing
`isinstance(model, (Feols, Fepois, Feiv))`. GLM and quantile results are leaves
of `BaseRegression` rather than of `Feols`, so they need an extractor whose
`can_handle` covers the base class instead. Everything that decides what a
table shows -- the coefficient table, the statistics map, the dependent-variable
and fixed-effect labels -- is inherited unchanged, so a table built for a GLM or
quantile fit is exactly the one `maketables` produced while those classes still
inherited from `Feols`.

Registration happens once, when `pyfixest.report` is imported.
"""

from __future__ import annotations

from typing import Any

from maketables.extractors import PyFixestExtractor, register_extractor

from pyfixest.estimation.models.base_regression_ import BaseRegression


class BaseRegressionExtractor(PyFixestExtractor):  # type: ignore[misc]
    """Extract any fitted pyfixest result, not only the `Feols` family."""

    def can_handle(self, model: Any) -> bool:
        """Report whether `model` is a fitted pyfixest result.

        Parameters
        ----------
        model : Any
            The object `maketables` is looking for an extractor for.

        Returns
        -------
        bool
            True for any `BaseRegression`, which covers OLS, IV, GLM, Poisson,
            and quantile results.
        """
        return isinstance(model, BaseRegression)


def register_pyfixest_extractor() -> None:
    """Register the pyfixest extractor with `maketables`, at most once.

    `maketables` returns the first registered extractor that accepts a model,
    so its own `PyFixestExtractor` still handles OLS and IV results and this one
    picks up the estimators it does not recognize.
    """
    from maketables.extractors import _EXTRACTOR_REGISTRY

    if any(
        isinstance(extractor, BaseRegressionExtractor)
        for extractor in _EXTRACTOR_REGISTRY
    ):
        return
    # maketables internal: its own `PyFixestExtractor`, which this class
    # extends, does not implement the `default_stat_keys` member its
    # `ModelExtractor` protocol declares. The registry looks the method up
    # defensively, so the extractor works exactly as the built-in one does.
    register_extractor(
        BaseRegressionExtractor()  # ty: ignore[invalid-argument-type]
    )


register_pyfixest_extractor()
