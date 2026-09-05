from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    import pandas as pd

    from pyfixest.estimation.internals.literals import InferenceType
    from pyfixest.estimation.models.base_regression_ import BaseRegression


class FittedModel(Protocol):
    """Structural contract consumed by the generic estimation pipeline."""

    _X_is_empty: bool
    _is_iv: bool

    def prepare_model_matrix(self) -> object:
        """Prepare and retain estimator inputs derived from the formula."""
        ...

    def _validate_response(self) -> None:
        """Validate estimator-specific dependent-variable constraints."""
        ...

    def get_fit(self) -> object:
        """Estimate the model parameters."""
        ...

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
    ) -> object:
        """Compute the requested covariance matrix."""
        ...

    def get_inference(self) -> object:
        """Compute coefficient-level inference from the covariance matrix."""
        ...

    def _finalize_fit(self) -> None:
        """Run estimator-specific post-fit orchestration."""
        ...

    def _iter_fitted_models(self) -> Iterable[BaseRegression]:
        """Yield concrete fitted results produced by this pipeline object."""
        ...

    def _clear_attributes(self) -> None:
        """Clear large state according to storage options."""
        ...


class FittedResult(Protocol):
    """Structural contract every fitted estimation result satisfies.

    This is the surface the result container, the reporting functions, and the
    multiple-testing corrections read: enough to identify a model, describe its
    estimation sample and covariance, and report coefficient-level inference.
    Estimator-specific post-estimation methods are deliberately absent, because
    only some result classes define them; ask `capabilities()` instead.

    Annotations name this protocol. Runtime `isinstance` checks name
    `BaseRegression`, the class that implements it.
    """

    # Identity and specification.
    _method: str
    _model_name: str
    _model_name_plot: str
    _fml: str
    _depvar: str
    _fixef: str | None
    _icovars: list[str] | None
    _coefnames: list[str]
    _is_iv: bool
    _sample_split_var: str | None
    _sample_split_value: Any
    _na_index: frozenset[int]

    # Estimation sample and covariance.
    _N: int | float
    _N_rows: int
    _k: int
    _vcov_type: str
    _vcov_type_detail: str
    _is_clustered: bool
    _G: list[int]

    # Goodness of fit. `deviance` is None outside the GLM families.
    _rmse: float
    _r2: float
    _r2_within: float
    deviance: float | None

    def tidy(
        self,
        alpha: float = 0.05,
        inference_type: InferenceType = "regular",
    ) -> pd.DataFrame:
        """Return point estimates and coefficient-level inference."""
        ...

    def coef(self) -> pd.Series:
        """Return the estimated coefficients."""
        ...

    def se(self) -> pd.Series:
        """Return the coefficient standard errors."""
        ...

    def tstat(self) -> pd.Series:
        """Return the coefficient t-statistics."""
        ...

    def pvalue(self) -> pd.Series:
        """Return the coefficient p-values."""
        ...

    def confint(
        self,
        alpha: float = 0.05,
        *,
        inference_type: InferenceType = "regular",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Return coefficient confidence intervals."""
        ...

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
    ) -> FittedResult:
        """Replace the covariance estimate and the inference derived from it."""
        ...

    def get_inference(self, alpha: float = 0.05) -> None:
        """Recompute coefficient-level inference from the covariance matrix."""
        ...


ModelFactory: TypeAlias = Callable[..., FittedModel]
