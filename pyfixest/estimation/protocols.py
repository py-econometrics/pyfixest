from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
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
        data: Any = None,
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


ModelFactory: TypeAlias = Callable[..., FittedModel]
