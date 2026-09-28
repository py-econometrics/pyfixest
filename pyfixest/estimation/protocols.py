from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from pyfixest.estimation.internals.model_state import VcovSpec
    from pyfixest.estimation.models.feols_ import Feols


class FittedModel(Protocol):
    """Structural contract consumed by the generic estimation pipeline."""

    _X_is_empty: bool

    def prepare_model_matrix(self) -> object:
        """Prepare and retain estimator inputs derived from the formula."""
        ...

    def _validate_response(self) -> None:
        """Validate estimator-specific dependent-variable constraints."""
        ...

    def get_fit(self) -> object:
        """Estimate the model parameters."""
        ...

    def _check_vcov_support(self, spec: VcovSpec) -> None:
        """Reject a covariance estimator the model cannot compute."""
        ...

    def _vcov_from_spec(self, spec: VcovSpec) -> object:
        """Compute the covariance matrix of a parsed, supported estimator."""
        ...

    def get_inference(self) -> object:
        """Compute coefficient-level inference from the covariance matrix."""
        ...

    def _finalize_fit(self) -> None:
        """Run estimator-specific post-fit orchestration."""
        ...

    def _iter_fitted_models(self) -> Iterable[Feols]:
        """Yield concrete fitted results produced by this pipeline object."""
        ...

    def _clear_attributes(self) -> None:
        """Clear large state according to storage options."""
        ...


ModelFactory: TypeAlias = Callable[..., FittedModel]
