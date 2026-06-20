from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pyfixest.demeaners import AnyDemeaner
from pyfixest.estimation.internals.literals import (
    QuantregMethodOptions,
    QuantregMultiOptions,
    SolverOptions,
)


@dataclass(frozen=True)
class EstimationConfig:
    """Immutable record of what an estimation call requests.

    A single flat container for every knob the public API exposes
    (`feols`, `fepois`, `feglm`, `quantreg`) plus the dispatch
    method string. The orchestrator and planner consume this object;
    method-specific fields (GLM, quantreg) carry inert defaults when
    they don't apply, so non-applicable methods can safely ignore them.
    """

    # --- dispatch ---
    method: str

    # --- data ---
    # IntoDataFrame (pandas / polars / narwhals); typed as Any so dataclass
    # field generation doesn't trip over the Protocol-like alias.
    data: Any

    # --- formula ---
    fml: str

    # --- data flags ---
    copy_data: bool = True
    store_data: bool = True
    lean: bool = False

    # --- formula extras ---
    fixef_rm: str = "none"
    drop_intercept: bool = False

    # --- vcov ---
    vcov: str | dict[str, str] | None = None
    vcov_kwargs: dict[str, str | int] | None = None
    ssc_dict: dict[str, str | bool] | None = None

    # --- fit knobs ---
    solver: SolverOptions = "scipy.linalg.solve"
    demeaner: AnyDemeaner | None = None
    collin_tol: float = 1e-6
    context: Mapping[str, Any] = field(default_factory=dict)

    # --- weights ---
    weights: str | None = None
    weights_type: str = "aweights"

    # --- splits ---
    split: str | None = None
    fsplit: str | None = None

    # --- GLM-only (ignored for non-GLM methods) ---
    iwls_tol: float = 1e-8
    iwls_maxiter: int = 25
    separation_check: list[str] | None = None
    offset: str | None = None
    accelerate: bool = True

    # --- quantreg-only (ignored otherwise) ---
    quantile: float | list[float] | None = None
    quantreg_method: QuantregMethodOptions = "fn"
    quantile_tol: float = 1e-6
    quantile_maxiter: int | None = None
    quantreg_multi_method: QuantregMultiOptions = "cfm1"
    seed: int | None = None  # consumed only by the quantreg "pfn" method
