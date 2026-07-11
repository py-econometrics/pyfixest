"""Define the immutable configuration passed through the estimation pipeline."""

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
from pyfixest.typing import (
    QuantregVcovType,
    RegressionVcovType,
    SscConfig,
    VcovKwargs,
    WeightsType,
)


@dataclass(frozen=True)
class EstimationConfig:
    """Immutable record of what an estimation call requests.

    A single flat container for every function argument of the
    public `feols`, `feglm`, `fepois`, `quantreg` APIs plus info
    on which model is to be fitted / to which method we dispatch.
    """

    # --- dispatch ---
    method: str

    # --- data ---
    data: Any

    # --- formula ---
    fml: str

    # --- data flags ---
    copy_data: bool = True
    store_data: bool = True
    lean: bool = False

    # --- formula extras ---
    fixef_rm: str = "singleton"
    drop_intercept: bool = False

    # --- vcov ---
    vcov: RegressionVcovType | QuantregVcovType | dict[str, str] | None = None
    vcov_kwargs: VcovKwargs | None = None
    ssc_dict: SscConfig | None = None

    # --- fit knobs ---
    solver: SolverOptions = "scipy.linalg.solve"
    demeaner: AnyDemeaner | None = None
    collin_tol: float = 1e-9
    context: Mapping[str, Any] = field(default_factory=dict)

    # --- weights ---
    weights: str | None = None
    weights_type: WeightsType = "aweights"

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
