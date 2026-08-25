"""Normalized pyfixest-to-fixest comparison building blocks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
import tomllib
from numpy.typing import NDArray

Estimator = Literal["feols", "fepois"]
DataSource = Literal["generated", "csv"]


class ReferenceError(RuntimeError):
    """Raised when a reference case or adapter cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class SmallSampleCorrection:
    """Small-sample correction shared by pyfixest and fixest."""

    k_adj: bool = True
    k_fixef: str = "nonnested"
    g_adj: bool = True
    g_df: str = "min"
    t_df: str = "min"


@dataclass(frozen=True, slots=True)
class VcovSpec:
    """Variance-covariance case specification."""

    kind: str = "iid"
    cluster: str | None = None

    def for_pyfixest(self) -> str | dict[str, str]:
        """Return the pyfixest vcov argument."""
        if self.kind in {"CRV1", "CRV3"}:
            if self.cluster is None:
                raise ReferenceError(f"{self.kind} requires a cluster column")
            return {self.kind: self.cluster}
        return self.kind


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Deterministic input-data specification."""

    source: DataSource
    seed: int | None = None
    n: int | None = None
    model: str | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    """One reproducible pyfixest-to-fixest comparison case."""

    id: str
    estimator: Estimator
    formula: str
    data: DataSpec
    vcov: VcovSpec
    ssc: SmallSampleCorrection
    weights: str | None
    prediction_rows: int
    rtol: float
    atol: float
    prediction_rtol: float
    prediction_atol: float
    case_hash: str
    source_path: Path

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable case description."""
        return {
            "id": self.id,
            "estimator": self.estimator,
            "formula": self.formula,
            "data": {
                **asdict(self.data),
                "path": str(self.data.path) if self.data.path else None,
            },
            "vcov": asdict(self.vcov),
            "ssc": asdict(self.ssc),
            "weights": self.weights,
            "prediction_rows": self.prediction_rows,
            "rtol": self.rtol,
            "atol": self.atol,
            "prediction_rtol": self.prediction_rtol,
            "prediction_atol": self.prediction_atol,
            "case_hash": self.case_hash,
            "source_path": str(self.source_path),
        }


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    """Version information for one comparison adapter."""

    name: str
    package_version: str
    runtime_version: str


@dataclass(frozen=True, slots=True)
class NormalizedResult:
    """Named numerical results shared by both adapters."""

    coefficient_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    vcov: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    degrees_of_freedom: float
    nobs: int
    dropped_variables: tuple[str, ...]
    converged: bool | None
    predictions: NDArray[np.float64] | None

    def __post_init__(self) -> None:
        """Validate normalized result shapes."""
        size = len(self.coefficient_names)
        if len(set(self.coefficient_names)) != size:
            raise ReferenceError("normalized coefficient names must be unique")
        if self.coefficients.shape != (size,):
            raise ReferenceError("coefficient vector does not match coefficient names")
        if self.standard_errors.shape != (size,):
            raise ReferenceError("standard errors do not match coefficient names")
        if self.vcov.shape != (size, size):
            raise ReferenceError("vcov shape does not match coefficient names")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable normalized values."""
        return {
            "coefficient_names": list(self.coefficient_names),
            "coefficients": self.coefficients.tolist(),
            "vcov": self.vcov.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "degrees_of_freedom": self.degrees_of_freedom,
            "nobs": self.nobs,
            "dropped_variables": list(self.dropped_variables),
            "converged": self.converged,
            "predictions": (
                self.predictions.tolist() if self.predictions is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Comparison outcome for one normalized quantity."""

    name: str
    passed: bool
    max_absolute_difference: float | None
    max_relative_difference: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Complete comparison result for one case."""

    case: ReferenceCase
    pyfixest: AdapterInfo
    reference: AdapterInfo
    metrics: tuple[MetricComparison, ...]

    @property
    def passed(self) -> bool:
        """Return whether every metric passed."""
        return all(metric.passed for metric in self.metrics)

    def to_dict(self, *, include_values: bool = False) -> dict[str, object]:
        """Return a versioned JSON-serializable report."""
        payload: dict[str, object] = {
            "schema_version": 1,
            "passed": self.passed,
            "case": self.case.to_dict(),
            "adapters": {
                "pyfixest": asdict(self.pyfixest),
                "reference": asdict(self.reference),
            },
            "metrics": [asdict(metric) for metric in self.metrics],
        }
        if include_values:
            payload["note"] = "normalized values are supplied by the CLI"
        return payload


class ReferenceAdapter(Protocol):
    """Adapter contract for normalized estimator output."""

    def info(self) -> AdapterInfo:
        """Return package and runtime versions."""

    def fit(self, case: ReferenceCase, data: pd.DataFrame) -> NormalizedResult:
        """Fit one case and return normalized output."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """Validate and return a TOML table."""
    if not isinstance(value, dict):
        raise ReferenceError(f"{name} must be a TOML table")
    return value


def _string(value: object, name: str) -> str:
    """Validate and return a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ReferenceError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    """Validate and return a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReferenceError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    """Validate and return a non-negative integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReferenceError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: object, name: str) -> bool:
    """Validate and return a Boolean."""
    if not isinstance(value, bool):
        raise ReferenceError(f"{name} must be true or false")
    return value


def _choice(value: object, name: str, choices: set[str]) -> str:
    """Validate and return one enumerated string."""
    result = _string(value, name)
    if result not in choices:
        allowed = ", ".join(sorted(choices))
        raise ReferenceError(f"{name} must be one of: {allowed}")
    return result


def _tolerance(value: object, name: str) -> float:
    """Validate and return a finite non-negative tolerance."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ReferenceError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ReferenceError(f"{name} must be finite and non-negative")
    return result


def load_case(path: Path) -> ReferenceCase:
    """Load and validate a versioned TOML comparison case."""
    try:
        raw_bytes = path.read_bytes()
        payload = tomllib.loads(raw_bytes.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ReferenceError(f"cannot read case {path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise ReferenceError("case schema_version must be 1")
    estimator = _string(payload.get("estimator"), "estimator")
    if estimator not in {"feols", "fepois"}:
        raise ReferenceError("estimator must be 'feols' or 'fepois'")

    data = _load_data_spec(_mapping(payload.get("data"), "data"), path.parent)
    vcov = _load_vcov_spec(_mapping(payload.get("vcov", {}), "vcov"))
    ssc = _load_ssc(_mapping(payload.get("ssc", {}), "ssc"))
    weights = payload.get("weights")
    if weights is not None and not isinstance(weights, str):
        raise ReferenceError("weights must be a column name")

    prediction_rows = _positive_int(
        payload.get("prediction_rows", 5), "prediction_rows"
    )
    rtol = _tolerance(payload.get("rtol", 1e-8), "rtol")
    atol = _tolerance(payload.get("atol", 1e-8), "atol")
    prediction_rtol = _tolerance(
        payload.get("prediction_rtol", rtol), "prediction_rtol"
    )
    prediction_atol = _tolerance(
        payload.get("prediction_atol", atol), "prediction_atol"
    )

    return ReferenceCase(
        id=_string(payload.get("id"), "id"),
        estimator=estimator,
        formula=_string(payload.get("formula"), "formula"),
        data=data,
        vcov=vcov,
        ssc=ssc,
        weights=weights,
        prediction_rows=prediction_rows,
        rtol=rtol,
        atol=atol,
        prediction_rtol=prediction_rtol,
        prediction_atol=prediction_atol,
        case_hash=hashlib.sha256(raw_bytes).hexdigest(),
        source_path=path.resolve(),
    )


def _load_data_spec(payload: Mapping[str, object], parent: Path) -> DataSpec:
    """Build and validate the data section of a case."""
    source = _string(payload.get("source"), "data.source")
    if source == "generated":
        return DataSpec(
            source="generated",
            seed=_nonnegative_int(payload.get("seed"), "data.seed"),
            n=_positive_int(payload.get("n"), "data.n"),
            model=_string(payload.get("model"), "data.model"),
        )
    if source == "csv":
        relative_path = Path(_string(payload.get("path"), "data.path"))
        return DataSpec(source="csv", path=(parent / relative_path).resolve())
    raise ReferenceError("data.source must be 'generated' or 'csv'")


def _load_vcov_spec(payload: Mapping[str, object]) -> VcovSpec:
    """Build and validate the vcov section of a case."""
    kind = _string(payload.get("type", "iid"), "vcov.type")
    if kind not in {"iid", "hetero", "CRV1", "CRV3"}:
        raise ReferenceError("vcov.type must be iid, hetero, CRV1, or CRV3")
    cluster = payload.get("cluster")
    if cluster is not None and not isinstance(cluster, str):
        raise ReferenceError("vcov.cluster must be a column name")
    return VcovSpec(kind=kind, cluster=cluster)


def _load_ssc(payload: Mapping[str, object]) -> SmallSampleCorrection:
    """Build the shared small-sample-correction specification."""
    return SmallSampleCorrection(
        k_adj=_boolean(payload.get("k_adj", True), "ssc.k_adj"),
        k_fixef=_choice(
            payload.get("k_fixef", "nonnested"),
            "ssc.k_fixef",
            {"none", "full", "nonnested"},
        ),
        g_adj=_boolean(payload.get("g_adj", True), "ssc.g_adj"),
        g_df=_choice(
            payload.get("g_df", "min"),
            "ssc.g_df",
            {"conventional", "min"},
        ),
        t_df=_choice(
            payload.get("t_df", "min"),
            "ssc.t_df",
            {"conventional", "min"},
        ),
    )


def load_case_data(case: ReferenceCase) -> pd.DataFrame:
    """Load the exact pandas data used by both adapters."""
    if case.data.source == "csv":
        if case.data.path is None:
            raise ReferenceError("CSV case has no resolved data path")
        try:
            return pd.read_csv(case.data.path)
        except OSError as exc:
            raise ReferenceError(
                f"cannot read case data {case.data.path}: {exc}"
            ) from exc

    import pyfixest as pf

    return pf.get_data(
        N=case.data.n,
        seed=case.data.seed,
        model=case.data.model,
    )


def hash_dataframe(data: pd.DataFrame) -> str:
    """Return a stable hash of values, dtypes, columns, and index."""
    value_hash = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
    schema = "\n".join(
        f"{column}:{dtype}"
        for column, dtype in zip(data.columns, data.dtypes, strict=True)
    ).encode()
    return hashlib.sha256(schema + value_hash).hexdigest()


def normalize_term_name(name: str) -> str:
    """Normalize understood cross-language coefficient-name differences."""
    normalized = "Intercept" if name == "(Intercept)" else name
    return normalized.removeprefix("fit_")


def translate_formula_for_fixest(formula: str) -> str:
    """Translate supported Formulaic spellings to R fixest spellings."""
    translated = re.sub(
        r"C\((.*?)\)",
        r"factor(\1, exclude = NA)",
        formula,
    )
    parts = translated.split("|")
    for index, part in enumerate(parts[1:], start=1):
        if "~" not in part:
            parts[index] = part.replace(":", "^")
            break
    return "|".join(parts)


class PyfixestAdapter:
    """Fit a case with the local pyfixest checkout."""

    def info(self) -> AdapterInfo:
        """Return local pyfixest and Python versions."""
        import platform

        return AdapterInfo(
            name="pyfixest",
            package_version=importlib.metadata.version("pyfixest"),
            runtime_version=platform.python_version(),
        )

    def fit(self, case: ReferenceCase, data: pd.DataFrame) -> NormalizedResult:
        """Fit and normalize one pyfixest case."""
        import pyfixest as pf

        estimator = getattr(pf, case.estimator)
        kwargs: dict[str, object] = {
            "fml": case.formula,
            "data": data,
            "vcov": case.vcov.for_pyfixest(),
            "ssc": pf.ssc(
                k_adj=case.ssc.k_adj,
                k_fixef=case.ssc.k_fixef,
                G_adj=case.ssc.g_adj,
                G_df=case.ssc.g_df,
            ),
        }
        if case.weights is not None:
            kwargs["weights"] = case.weights
        model = estimator(**kwargs)

        names = tuple(normalize_term_name(str(name)) for name in model.coef().index)
        predictions = _py_predictions(model, data=data, rows=case.prediction_rows)
        return NormalizedResult(
            coefficient_names=names,
            coefficients=np.asarray(model.coef(), dtype=np.float64),
            vcov=np.asarray(model._vcov, dtype=np.float64),
            standard_errors=np.asarray(model.se(), dtype=np.float64),
            degrees_of_freedom=float(model._df_t),
            nobs=int(model._N),
            dropped_variables=tuple(
                normalize_term_name(str(name)) for name in model._collin_vars
            ),
            converged=bool(getattr(model, "convergence", True)),
            predictions=predictions,
        )


def _py_predictions(
    model: object,
    *,
    data: pd.DataFrame,
    rows: int,
) -> NDArray[np.float64]:
    """Return a deterministic pyfixest prediction subset."""
    values = model.predict(newdata=data.iloc[:rows], type="response")
    return np.asarray(values, dtype=np.float64).reshape(-1)


class RFixestAdapter:
    """Fit a case with R fixest through a lazy rpy2 boundary."""

    def info(self) -> AdapterInfo:
        """Return installed fixest and R versions."""
        try:
            import rpy2.robjects as ro
        except ImportError as exc:
            raise ReferenceError(
                "R comparison requires the py312-r Pixi environment"
            ) from exc

        package_version = str(ro.r('as.character(packageVersion("fixest"))')[0])
        runtime_version = str(ro.r("R.version.string")[0])
        return AdapterInfo(
            name="R fixest",
            package_version=package_version,
            runtime_version=runtime_version,
        )

    def fit(self, case: ReferenceCase, data: pd.DataFrame) -> NormalizedResult:
        """Fit and normalize one R fixest case."""
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.packages import importr
        except ImportError as exc:
            raise ReferenceError(
                "R comparison requires the py312-r Pixi environment"
            ) from exc

        fixest = importr("fixest")
        stats = importr("stats")
        converter = ro.default_converter + numpy2ri.converter + pandas2ri.converter
        with converter.context():
            data_r = ro.conversion.get_conversion().py2rpy(data)
            newdata_r = ro.conversion.get_conversion().py2rpy(
                data.iloc[: case.prediction_rows]
            )

        kwargs = self._fit_kwargs(
            case=case,
            data_r=data_r,
            ro=ro,
            fixest=fixest,
        )
        estimator = fixest.feols if case.estimator == "feols" else fixest.fepois
        model = estimator(
            ro.Formula(translate_formula_for_fixest(case.formula)), **kwargs
        )
        coefficients = stats.coef(model)
        ro.globalenv[".pyfixest_reference_model"] = model
        r_names = ro.r("names(coef(.pyfixest_reference_model))")
        names = tuple(normalize_term_name(str(name)) for name in r_names)
        predictions = np.asarray(
            stats.predict(model, newdata=newdata_r, type="response"),
            dtype=np.float64,
        ).reshape(-1)
        df_t = ro.r('attr(vcov(.pyfixest_reference_model), "df.t")')
        dropped = ro.r(
            "if (is.null(.pyfixest_reference_model$collin.var)) "
            "character() else .pyfixest_reference_model$collin.var"
        )
        convergence = ro.r(
            "if (is.null(.pyfixest_reference_model$convStatus)) "
            "TRUE else .pyfixest_reference_model$convStatus"
        )
        return NormalizedResult(
            coefficient_names=names,
            coefficients=np.asarray(coefficients, dtype=np.float64),
            vcov=np.asarray(stats.vcov(model), dtype=np.float64),
            standard_errors=np.asarray(model.rx2("se"), dtype=np.float64),
            degrees_of_freedom=float(df_t[0]),
            nobs=int(stats.nobs(model)[0]),
            dropped_variables=tuple(normalize_term_name(str(name)) for name in dropped),
            converged=bool(convergence[0]),
            predictions=predictions,
        )

    def _fit_kwargs(
        self,
        *,
        case: ReferenceCase,
        data_r: object,
        ro: object,
        fixest: object,
    ) -> dict[str, object]:
        """Build R fixest arguments from the shared case."""
        vcov: object = case.vcov.kind
        if case.vcov.kind in {"CRV1", "CRV3"}:
            if case.vcov.cluster is None:
                raise ReferenceError(f"{case.vcov.kind} requires a cluster column")
            vcov = ro.Formula(f"~{case.vcov.cluster}")
        kwargs: dict[str, object] = {
            "data": data_r,
            "vcov": vcov,
            "ssc": fixest.ssc(
                case.ssc.k_adj,
                case.ssc.k_fixef,
                False,
                case.ssc.g_adj,
                case.ssc.g_df,
                case.ssc.t_df,
            ),
        }
        if case.weights is not None:
            kwargs["weights"] = ro.Formula(f"~{case.weights}")
        return kwargs


def compare_results(
    *,
    case: ReferenceCase,
    pyfixest: NormalizedResult,
    reference: NormalizedResult,
    pyfixest_info: AdapterInfo,
    reference_info: AdapterInfo,
) -> ComparisonReport:
    """Compare normalized output by coefficient name and structural metadata."""
    metrics: list[MetricComparison] = []
    py_names = pyfixest.coefficient_names
    ref_names = reference.coefficient_names
    same_terms = set(py_names) == set(ref_names) and len(py_names) == len(ref_names)
    metrics.append(
        _structural_metric(
            "coefficient_names",
            same_terms,
            f"pyfixest={py_names}; reference={ref_names}",
        )
    )

    if same_terms:
        order = [py_names.index(name) for name in ref_names]
        py_coef = pyfixest.coefficients[order]
        py_se = pyfixest.standard_errors[order]
        py_vcov = pyfixest.vcov[np.ix_(order, order)]
        metrics.extend(
            [
                _numeric_metric(
                    "coefficients",
                    py_coef,
                    reference.coefficients,
                    rtol=case.rtol,
                    atol=case.atol,
                ),
                _numeric_metric(
                    "vcov",
                    py_vcov,
                    reference.vcov,
                    rtol=case.rtol,
                    atol=case.atol,
                ),
                _numeric_metric(
                    "standard_errors",
                    py_se,
                    reference.standard_errors,
                    rtol=case.rtol,
                    atol=case.atol,
                ),
            ]
        )

    metrics.extend(
        [
            _structural_metric(
                "degrees_of_freedom",
                pyfixest.degrees_of_freedom == reference.degrees_of_freedom,
                (
                    f"pyfixest={pyfixest.degrees_of_freedom}; "
                    f"reference={reference.degrees_of_freedom}"
                ),
            ),
            _structural_metric(
                "nobs",
                pyfixest.nobs == reference.nobs,
                f"pyfixest={pyfixest.nobs}; reference={reference.nobs}",
            ),
            _structural_metric(
                "dropped_variables",
                set(pyfixest.dropped_variables) == set(reference.dropped_variables),
                (
                    f"pyfixest={pyfixest.dropped_variables}; "
                    f"reference={reference.dropped_variables}"
                ),
            ),
            _structural_metric(
                "convergence",
                pyfixest.converged == reference.converged,
                f"pyfixest={pyfixest.converged}; reference={reference.converged}",
            ),
        ]
    )

    if pyfixest.predictions is not None or reference.predictions is not None:
        if pyfixest.predictions is None or reference.predictions is None:
            metrics.append(
                _structural_metric(
                    "predictions",
                    False,
                    "prediction availability differs between adapters",
                )
            )
        else:
            metrics.append(
                _numeric_metric(
                    "predictions",
                    pyfixest.predictions,
                    reference.predictions,
                    rtol=case.prediction_rtol,
                    atol=case.prediction_atol,
                )
            )

    return ComparisonReport(
        case=case,
        pyfixest=pyfixest_info,
        reference=reference_info,
        metrics=tuple(metrics),
    )


def _structural_metric(name: str, passed: bool, detail: str) -> MetricComparison:
    """Build an exact structural comparison."""
    return MetricComparison(
        name=name,
        passed=passed,
        max_absolute_difference=None,
        max_relative_difference=None,
        detail=detail,
    )


def _numeric_metric(
    name: str,
    actual: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> MetricComparison:
    """Build one shape-aware allclose comparison."""
    if actual.shape != expected.shape:
        return MetricComparison(
            name=name,
            passed=False,
            max_absolute_difference=None,
            max_relative_difference=None,
            detail=f"shape mismatch: pyfixest={actual.shape}; reference={expected.shape}",
        )
    if actual.size == 0:
        max_absolute = 0.0
        max_relative = 0.0
    else:
        difference = np.abs(actual - expected)
        denominator = np.maximum(np.abs(expected), atol or np.finfo(float).eps)
        max_absolute = float(np.nanmax(difference))
        max_relative = float(np.nanmax(difference / denominator))
    passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
    return MetricComparison(
        name=name,
        passed=passed,
        max_absolute_difference=max_absolute,
        max_relative_difference=max_relative,
        detail=f"rtol={rtol}; atol={atol}",
    )


def run_comparison(
    *,
    case: ReferenceCase,
    data: pd.DataFrame,
    pyfixest_adapter: ReferenceAdapter,
    reference_adapter: ReferenceAdapter,
) -> tuple[ComparisonReport, NormalizedResult, NormalizedResult]:
    """Fit both adapters and compare their normalized results."""
    py_result = pyfixest_adapter.fit(case, data)
    reference_result = reference_adapter.fit(case, data)
    report = compare_results(
        case=case,
        pyfixest=py_result,
        reference=reference_result,
        pyfixest_info=pyfixest_adapter.info(),
        reference_info=reference_adapter.info(),
    )
    return report, py_result, reference_result
