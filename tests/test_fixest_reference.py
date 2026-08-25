from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reference.fixest_reference import (
    AdapterInfo,
    DataSpec,
    NormalizedResult,
    ReferenceCase,
    SmallSampleCorrection,
    VcovSpec,
    compare_results,
    normalize_term_name,
    run_comparison,
)


def _case() -> ReferenceCase:
    return ReferenceCase(
        id="unit",
        estimator="feols",
        formula="Y ~ X1",
        data=DataSpec(source="generated", seed=1, n=10, model="Feols"),
        vcov=VcovSpec(),
        ssc=SmallSampleCorrection(),
        weights=None,
        prediction_rows=2,
        rtol=1e-8,
        atol=1e-8,
        case_hash="hash",
        source_path=Path("unit.toml"),
    )


def _result() -> NormalizedResult:
    return NormalizedResult(
        coefficient_names=("Intercept", "X1"),
        coefficients=np.array([1.0, 2.0]),
        vcov=np.array([[0.2, 0.01], [0.01, 0.3]]),
        standard_errors=np.sqrt(np.array([0.2, 0.3])),
        degrees_of_freedom=8.0,
        nobs=10,
        dropped_variables=(),
        converged=True,
        predictions=np.array([1.5, 2.5]),
    )


class _StaticAdapter:
    def __init__(self, name: str, result: NormalizedResult):
        self._name = name
        self._result = result

    def info(self) -> AdapterInfo:
        return AdapterInfo(self._name, "1.0", "runtime")

    def fit(self, case, data):
        return self._result


def test_identical_normalized_results_pass():
    result = _result()
    report = compare_results(
        case=_case(),
        pyfixest=result,
        reference=result,
        pyfixest_info=AdapterInfo("pyfixest", "1", "python"),
        reference_info=AdapterInfo("fixest", "1", "R"),
    )

    assert report.passed
    assert all(metric.passed for metric in report.metrics)


def test_named_results_are_reordered_before_comparison():
    reference = _result()
    pyfixest = NormalizedResult(
        coefficient_names=("X1", "Intercept"),
        coefficients=reference.coefficients[::-1],
        vcov=reference.vcov[::-1, ::-1],
        standard_errors=reference.standard_errors[::-1],
        degrees_of_freedom=reference.degrees_of_freedom,
        nobs=reference.nobs,
        dropped_variables=reference.dropped_variables,
        converged=reference.converged,
        predictions=reference.predictions,
    )

    report = compare_results(
        case=_case(),
        pyfixest=pyfixest,
        reference=reference,
        pyfixest_info=AdapterInfo("pyfixest", "1", "python"),
        reference_info=AdapterInfo("fixest", "1", "R"),
    )

    assert report.passed


def test_numerical_mismatch_fails_with_maximum_difference():
    reference = _result()
    pyfixest = replace(reference, coefficients=np.array([1.0, 2.5]))

    report = compare_results(
        case=_case(),
        pyfixest=pyfixest,
        reference=reference,
        pyfixest_info=AdapterInfo("pyfixest", "1", "python"),
        reference_info=AdapterInfo("fixest", "1", "R"),
    )

    coefficient_metric = next(
        metric for metric in report.metrics if metric.name == "coefficients"
    )
    assert not report.passed
    assert coefficient_metric.max_absolute_difference == 0.5


def test_fake_adapters_drive_complete_comparison():
    result = _result()

    report, py_result, reference_result = run_comparison(
        case=_case(),
        data=pd.DataFrame({"Y": [1.0], "X1": [2.0]}),
        pyfixest_adapter=_StaticAdapter("pyfixest", result),
        reference_adapter=_StaticAdapter("reference", result),
    )

    assert report.passed
    assert py_result is result
    assert reference_result is result


def test_normalize_term_name_only_changes_understood_spellings():
    assert normalize_term_name("(Intercept)") == "Intercept"
    assert normalize_term_name("fit_X1") == "X1"
    assert normalize_term_name("factor(f1)2") == "factor(f1)2"
