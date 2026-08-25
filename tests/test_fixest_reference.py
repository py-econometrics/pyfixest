from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reference import compare_fixest
from scripts.reference.compare_fixest import _report_payload, write_report
from scripts.reference.fixest_reference import (
    AdapterInfo,
    DataSpec,
    NormalizedResult,
    ReferenceCase,
    SmallSampleCorrection,
    VcovSpec,
    compare_results,
    hash_dataframe,
    load_case,
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
        prediction_rtol=1e-8,
        prediction_atol=1e-8,
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


def test_numerical_difference_at_tolerance_boundary_passes():
    reference = _result()
    pyfixest = replace(reference, coefficients=np.array([1.0, 2.5]))

    report = compare_results(
        case=replace(_case(), rtol=0.0, atol=0.5),
        pyfixest=pyfixest,
        reference=reference,
        pyfixest_info=AdapterInfo("pyfixest", "1", "python"),
        reference_info=AdapterInfo("fixest", "1", "R"),
    )

    coefficient_metric = next(
        metric for metric in report.metrics if metric.name == "coefficients"
    )
    assert coefficient_metric.passed


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


def test_load_case_records_reproducible_inputs():
    case = load_case(Path("scripts/reference/cases/feols-smoke.toml"))

    assert case.estimator == "feols"
    assert case.data.seed == 9289
    assert case.vcov.kind == "iid"
    assert len(case.case_hash) == 64


def test_load_case_rejects_string_booleans(tmp_path):
    source = Path("scripts/reference/cases/feols-smoke.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("k_adj = true", 'k_adj = "false"'))

    with np.testing.assert_raises_regex(RuntimeError, "ssc.k_adj"):
        load_case(path)


def test_load_case_accepts_zero_seed(tmp_path):
    source = Path("scripts/reference/cases/feols-smoke.toml").read_text()
    path = tmp_path / "zero-seed.toml"
    path.write_text(source.replace("seed = 9289", "seed = 0"))

    assert load_case(path).data.seed == 0


def test_dataframe_hash_changes_with_values():
    first = pd.DataFrame({"x": [1.0, 2.0]})
    second = pd.DataFrame({"x": [1.0, 3.0]})

    assert hash_dataframe(first) != hash_dataframe(second)


def test_write_report_refuses_to_overwrite(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, {"passed": True})

    with np.testing.assert_raises_regex(
        RuntimeError,
        "refusing to overwrite",
    ):
        write_report(path, {"passed": False})


def test_report_payload_records_versions_hashes_and_values():
    result = _result()
    report = compare_results(
        case=_case(),
        pyfixest=result,
        reference=result,
        pyfixest_info=AdapterInfo("pyfixest", "1", "python"),
        reference_info=AdapterInfo("fixest", "2", "R"),
    )

    payload = _report_payload(
        report=report,
        data_hash="data-hash",
        command=["compare-fixest", "case.toml"],
        pyfixest_result=result,
        reference_result=result,
    )

    assert payload["case"]["case_hash"] == "hash"
    assert payload["adapters"]["reference"]["package_version"] == "2"
    assert payload["provenance"]["data_hash"] == "data-hash"
    assert payload["provenance"]["command"] == "compare-fixest case.toml"
    assert payload["normalized_results"]["reference"]["nobs"] == 10


def test_cli_maps_unexpected_execution_failure_to_exit_code_two(monkeypatch, capsys):
    def _fail(**kwargs):
        raise ImportError("reference package is unavailable")

    monkeypatch.setattr(compare_fixest, "run_comparison", _fail)

    exit_code = compare_fixest.main(["scripts/reference/cases/feols-smoke.toml"])

    assert exit_code == 2
    assert "reference package is unavailable" in capsys.readouterr().out
