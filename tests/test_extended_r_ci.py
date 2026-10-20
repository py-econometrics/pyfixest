"""Exercise the reference CI report gate without requiring R packages."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_extended_r_results",
    Path(__file__).parents[1] / "scripts" / "check_extended_r_results.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORT)


@pytest.mark.parametrize(
    "outcome, expected_status, expected_summary",
    [
        ("passed", 0, "4 passed, 0 skipped, 0 failed, 0 errors"),
        ("skipped", 1, "Required comparisons did not pass: avlm"),
        ("failure", 1, "3 passed, 0 skipped, 1 failed, 0 errors"),
        ("error", 1, "3 passed, 0 skipped, 0 failed, 1 errors"),
        ("unrelated_skip", 0, "4 passed, 1 skipped, 0 failed, 0 errors"),
        ("required_partial_skip", 1, "Required comparisons were skipped: avlm"),
        ("missing", 1, "no test report; qualification failed"),
        (
            "empty",
            1,
            "Required comparisons did not pass: avlm, did2s, ivDiag, wildrwolf",
        ),
    ],
)
def test_extended_reference_report_gate(
    tmp_path, monkeypatch, capsys, outcome, expected_status, expected_summary
):
    report = tmp_path / "results.xml"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    if outcome not in ("missing", "empty"):
        for index, module in enumerate(_REPORT.REQUIRED_COMPARISONS.values()):
            case = ET.SubElement(suite, "testcase", classname=module, name="comparison")
            if index == 0 and outcome in ("skipped", "failure", "error"):
                ET.SubElement(case, outcome)
        if outcome == "unrelated_skip":
            case = ET.SubElement(
                suite, "testcase", classname="tests.other", name="intentional_skip"
            )
            ET.SubElement(case, "skipped")
        if outcome == "required_partial_skip":
            case = ET.SubElement(
                suite,
                "testcase",
                classname="tests.test_savi_vs_avlm",
                name="missing_comparison",
            )
            ET.SubElement(case, "skipped")
    if outcome != "missing":
        ET.ElementTree(root).write(report)

    assert _REPORT.main(report) == expected_status
    assert expected_summary in capsys.readouterr().out
    assert expected_summary in summary.read_text()
