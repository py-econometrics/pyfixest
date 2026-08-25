"""Compare one reproducible pyfixest case with R fixest."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reference.fixest_reference import (
    ComparisonReport,
    NormalizedResult,
    PyfixestAdapter,
    ReferenceError,
    RFixestAdapter,
    hash_dataframe,
    load_case,
    load_case_data,
    run_comparison,
)


def _format_number(value: float | None) -> str:
    """Format an optional numerical difference for the report table."""
    return "-" if value is None else f"{value:.6g}"


def format_report(report: ComparisonReport) -> str:
    """Format one comparison for human review."""
    lines = [
        f"Case: {report.case.id}",
        f"Formula: {report.case.formula}",
        (
            f"pyfixest {report.pyfixest.package_version} "
            f"({report.pyfixest.runtime_version})"
        ),
        (
            f"R fixest {report.reference.package_version} "
            f"({report.reference.runtime_version})"
        ),
        "",
        "Metric | Status | Max abs diff | Max rel diff | Detail",
        "--- | --- | ---: | ---: | ---",
    ]
    for metric in report.metrics:
        lines.append(
            " | ".join(
                [
                    metric.name,
                    "PASS" if metric.passed else "FAIL",
                    _format_number(metric.max_absolute_difference),
                    _format_number(metric.max_relative_difference),
                    metric.detail,
                ]
            )
        )
    lines.extend(["", f"Overall: {'PASS' if report.passed else 'FAIL'}"])
    return "\n".join(lines)


def _report_payload(
    *,
    report: ComparisonReport,
    data_hash: str,
    command: Sequence[str],
    pyfixest_result: NormalizedResult | None,
    reference_result: NormalizedResult | None,
) -> dict[str, object]:
    """Build a provenance-rich JSON report."""
    payload = report.to_dict()
    payload["provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "python": sys.version,
        "platform": platform.platform(),
        "data_hash": data_hash,
    }
    if pyfixest_result is not None and reference_result is not None:
        payload["normalized_results"] = {
            "pyfixest": pyfixest_result.to_dict(),
            "reference": reference_result.to_dict(),
        }
    return payload


def write_report(path: Path, payload: dict[str, object]) -> None:
    """Write a report without silently replacing an existing artifact."""
    if path.exists():
        raise ReferenceError(f"refusing to overwrite existing report: {path}")
    if not path.parent.exists():
        raise ReferenceError(f"report directory does not exist: {path.parent}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Versioned TOML comparison case.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument(
        "--json-output",
        type=Path,
        help="Write summary JSON to a new explicit path.",
    )
    outputs.add_argument(
        "--record",
        type=Path,
        help="Write summary and normalized values to a new explicit path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixest comparison CLI."""
    args = _parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve())]
    command.extend(argv if argv is not None else sys.argv[1:])
    try:
        case = load_case(args.case)
        data = load_case_data(case)
        report, pyfixest_result, reference_result = run_comparison(
            case=case,
            data=data,
            pyfixest_adapter=PyfixestAdapter(),
            reference_adapter=RFixestAdapter(),
        )
        print(format_report(report))

        output = args.record or args.json_output
        if output is not None:
            record_values = args.record is not None
            write_report(
                output,
                _report_payload(
                    report=report,
                    data_hash=hash_dataframe(data),
                    command=command,
                    pyfixest_result=pyfixest_result if record_values else None,
                    reference_result=reference_result if record_values else None,
                ),
            )
    # The CLI boundary maps dependency and execution failures to documented code 2.
    except Exception as exc:
        print(f"error: {exc}")
        return 2
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
