"""Summarize extended-reference CI and reject missing required comparison groups."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Each CRAN reference must contribute passing comparisons. Other skips remain
# visible in the report without making unrelated, deliberately skipped cases fail.
REQUIRED_COMPARISONS = {
    "avlm": "tests.test_savi_vs_avlm",
    "did2s": "tests.test_did",
    "ivDiag": "tests.test_iv",
    "wildrwolf": "tests.test_multcomp",
}


def main(report: Path) -> int:
    """Write passed/skipped counts and fail if a required reference did not run."""
    if not report.exists():
        summary = "Extended R references: no test report; qualification failed.\n"
        failed = True
    else:
        root = ET.parse(report).getroot()
        counts: Counter[str] = Counter()
        passed: Counter[str] = Counter()
        skipped = []
        for case in root.iter("testcase"):
            outcome = next(
                (
                    kind
                    for kind in ("error", "failure", "skipped")
                    if case.find(kind) is not None
                ),
                "passed",
            )
            counts[outcome] += 1
            if outcome == "passed":
                passed[case.get("classname", "")] += 1
            elif outcome == "skipped":
                skipped.append(f"{case.get('classname')}.{case.get('name')}")
        missing = [
            package
            for package, module in REQUIRED_COMPARISONS.items()
            if not passed[module]
        ]
        summary = (
            "Extended R references: "
            f"{counts['passed']} passed, {counts['skipped']} skipped, "
            f"{counts['failure']} failed, {counts['error']} errors.\n"
        )
        for package, module in REQUIRED_COMPARISONS.items():
            summary += f"- {package}: {passed[module]} passed\n"
        if skipped:
            summary += (
                "Skipped comparisons:\n"
                + "\n".join(f"- {case}" for case in skipped)
                + "\n"
            )
        if missing:
            summary += f"Required comparisons did not pass: {', '.join(missing)}.\n"
        failed = bool(missing or counts["failure"] or counts["error"])
    print(summary, end="")
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_file).open("a") as output:
            output.write(summary)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
