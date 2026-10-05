"""Check that every collected test is selected by a CI pytest task."""

from __future__ import annotations

import itertools
import shlex
import sys
from pathlib import Path

import pytest
import tomllib
import yaml
from _pytest.mark.expression import Expression

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def ci_selections():
    """Read pytest task selections from the actual workflow and Pixi commands."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pixi"]
    tasks = dict(config["tasks"])
    tasks.update(config["feature"]["r"]["tasks"])
    r_environments = {
        name
        for name, spec in config["environments"].items()
        if "r" in (spec if isinstance(spec, list) else spec["features"])
    }
    selections = {}
    for workflow in ("ci-tests.yaml", "extended_tests.yaml"):
        jobs = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())[
            "jobs"
        ]
        for job in jobs.values():
            matrix = job.get("strategy", {}).get("matrix", {})
            for values in itertools.product(*matrix.values()):
                variant = dict(zip(matrix, values, strict=True))
                for step in job["steps"]:
                    command = step.get("run", "")
                    for key, value in variant.items():
                        command = command.replace(
                            "${{ matrix." + key + " }}", str(value)
                        )
                    args = shlex.split(command)
                    if args[:3] != ["pixi", "run", "-e"] or len(args) != 5:
                        continue
                    environment, task = args[3:]
                    task_args = shlex.split(tasks.get(task, {}).get("cmd", ""))
                    if not task_args or task_args[0] != "pytest":
                        continue
                    if "-k" in task_args:
                        raise ValueError(f"CI audit needs support for -k in {task}")
                    expression = (
                        task_args[task_args.index("-m") + 1]
                        if "-m" in task_args
                        else ""
                    )
                    selections[environment, task] = (
                        environment in r_environments,
                        Expression.compile(expression) if expression else None,
                        [arg for arg in task_args if arg.startswith("tests")],
                        [
                            arg.removeprefix("--ignore=")
                            for arg in task_args
                            if arg.startswith("--ignore=")
                        ],
                        "--rpy2-files" in task_args,
                    )
    if not selections:
        raise ValueError("No CI pytest tasks found")
    return selections


def path_matches(nodeid, path):
    """Match a pytest directory, file, or explicit node selection."""
    return (
        nodeid == path
        or nodeid.startswith(path + "/")
        or nodeid.startswith(path + "::")
    )


class CoverageAudit:
    """Audit the full collection before any test is executed."""

    def pytest_collection_finish(self, session):
        """Fail collection when a test has no matching CI selection."""
        from tests import conftest

        if hasattr(conftest, "collect_ignore"):
            raise pytest.UsageError(
                "CI coverage audit requires an R-enabled environment"
            )
        if not session.items:
            raise pytest.UsageError("CI coverage audit collected no tests")

        selections = ci_selections()
        uncovered = []
        for item in session.items:
            marks = {mark.name for mark in item.iter_markers()}
            needs_r = item.path.name in conftest._rpy2_test_files
            for (
                has_r,
                expression,
                paths,
                ignored,
                rpy2_files,
            ) in selections.values():
                if needs_r and not has_r:
                    continue
                if rpy2_files and not needs_r:
                    continue
                if not any(path_matches(item.nodeid, path) for path in paths):
                    continue
                if any(path_matches(item.nodeid, path) for path in ignored):
                    continue
                if expression is None or expression.evaluate(marks.__contains__):
                    break
            else:
                uncovered.append(item.nodeid)
        if uncovered:
            raise pytest.UsageError(
                f"{len(uncovered)} tests are not selected by any CI task:\n"
                + "\n".join(uncovered)
            )
        print(f"All {len(session.items)} test items are selected by a CI task.")


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            ["tests", "--collect-only", "-qq", "--no-cov"], plugins=[CoverageAudit()]
        )
    )
