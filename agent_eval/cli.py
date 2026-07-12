"""Prepare and score provider-neutral PyFixest agent evaluations.

The utility creates inert prompts and JSON templates. It never launches an
agent or imports a provider SDK; a maintainer must run and score trials
explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised outside the tooling environment
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = ROOT / "tasks.yaml"
DEFAULT_RUBRIC = ROOT / "rubric.yaml"
DEFAULT_PROMPT = ROOT / "prompt-template.txt"


class EvaluationConfigError(ValueError):
    """Indicate invalid evaluation configuration or results."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise EvaluationConfigError(
            "PyYAML is required only for agent-evaluation tooling. "
            "Run this command through the repository's Pixi environment."
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationConfigError(f"Expected a top-level mapping in {path}")
    return value


def _require(mapping: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - mapping.keys())
    if missing:
        raise EvaluationConfigError(f"{label} is missing fields: {missing}")


def load_and_validate_suite(
    tasks_path: Path = DEFAULT_TASKS, rubric_path: Path = DEFAULT_RUBRIC
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the frozen tasks and scoring rubric."""
    config, rubric = _load_yaml(tasks_path), _load_yaml(rubric_path)
    _require(
        config, {"schema_version", "suite", "defaults", "profiles", "tasks"}, "Tasks"
    )
    _require(
        rubric,
        {
            "schema_version",
            "suite",
            "scoring",
            "hard_failure_codes",
            "run_metadata",
            "aggregate_acceptance",
        },
        "Rubric",
    )
    if (config["schema_version"], config["suite"]) != (
        rubric["schema_version"],
        rubric["suite"],
    ):
        raise EvaluationConfigError("Task and rubric suite identities differ")
    if config["defaults"].get("repeats") != 3:
        raise EvaluationConfigError("The frozen suite must request three repeats")
    profiles, tasks = config["profiles"], config["tasks"]
    if not isinstance(profiles, dict) or not isinstance(tasks, list) or not tasks:
        raise EvaluationConfigError("Profiles and a non-empty task list are required")
    required_task = {
        "id",
        "profile",
        "topic",
        "prompt",
        "expected_answer",
        "preferred_first_sources",
        "relevant_queries",
    }
    ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise EvaluationConfigError("Every task must be a mapping")
        _require(task, required_task, "Task")
        if task["id"] in ids or task["profile"] not in profiles:
            raise EvaluationConfigError(
                f"Duplicate task or unknown profile: {task['id']}"
            )
        ids.add(task["id"])
        if any(
            not isinstance(task[field], list) or not task[field]
            for field in required_task - {"id", "profile", "topic", "prompt"}
        ):
            raise EvaluationConfigError(
                f"Task {task['id']} has an empty answer/source list"
            )
    scoring = rubric["scoring"]
    dimensions = scoring.get("dimensions", {})
    max_points = scoring.get("max_points")
    dimension_points = sum(item.get("max_points", -1) for item in dimensions.values())
    if max_points != 10 or dimension_points != max_points:
        raise EvaluationConfigError("Scoring dimensions must sum to 10")
    thresholds = rubric["aggregate_acceptance"].get("thresholds", {})
    if thresholds.get("repeats_per_task") != 3:
        raise EvaluationConfigError("Aggregate thresholds must require three repeats")
    return config, rubric


def suite_fingerprint(tasks_path: Path, rubric_path: Path, prompt_path: Path) -> str:
    """Hash the exact task, rubric, and prompt inputs used for a run."""
    digest = hashlib.sha256()
    for path in (tasks_path, rubric_path, prompt_path):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare_run(
    *,
    output: Path,
    tasks_path: Path,
    rubric_path: Path,
    prompt_path: Path,
    profile: str,
    date: str,
    pyfixest_ref: str,
    pyfixest_version: str,
    source_access_mode: str,
    install_source: str,
    agent_provider: str,
    agent_harness: str,
    model: str,
    system_prompt_id: str,
    tools: list[str],
    platform: str,
) -> dict[str, Any]:
    """Write prompts, blank records, and a frozen manifest without running agents."""
    config, rubric = load_and_validate_suite(tasks_path, rubric_path)
    if profile != "all" and profile not in config["profiles"]:
        raise EvaluationConfigError(f"Unknown profile: {profile}")
    if output.exists() and any(output.iterdir()):
        raise EvaluationConfigError(f"Output directory is not empty: {output}")
    template = prompt_path.read_text(encoding="utf-8")
    if template.count("{profile}") != 1 or template.count("{task_prompt}") != 1:
        raise EvaluationConfigError("Prompt template placeholders are invalid")
    tasks = [
        task
        for task in config["tasks"]
        if profile == "all" or task["profile"] == profile
    ]
    fingerprint = suite_fingerprint(tasks_path, rubric_path, prompt_path)
    context = {
        "source_access_mode": source_access_mode,
        "install_source": install_source,
        "network": "disabled",
        "agent_provider": agent_provider,
        "agent_harness": agent_harness,
        "model": model,
        "system_prompt_id": system_prompt_id,
        "tools": tools,
        "platform": platform,
    }
    expected: list[dict[str, Any]] = []
    for task in tasks:
        for run_number in range(1, 4):
            stem = f"{task['id']}-run-{run_number}"
            prompt = template.format(
                profile=task["profile"], task_prompt=task["prompt"]
            )
            prompt_file = output / "prompts" / f"{stem}.txt"
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(prompt, encoding="utf-8")
            record = {
                "suite_version": config["schema_version"],
                "suite_fingerprint": fingerprint,
                "date": date,
                "profile": task["profile"],
                "task_id": task["id"],
                "run_number": run_number,
                "pyfixest_ref": pyfixest_ref,
                "pyfixest_version": pyfixest_version,
                **context,
                "elapsed_seconds": None,
                "commands_or_searches_used": [],
                "files_consulted": [],
                "first_relevant_source": None,
                "final_answer": None,
                "score": {name: None for name in rubric["scoring"]["dimensions"]}
                | {"total": None},
                "hard_failure_code": None,
                "hard_failure": None,
                "notes": None,
            }
            record_file = output / "records" / f"{stem}.json"
            _write_json(record_file, record)
            expected.append(
                {
                    "task_id": task["id"],
                    "run_number": run_number,
                    "prompt": prompt_file.relative_to(output).as_posix(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "record": record_file.relative_to(output).as_posix(),
                }
            )
    manifest = {
        "manifest_schema_version": 1,
        "suite": config["suite"],
        "suite_version": config["schema_version"],
        "suite_fingerprint": fingerprint,
        "profile_selection": profile,
        "task_ids": [task["id"] for task in tasks],
        "repeats_per_task": 3,
        "date": date,
        "pyfixest_ref": pyfixest_ref,
        "pyfixest_version": pyfixest_version,
        "comparison_context": context,
        "expected_runs": expected,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    if not files:
        raise EvaluationConfigError(f"No JSON records found in {path}")
    records: list[dict[str, Any]] = []
    for file in files:
        value = json.loads(file.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
        if not all(isinstance(item, dict) for item in values):
            raise EvaluationConfigError(f"Expected JSON object records in {file}")
        records.extend(values)
    return records


def _validate_record(
    record: dict[str, Any],
    *,
    task: dict[str, Any],
    rubric: dict[str, Any],
    require_complete: bool,
) -> None:
    _require(record, set(rubric["run_metadata"]["required_fields"]), "Result")
    if (record["task_id"], record["profile"]) != (task["id"], task["profile"]):
        raise EvaluationConfigError(f"Result identity differs from task {task['id']}")
    if record["suite_version"] != rubric["schema_version"] or record[
        "run_number"
    ] not in {1, 2, 3}:
        raise EvaluationConfigError(
            f"Result {task['id']} has invalid suite/run metadata"
        )
    if record["network"] != "disabled":
        raise EvaluationConfigError(
            f"Result {task['id']} did not disable network access"
        )
    for field in ("commands_or_searches_used", "files_consulted", "tools"):
        if not isinstance(record[field], list) or not all(
            isinstance(value, str) for value in record[field]
        ):
            raise EvaluationConfigError(
                f"Result field {field} must be a list of strings"
            )
    score, dimensions = record["score"], rubric["scoring"]["dimensions"]
    _require(score, {*dimensions, "total"}, "Score")
    values = [score[name] for name in dimensions]
    if not require_complete and all(
        value is None for value in [*values, score["total"]]
    ):
        return
    elapsed, answer = record["elapsed_seconds"], record["final_answer"]
    if (
        not isinstance(elapsed, int | float)
        or isinstance(elapsed, bool)
        or elapsed < 0
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        raise EvaluationConfigError(f"Result {task['id']} is incomplete")
    for name, value in zip(dimensions, values, strict=True):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= dimensions[name]["max_points"]
        ):
            raise EvaluationConfigError(f"Score {name} is outside its allowed range")
    if score["total"] != sum(values):
        raise EvaluationConfigError(f"Result {task['id']} has an incorrect score total")
    if score["source_discovery"] > 0 and not record["first_relevant_source"]:
        raise EvaluationConfigError("Positive source discovery requires a first source")
    failure, code = record["hard_failure"], record["hard_failure_code"]
    if (failure is None) != (code is None) or code not in {
        None,
        *rubric["hard_failure_codes"],
    }:
        raise EvaluationConfigError("Hard failure text and code are inconsistent")


def _intended(source: str | None, preferred: list[str]) -> bool:
    if not isinstance(source, str):
        return False
    source = source.replace("\\", "/")
    if any(source.endswith(item) for item in preferred):
        return True

    def route(path: str) -> str:
        for marker in ("pyfixest/docs/pages/", "docs/"):
            if marker in path:
                path = path.split(marker, 1)[1]
        return path.removesuffix(".qmd").removesuffix(".md")

    return route(source) in {route(item) for item in preferred}


def aggregate_results(
    *,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    tasks_config: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Validate exactly three records per task and calculate aggregate metrics."""
    _require(
        manifest,
        {
            "suite",
            "suite_version",
            "suite_fingerprint",
            "task_ids",
            "repeats_per_task",
            "date",
            "pyfixest_ref",
            "pyfixest_version",
            "comparison_context",
            "expected_runs",
        },
        "Manifest",
    )
    if manifest["repeats_per_task"] != 3 or manifest["suite"] != tasks_config["suite"]:
        raise EvaluationConfigError(
            "Manifest must describe this suite with exactly three repeats"
        )
    task_lookup = {task["id"]: task for task in tasks_config["tasks"]}
    task_ids = manifest["task_ids"]
    expected = {(task_id, run) for task_id in task_ids for run in range(1, 4)}
    declared = {
        (item["task_id"], item["run_number"]) for item in manifest["expected_runs"]
    }
    if (
        len(task_ids) != len(set(task_ids))
        or declared != expected
        or not set(task_ids) <= task_lookup.keys()
    ):
        raise EvaluationConfigError(
            "Manifest does not declare three unique runs for known tasks"
        )
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record.get("task_id"), record.get("run_number"))
        if key not in expected:
            raise EvaluationConfigError(f"Unexpected result record: {key}")
        if key in observed:
            raise EvaluationConfigError(f"Duplicate result record: {key}")
        _validate_record(
            record, task=task_lookup[key[0]], rubric=rubric, require_complete=True
        )
        invariants = {
            "suite_fingerprint": manifest["suite_fingerprint"],
            "date": manifest["date"],
            "pyfixest_ref": manifest["pyfixest_ref"],
            "pyfixest_version": manifest["pyfixest_version"],
            **manifest["comparison_context"],
        }
        if any(record.get(field) != value for field, value in invariants.items()):
            raise EvaluationConfigError(f"Result {key} differs from its manifest")
        observed[key] = record
    if observed.keys() != expected:
        raise EvaluationConfigError(
            f"Missing {len(expected - observed.keys())} result records"
        )
    task_scores = {
        task_id: [observed[task_id, run]["score"]["total"] for run in range(1, 4)]
        for task_id in task_ids
    }
    profile_scores: dict[str, list[int]] = {}
    failures: dict[str, int] = {}
    intended = 0
    for (task_id, _), record in observed.items():
        profile_scores.setdefault(record["profile"], []).append(
            record["score"]["total"]
        )
        code = record["hard_failure_code"]
        if code:
            failures[code] = failures.get(code, 0) + 1
        intended += _intended(
            record["first_relevant_source"],
            task_lookup[task_id]["preferred_first_sources"],
        )
    task_medians = {
        task: statistics.median(scores) for task, scores in task_scores.items()
    }
    thresholds = rubric["aggregate_acceptance"]["thresholds"]
    rate = intended / len(observed)
    aggregate = {
        "task_median_scores": task_medians,
        "profile_median_scores": {
            profile: statistics.median(scores)
            for profile, scores in profile_scores.items()
        },
        "aggregate_median_score": statistics.median(
            record["score"]["total"] for record in observed.values()
        ),
        "intended_first_source_rate": rate,
        "hard_failure_count_by_code": failures,
    }
    return {
        "summary_schema_version": 1,
        "status": "complete",
        "suite": manifest["suite"],
        "suite_version": manifest["suite_version"],
        "suite_fingerprint": manifest["suite_fingerprint"],
        "profile_selection": manifest.get("profile_selection", "all"),
        "task_ids": task_ids,
        "repeats_per_task": 3,
        "completed_run_count": len(observed),
        "pyfixest_ref": manifest["pyfixest_ref"],
        "pyfixest_version": manifest["pyfixest_version"],
        "comparison_context": manifest["comparison_context"],
        "aggregate": aggregate,
        "validation": {
            "all_expected_runs_present": True,
            "every_task_median_at_least_minimum": all(
                value >= thresholds["minimum_task_median_score"]
                for value in task_medians.values()
            ),
            "no_unsupported_api_failures": failures.get("unsupported_api", 0)
            <= thresholds["maximum_unsupported_api_failures"],
            "intended_first_source_rate_at_least_minimum": rate
            >= thresholds["minimum_intended_first_source_rate"],
        },
    }


def compare_summaries(
    baseline: dict[str, Any], candidate: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, Any]:
    """Compare complete baseline and candidate summaries under frozen settings."""
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise EvaluationConfigError("Both summaries must be complete")
    frozen = {
        "suite",
        "suite_version",
        "suite_fingerprint",
        "profile_selection",
        "task_ids",
        "repeats_per_task",
        "comparison_context",
    }
    differing = [
        field for field in frozen if baseline.get(field) != candidate.get(field)
    ]
    if differing:
        raise EvaluationConfigError(f"Baseline and candidate differ on {differing[0]}")
    thresholds = rubric["aggregate_acceptance"]["thresholds"]
    base, cand = baseline["aggregate"], candidate["aggregate"]
    checks = {
        "all_runs_complete": candidate["completed_run_count"]
        == len(candidate["task_ids"]) * 3,
        "every_task_median_at_least_minimum": all(
            value >= thresholds["minimum_task_median_score"]
            for value in cand["task_median_scores"].values()
        ),
        "no_unsupported_api_failures": cand["hard_failure_count_by_code"].get(
            "unsupported_api", 0
        )
        <= thresholds["maximum_unsupported_api_failures"],
        "intended_first_source_rate_at_least_minimum": cand[
            "intended_first_source_rate"
        ]
        >= thresholds["minimum_intended_first_source_rate"],
        "aggregate_did_not_regress": cand["aggregate_median_score"]
        >= base["aggregate_median_score"],
    }
    return {
        "comparison_schema_version": 1,
        "passed": all(checks.values()),
        "baseline_ref": baseline["pyfixest_ref"],
        "candidate_ref": candidate["pyfixest_ref"],
        "checks": checks,
        "deltas_from_baseline": {
            "task_median_scores": {
                task: cand["task_median_scores"][task]
                - base["task_median_scores"][task]
                for task in candidate["task_ids"]
            },
            "profile_median_scores": {
                profile: value - base["profile_median_scores"][profile]
                for profile, value in cand["profile_median_scores"].items()
            },
            "aggregate_median_score": cand["aggregate_median_score"]
            - base["aggregate_median_score"],
            "intended_first_source_rate": cand["intended_first_source_rate"]
            - base["intended_first_source_rate"],
        },
    }


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def config(command: argparse.ArgumentParser) -> None:
        command.add_argument("--tasks", type=_path, default=DEFAULT_TASKS)
        command.add_argument("--rubric", type=_path, default=DEFAULT_RUBRIC)

    validate = commands.add_parser("validate-suite")
    config(validate)
    prepare = commands.add_parser("prepare")
    config(prepare)
    prepare.add_argument("--prompt-template", type=_path, default=DEFAULT_PROMPT)
    for name in (
        "output",
        "date",
        "pyfixest-ref",
        "pyfixest-version",
        "source-access-mode",
        "install-source",
    ):
        prepare.add_argument(
            f"--{name}", required=True, type=_path if name == "output" else str
        )
    prepare.add_argument("--profile", default="all")
    for name in (
        "agent-provider",
        "agent-harness",
        "model",
        "system-prompt-id",
        "platform",
    ):
        prepare.add_argument(f"--{name}", default="unknown")
    prepare.add_argument("--tool", action="append", default=[])
    validate_results = commands.add_parser("validate-results")
    config(validate_results)
    validate_results.add_argument("--results", type=_path, required=True)
    validate_results.add_argument("--allow-incomplete", action="store_true")
    aggregate = commands.add_parser("aggregate")
    config(aggregate)
    for name in ("manifest", "results", "output"):
        aggregate.add_argument(f"--{name}", type=_path, required=True)
    compare = commands.add_parser("compare")
    config(compare)
    for name in ("baseline", "candidate"):
        compare.add_argument(f"--{name}", type=_path, required=True)
    compare.add_argument("--output", type=_path)
    return parser


def _validate_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    rubric: dict[str, Any],
    *,
    complete: bool,
) -> None:
    tasks = {task["id"]: task for task in config["tasks"]}
    for record in records:
        if record.get("task_id") not in tasks:
            raise EvaluationConfigError(f"Unknown task: {record.get('task_id')}")
        _validate_record(
            record,
            task=tasks[record["task_id"]],
            rubric=rubric,
            require_complete=complete,
        )


def main(argv: list[str] | None = None) -> int:
    """Run deterministic preparation, validation, aggregation, or comparison."""
    args = build_parser().parse_args(argv)
    try:
        config, rubric = load_and_validate_suite(args.tasks, args.rubric)
        exit_code = 0
        if args.command == "validate-suite":
            print(f"valid: {len(config['tasks'])} tasks")
        elif args.command == "prepare":
            manifest = prepare_run(
                output=args.output,
                tasks_path=args.tasks,
                rubric_path=args.rubric,
                prompt_path=args.prompt_template,
                profile=args.profile,
                date=args.date,
                pyfixest_ref=args.pyfixest_ref,
                pyfixest_version=args.pyfixest_version,
                source_access_mode=args.source_access_mode,
                install_source=args.install_source,
                agent_provider=args.agent_provider,
                agent_harness=args.agent_harness,
                model=args.model,
                system_prompt_id=args.system_prompt_id,
                tools=args.tool,
                platform=args.platform,
            )
            print(f"prepared: {len(manifest['expected_runs'])} inert trials")
        elif args.command == "validate-results":
            records = _load_json_records(args.results)
            _validate_records(
                records, config, rubric, complete=not args.allow_incomplete
            )
            print(f"valid: {len(records)} result records")
        elif args.command == "aggregate":
            summary = aggregate_results(
                manifest=json.loads(args.manifest.read_text(encoding="utf-8")),
                records=_load_json_records(args.results),
                tasks_config=config,
                rubric=rubric,
            )
            _write_json(args.output, summary)
            print(f"aggregated: {summary['completed_run_count']} complete trials")
        else:
            comparison = compare_summaries(
                json.loads(args.baseline.read_text(encoding="utf-8")),
                json.loads(args.candidate.read_text(encoding="utf-8")),
                rubric,
            )
            if args.output:
                _write_json(args.output, comparison)
            else:
                print(json.dumps(comparison, indent=2, sort_keys=True))
            exit_code = 0 if comparison["passed"] else 1
    except (EvaluationConfigError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    else:
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
