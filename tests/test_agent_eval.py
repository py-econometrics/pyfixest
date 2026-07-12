from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_eval.cli import (
    DEFAULT_PROMPT,
    DEFAULT_RUBRIC,
    DEFAULT_TASKS,
    EvaluationConfigError,
    _load_json_records,
    _validate_record,
    aggregate_results,
    compare_summaries,
    load_and_validate_suite,
    prepare_run,
)


def _prepare(tmp_path: Path) -> tuple[dict, dict, dict, Path]:
    output = tmp_path / "prepared"
    manifest = prepare_run(
        output=output,
        tasks_path=DEFAULT_TASKS,
        rubric_path=DEFAULT_RUBRIC,
        prompt_path=DEFAULT_PROMPT,
        profile="all",
        date="2026-07-11",
        pyfixest_ref="a" * 40,
        pyfixest_version="0.0.test",
        source_access_mode="source_checkout",
        install_source="source",
        agent_provider="test-provider",
        agent_harness="test-harness",
        model="test-model",
        system_prompt_id="test-prompt",
        tools=["shell", "read"],
        platform="test-platform",
    )
    tasks_config, rubric = load_and_validate_suite()
    return manifest, tasks_config, rubric, output


def _complete_records(
    output: Path, tasks_config: dict, *, total: int = 8
) -> list[dict]:
    tasks = {task["id"]: task for task in tasks_config["tasks"]}
    records = _load_json_records(output / "records")
    for record in records:
        record["elapsed_seconds"] = 1.25
        record["commands_or_searches_used"] = ["rg relevant term"]
        record["files_consulted"] = [
            tasks[record["task_id"]]["preferred_first_sources"][0]
        ]
        record["first_relevant_source"] = record["files_consulted"][0]
        record["final_answer"] = "A manually reviewed answer."
        record["score"] = {
            "correctness": 4,
            "source_discovery": 2,
            "efficiency": 1,
            "evidence": 1,
            "epistemic_discipline": total - 8,
            "total": total,
        }
    return records


def test_suite_configuration_is_valid_and_frozen_to_three_repeats() -> None:
    tasks_config, rubric = load_and_validate_suite()

    assert len(tasks_config["tasks"]) == 15
    assert tasks_config["defaults"]["repeats"] == 3
    assert rubric["aggregate_acceptance"]["thresholds"]["repeats_per_task"] == 3


def test_prepare_creates_only_inert_prompts_records_and_manifest(
    tmp_path: Path,
) -> None:
    manifest, _, _, output = _prepare(tmp_path)

    assert len(manifest["expected_runs"]) == 45
    assert len(list((output / "prompts").glob("*.txt"))) == 45
    assert len(list((output / "records").glob("*.json"))) == 45
    assert (output / "manifest.json").is_file()
    assert "provider" not in {path.name for path in output.iterdir()}

    record = json.loads(next((output / "records").glob("*.json")).read_text())
    assert record["final_answer"] is None
    assert record["score"]["total"] is None
    prompt = next((output / "prompts").glob("*.txt")).read_text()
    assert "Do not use the network" in prompt
    assert "Do not modify files" in prompt


def test_aggregate_requires_exactly_one_complete_record_per_trial(
    tmp_path: Path,
) -> None:
    manifest, tasks_config, rubric, output = _prepare(tmp_path)
    records = _complete_records(output, tasks_config)

    summary = aggregate_results(
        manifest=manifest,
        records=records,
        tasks_config=tasks_config,
        rubric=rubric,
    )

    assert summary["completed_run_count"] == 45
    assert summary["aggregate"]["aggregate_median_score"] == 8
    assert summary["aggregate"]["intended_first_source_rate"] == 1
    assert all(summary["validation"].values())

    with pytest.raises(EvaluationConfigError, match="Duplicate result record"):
        aggregate_results(
            manifest=manifest,
            records=[*records, records[0]],
            tasks_config=tasks_config,
            rubric=rubric,
        )


def test_aggregate_rejects_manifest_without_three_runs_per_task(
    tmp_path: Path,
) -> None:
    manifest, tasks_config, rubric, output = _prepare(tmp_path)
    manifest["repeats_per_task"] = 2

    with pytest.raises(EvaluationConfigError, match="exactly three repeats"):
        aggregate_results(
            manifest=manifest,
            records=_complete_records(output, tasks_config),
            tasks_config=tasks_config,
            rubric=rubric,
        )


def test_result_validation_rejects_inconsistent_manual_scoring(tmp_path: Path) -> None:
    _, tasks_config, rubric, output = _prepare(tmp_path)
    record = _complete_records(output, tasks_config)[0]
    record["score"]["total"] = 10
    task = next(
        task for task in tasks_config["tasks"] if task["id"] == record["task_id"]
    )

    with pytest.raises(EvaluationConfigError, match="incorrect score total"):
        _validate_record(record, task=task, rubric=rubric, require_complete=True)


def test_compare_enforces_thresholds_and_no_aggregate_regression(
    tmp_path: Path,
) -> None:
    manifest, tasks_config, rubric, output = _prepare(tmp_path)
    records = _complete_records(output, tasks_config)
    baseline = aggregate_results(
        manifest=manifest,
        records=records,
        tasks_config=tasks_config,
        rubric=rubric,
    )
    candidate = copy.deepcopy(baseline)
    candidate["pyfixest_ref"] = "b" * 40

    passing = compare_summaries(baseline, candidate, rubric)
    assert passing["passed"] is True

    candidate["aggregate"]["aggregate_median_score"] = 7
    candidate["aggregate"]["task_median_scores"][candidate["task_ids"][0]] = 7
    failing = compare_summaries(baseline, candidate, rubric)
    assert failing["passed"] is False
    assert failing["checks"]["aggregate_did_not_regress"] is False
    assert failing["checks"]["every_task_median_at_least_minimum"] is False


def test_compare_rejects_changed_protocol(tmp_path: Path) -> None:
    manifest, tasks_config, rubric, output = _prepare(tmp_path)
    baseline = aggregate_results(
        manifest=manifest,
        records=_complete_records(output, tasks_config),
        tasks_config=tasks_config,
        rubric=rubric,
    )
    candidate = copy.deepcopy(baseline)
    candidate["comparison_context"]["model"] = "different-model"

    with pytest.raises(EvaluationConfigError, match="comparison_context"):
        compare_summaries(baseline, candidate, rubric)
