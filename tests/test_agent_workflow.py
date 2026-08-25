import json
from pathlib import Path

import pytest

from scripts.agent.change_scope import (
    ChangeScope,
    classify_changes,
    classify_path,
    content_risk_flags,
    risk_flags,
)
from scripts.agent.verify import (
    Check,
    VerificationError,
    expand_command,
    load_checks,
    parse_deferred,
    run_verification,
    verification_exit_code,
    write_json_report,
)


def test_classify_path_maps_estimation_api_to_numerical_risk_domain():
    assert classify_path("pyfixest/estimation/api/feols.py") == {
        "api",
        "numerical",
        "python",
    }


def test_classify_path_maps_special_test_domains():
    assert classify_path("tests/test_hac_vs_fixest.py") == {
        "hac",
        "python",
        "reference",
        "tests",
    }


def test_classify_path_falls_back_conservatively():
    assert classify_path("unexpected/new-tool.conf") == {"unknown"}


def test_classify_path_recognizes_root_agent_policy():
    assert classify_path("AGENTS.md") == {"agent"}


def test_risk_flags_cover_public_core_and_reference_changes():
    assert risk_flags("pyfixest/estimation/api/feols.py") == {"public-api"}
    assert risk_flags("pyfixest/estimation/formula/parse.py") == {
        "formula-semantics",
        "shared-estimation-core",
    }
    assert risk_flags("tests/data/reference.csv") == {"stored-reference"}


def test_content_risk_flags_cover_signatures_weights_and_estimation_flow():
    changed_content = """
def public_estimator(data, weights=None):
    config = EstimationConfig()
    return get_fit(config, vcov=\"iid\")
"""

    assert content_risk_flags("pyfixest/estimation/add_on.py", changed_content) == {
        "estimation-flow",
        "inference",
        "public-signature",
        "weights",
    }
    assert content_risk_flags("docs/developer/testing.md", changed_content) == set()


def test_classify_changes_is_sorted_and_deduplicated():
    scope = classify_changes(
        base="parent",
        merge_base="abc123",
        files=[
            "tests/test_hac_vs_fixest.py",
            "pyfixest/estimation/api/feols.py",
            "tests/test_hac_vs_fixest.py",
        ],
    )

    assert scope.files == (
        "pyfixest/estimation/api/feols.py",
        "tests/test_hac_vs_fixest.py",
    )
    assert scope.domains == ("api", "hac", "numerical", "python", "reference", "tests")
    assert scope.risks == ("public-api",)


def test_verification_matrix_is_valid_and_has_unique_ids():
    matrix = Path("scripts/agent/verification_matrix.toml")
    checks = load_checks(matrix)

    assert checks
    assert len({check.id for check in checks}) == len(checks)
    assert all(command.command[0] == "pixi" for command in checks)
    assert {"targeted-tests", "docs-render", "rust-kernel-tests"}.issubset(
        {check.id for check in checks}
    )


def test_verification_matrix_rejects_non_boolean_policy(tmp_path):
    matrix = tmp_path / "matrix.toml"
    matrix.write_text(
        """
[[check]]
id = "invalid"
description = "Invalid"
tier = "edit"
runtime = "seconds"
domains = ["always"]
command = ["pixi", "run", "invalid"]
required_local = "false"
ci_allowed = true
"""
    )

    with pytest.raises(VerificationError, match="required_local"):
        load_checks(matrix)


def test_expand_command_passes_changed_files_as_separate_arguments():
    check = Check(
        id="format",
        description="Format",
        tier="edit",
        runtime="seconds",
        domains=("python",),
        command=("pixi", "run", "tool", "{files}"),
        required_local=True,
        ci_allowed=False,
        file_scope="python",
    )

    command = expand_command(check, ("one.py", "notes.md", "two.pyi"))

    assert command == ("pixi", "run", "tool", "one.py", "two.pyi")


def test_expand_command_selects_only_test_modules():
    check = Check(
        id="targeted",
        description="Targeted",
        tier="edit",
        runtime="seconds",
        domains=("tests",),
        command=("pixi", "run", "pytest", "{files}"),
        required_local=True,
        ci_allowed=False,
        file_scope="test-python",
    )

    command = expand_command(
        check,
        ("tests/conftest.py", "tests/test_api.py", "pyfixest/api.py"),
    )

    assert command == ("pixi", "run", "pytest", "tests/test_api.py")


def test_run_verification_selects_tier_and_records_failures():
    checks = (
        Check(
            id="targeted",
            description="Targeted",
            tier="edit",
            runtime="seconds",
            domains=("agent",),
            command=("pixi", "run", "targeted"),
            required_local=True,
            ci_allowed=False,
        ),
        Check(
            id="baseline",
            description="Baseline",
            tier="pr",
            runtime="minutes",
            domains=("agent",),
            command=("pixi", "run", "baseline"),
            required_local=True,
            ci_allowed=False,
        ),
    )
    scope = ChangeScope(
        base="parent",
        merge_base="abc",
        files=("scripts/agent/verify.py",),
        domains=("agent",),
        risks=(),
    )

    results = run_verification(
        scope=scope,
        checks=checks,
        requested_tier="edit",
        deferred={},
        dry_run=False,
        cwd=Path("."),
        executor=lambda command, cwd: 1,
    )

    assert [result.status for result in results] == ["failed", "not_run"]
    assert verification_exit_code(results, dry_run=False) == 1


def test_ci_eligible_deferral_is_reported_without_failing():
    check = Check(
        id="long",
        description="Long",
        tier="domain",
        runtime="tens-of-minutes",
        domains=("numerical",),
        command=("pixi", "run", "long"),
        required_local=False,
        ci_allowed=True,
    )
    scope = ChangeScope(
        base="parent",
        merge_base="abc",
        files=("pyfixest/estimation/models/feols_.py",),
        domains=("numerical",),
        risks=("shared-estimation-core",),
    )

    results = run_verification(
        scope=scope,
        checks=(check,),
        requested_tier="domain",
        deferred={"long": "runs in CI"},
        dry_run=False,
        cwd=Path("."),
    )

    assert results[0].status == "deferred"
    assert results[0].reason == "runs in CI"
    assert verification_exit_code(results, dry_run=False) == 0


def test_local_required_check_cannot_be_deferred_successfully():
    check = Check(
        id="required",
        description="Required",
        tier="edit",
        runtime="seconds",
        domains=("agent",),
        command=("pixi", "run", "required"),
        required_local=True,
        ci_allowed=False,
    )
    scope = ChangeScope(
        base="parent",
        merge_base="abc",
        files=("scripts/agent/verify.py",),
        domains=("agent",),
        risks=(),
    )

    results = run_verification(
        scope=scope,
        checks=(check,),
        requested_tier="edit",
        deferred={"required": "not available"},
        dry_run=False,
        cwd=Path("."),
    )

    assert verification_exit_code(results, dry_run=False) == 1


def test_json_report_is_versioned(tmp_path):
    scope = ChangeScope(
        base="parent",
        merge_base="abc",
        files=(),
        domains=(),
        risks=(),
    )

    output = tmp_path / "report.json"
    write_json_report(
        path=output,
        scope=scope,
        requested_tier="pr",
        dry_run=True,
        results=(),
    )

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["scope"]["merge_base"] == "abc"


@pytest.mark.parametrize("value", ["missing-reason", "=reason", "check="])
def test_parse_deferred_rejects_incomplete_values(value):
    with pytest.raises(VerificationError):
        parse_deferred([value])
