"""Run deterministic, risk-based verification for a pyfixest change."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tomllib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.agent.change_scope import ChangeScope, ScopeError, get_change_scope

Tier = Literal["edit", "pr", "domain", "exhaustive"]
Status = Literal["passed", "failed", "deferred", "not_run"]
CommandExecutor = Callable[[tuple[str, ...], Path], int]

TIER_RANK: dict[str, int] = {
    "edit": 0,
    "pr": 1,
    "domain": 2,
    "exhaustive": 3,
}
FILE_SCOPES = {
    "all",
    "package-python",
    "python",
    "test-python",
    "toml",
    "workflow",
    "yaml",
}


class VerificationError(RuntimeError):
    """Raised when verification configuration or arguments are invalid."""


@dataclass(frozen=True, slots=True)
class Check:
    """One configured verification command."""

    id: str
    description: str
    tier: Tier
    runtime: str
    domains: tuple[str, ...]
    command: tuple[str, ...]
    required_local: bool
    ci_allowed: bool
    file_scope: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Observed result for one applicable check."""

    id: str
    description: str
    tier: Tier
    runtime: str
    status: Status
    command: tuple[str, ...]
    duration_seconds: float
    reason: str
    required_local: bool
    ci_allowed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result."""
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


def _require_string(record: Mapping[str, object], key: str, check_id: str) -> str:
    """Read one required string from a TOML check record."""
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise VerificationError(f"check {check_id!r} needs a non-empty {key!r}")
    return value


def _string_tuple(
    record: Mapping[str, object],
    key: str,
    check_id: str,
) -> tuple[str, ...]:
    """Read a non-empty list of strings from a TOML check record."""
    value = record.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise VerificationError(f"check {check_id!r} needs string list {key!r}")
    return tuple(value)


def _boolean(
    record: Mapping[str, object],
    key: str,
    check_id: str,
) -> bool:
    """Read one required Boolean from a TOML check record."""
    value = record.get(key)
    if not isinstance(value, bool):
        raise VerificationError(f"check {check_id!r} needs Boolean {key!r}")
    return value


def load_checks(path: Path) -> tuple[Check, ...]:
    """Load and validate the verification matrix."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VerificationError(
            f"cannot read verification matrix {path}: {exc}"
        ) from exc

    records = payload.get("check")
    if not isinstance(records, list) or not records:
        raise VerificationError("verification matrix must contain [[check]] records")

    checks: list[Check] = []
    seen: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise VerificationError("each [[check]] entry must be a table")
        check_id = _require_string(raw_record, "id", "<unknown>")
        if check_id in seen:
            raise VerificationError(f"duplicate check id {check_id!r}")
        seen.add(check_id)

        tier = _require_string(raw_record, "tier", check_id)
        if tier not in TIER_RANK:
            raise VerificationError(f"check {check_id!r} has invalid tier {tier!r}")

        file_scope = raw_record.get("file_scope")
        if file_scope is not None and file_scope not in FILE_SCOPES:
            raise VerificationError(
                f"check {check_id!r} has invalid file_scope {file_scope!r}"
            )

        checks.append(
            Check(
                id=check_id,
                description=_require_string(raw_record, "description", check_id),
                tier=tier,
                runtime=_require_string(raw_record, "runtime", check_id),
                domains=_string_tuple(raw_record, "domains", check_id),
                command=_string_tuple(raw_record, "command", check_id),
                required_local=_boolean(raw_record, "required_local", check_id),
                ci_allowed=_boolean(raw_record, "ci_allowed", check_id),
                file_scope=file_scope,
            )
        )
    return tuple(checks)


def is_applicable(check: Check, domains: Sequence[str]) -> bool:
    """Return whether a check applies to the classified domains."""
    return "always" in check.domains or bool(set(check.domains).intersection(domains))


def files_for_scope(file_scope: str | None, files: Sequence[str]) -> tuple[str, ...]:
    """Filter changed paths for a configured command placeholder."""
    if file_scope is None:
        return ()
    if file_scope == "all":
        return tuple(files)
    if file_scope == "python":
        return tuple(path for path in files if path.endswith((".py", ".pyi")))
    if file_scope == "test-python":
        return tuple(
            path
            for path in files
            if path.startswith("tests/test_") and path.endswith(".py")
        )
    if file_scope == "package-python":
        return tuple(
            path
            for path in files
            if path.startswith("pyfixest/") and path.endswith((".py", ".pyi"))
        )
    if file_scope == "toml":
        return tuple(path for path in files if path.endswith(".toml"))
    if file_scope == "yaml":
        return tuple(path for path in files if path.endswith((".yaml", ".yml")))
    if file_scope == "workflow":
        return tuple(
            path
            for path in files
            if path.startswith(".github/workflows/")
            and path.endswith((".yaml", ".yml"))
        )
    raise VerificationError(f"unsupported file scope {file_scope!r}")


def expand_command(check: Check, files: Sequence[str]) -> tuple[str, ...]:
    """Expand the file placeholder without shell interpolation."""
    selected_files = files_for_scope(check.file_scope, files)
    expanded: list[str] = []
    placeholders = 0
    for argument in check.command:
        if argument == "{files}":
            placeholders += 1
            expanded.extend(selected_files)
        else:
            expanded.append(argument)
    if placeholders > 1:
        raise VerificationError(f"check {check.id!r} uses multiple file placeholders")
    if check.file_scope is not None and placeholders != 1:
        raise VerificationError(
            f"check {check.id!r} declares file_scope without a {{files}} placeholder"
        )
    if check.file_scope is None and placeholders:
        raise VerificationError(
            f"check {check.id!r} uses {{files}} without declaring file_scope"
        )
    return tuple(expanded)


def _execute(command: tuple[str, ...], cwd: Path) -> int:
    """Execute one configured command."""
    return subprocess.run(command, cwd=cwd, check=False).returncode


def run_verification(
    *,
    scope: ChangeScope,
    checks: Sequence[Check],
    requested_tier: Tier,
    deferred: Mapping[str, str],
    dry_run: bool,
    cwd: Path,
    executor: CommandExecutor = _execute,
) -> tuple[CheckResult, ...]:
    """Run or classify every applicable check."""
    known_ids = {check.id for check in checks}
    unknown_deferred = set(deferred).difference(known_ids)
    if unknown_deferred:
        unknown = ", ".join(sorted(unknown_deferred))
        raise VerificationError(f"unknown deferred check(s): {unknown}")

    results: list[CheckResult] = []
    for check in checks:
        if not is_applicable(check, scope.domains):
            continue

        command = expand_command(check, scope.files)
        reason = ""
        duration = 0.0
        if TIER_RANK[check.tier] > TIER_RANK[requested_tier]:
            status: Status = "not_run"
            reason = f"above requested {requested_tier!r} tier"
        elif check.file_scope is not None and not files_for_scope(
            check.file_scope, scope.files
        ):
            status = "not_run"
            reason = f"no changed files in {check.file_scope!r} scope"
        elif check.id in deferred:
            status = "deferred"
            reason = deferred[check.id]
        elif dry_run:
            status = "not_run"
            reason = "dry run"
        else:
            started = time.monotonic()
            returncode = executor(command, cwd)
            duration = time.monotonic() - started
            status = "passed" if returncode == 0 else "failed"
            if returncode:
                reason = f"command exited with status {returncode}"

        results.append(
            CheckResult(
                id=check.id,
                description=check.description,
                tier=check.tier,
                runtime=check.runtime,
                status=status,
                command=command,
                duration_seconds=round(duration, 3),
                reason=reason,
                required_local=check.required_local,
                ci_allowed=check.ci_allowed,
            )
        )
    return tuple(results)


def verification_exit_code(
    results: Sequence[CheckResult],
    *,
    dry_run: bool,
) -> int:
    """Return the process exit code implied by verification results."""
    if dry_run:
        return 0
    if any(result.status == "failed" for result in results):
        return 1
    if any(
        result.status == "deferred" and (result.required_local or not result.ci_allowed)
        for result in results
    ):
        return 1
    return 0


def _result_label(status: Status) -> str:
    """Return a compact terminal label for a result status."""
    return {
        "passed": "PASS",
        "failed": "FAIL",
        "deferred": "DEFER",
        "not_run": "NOT RUN",
    }[status]


def format_report(
    *,
    scope: ChangeScope,
    requested_tier: Tier,
    results: Sequence[CheckResult],
) -> str:
    """Format a verification report for terminal review."""
    lines = [
        f"Verification base: {scope.base} ({scope.merge_base})",
        f"Domains: {', '.join(scope.domains) or 'none'}",
        f"Risks: {', '.join(scope.risks) or 'none'}",
        f"Requested tier: {requested_tier}",
    ]
    for result in results:
        command = shlex.join(result.command) if result.command else "<none>"
        suffix = f" — {result.reason}" if result.reason else ""
        lines.append(
            f"[{_result_label(result.status)}] {result.id} "
            f"({result.duration_seconds:.3f}s): {command}{suffix}"
        )
    return "\n".join(lines)


def write_json_report(
    *,
    path: Path,
    scope: ChangeScope,
    requested_tier: Tier,
    dry_run: bool,
    results: Sequence[CheckResult],
) -> None:
    """Write a versioned verification report to an explicit path."""
    if not path.parent.exists():
        raise VerificationError(f"JSON output directory does not exist: {path.parent}")
    payload = {
        "schema_version": 1,
        "requested_tier": requested_tier,
        "dry_run": dry_run,
        "scope": scope.to_dict(),
        "results": [result.to_dict() for result in results],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_deferred(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated CHECK=REASON deferral arguments."""
    deferred: dict[str, str] = {}
    for value in values:
        check_id, separator, reason = value.partition("=")
        if not separator or not check_id or not reason:
            raise VerificationError(
                "--defer values must use CHECK_ID=REASON with a non-empty reason"
            )
        if check_id in deferred:
            raise VerificationError(f"duplicate deferral for {check_id!r}")
        deferred[check_id] = reason
    return deferred


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/master")
    parser.add_argument(
        "--tier",
        choices=tuple(TIER_RANK),
        default="pr",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the versioned report to this explicit path.",
    )
    parser.add_argument(
        "--defer",
        action="append",
        default=[],
        metavar="CHECK_ID=REASON",
        help="Defer one CI-eligible check with an explicit reason.",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("verification_matrix.toml"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic verification CLI."""
    args = _parse_args(argv)
    try:
        checks = load_checks(args.matrix)
        scope = get_change_scope(base=args.base, cwd=Path.cwd())
        deferred = parse_deferred(args.defer)
        results = run_verification(
            scope=scope,
            checks=checks,
            requested_tier=args.tier,
            deferred=deferred,
            dry_run=args.dry_run,
            cwd=Path.cwd(),
        )
        print(
            format_report(
                scope=scope,
                requested_tier=args.tier,
                results=results,
            )
        )
        if args.json_output is not None:
            write_json_report(
                path=args.json_output,
                scope=scope,
                requested_tier=args.tier,
                dry_run=args.dry_run,
                results=results,
            )
    except (ScopeError, VerificationError) as exc:
        print(f"error: {exc}")
        return 2
    return verification_exit_code(results, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
