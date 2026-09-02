"""Record and compare a pinned pyfixest release baseline.

`tests/test_release_contract.py` runs twice: once inside the locked workspace in
`tests/snapshots/release/` against the pinned release wheel, where every
`baseline.check(...)` call records its value, and once against the current
checkout, where the same calls compare. Because both runs execute the same test
file, the two sides cannot describe different case matrices.

The recorded cache is platform-local and gitignored; see
`docs/developer/testing.md`.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests._feols_test_cases import fixed_effect_interactions_to_legacy

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "tests" / "snapshots" / "release" / ".cache" / "contract.json.gz"
RELEASE_VERSION = "0.60.0"
RECORD_OPTION = "--record-release-baseline"
RECORD_COMMAND = (
    "pixi run --locked --clean-env "
    "--manifest-path tests/snapshots/release/pixi.toml record"
)

# Both sides run the same algorithm on the same inputs, so only the two
# environments' interpreter, NumPy and BLAS builds separate them. The measured
# drift across the whole matrix stays below 1e-7 wherever behaviour did not
# change, so this bound is roughly ten times the observed noise. Widen a single
# call with an explicit ``reason`` rather than relaxing it here.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-10

# Anything that changes a recorded value has to invalidate the cache, which is
# also what keeps pytest node ids usable as cache keys.
FINGERPRINT_SOURCES = (
    "tests/test_release_contract.py",
    "tests/_release_baseline.py",
    "tests/_feols_test_cases.py",
    "tests/snapshots/release/pixi.lock",
)


def fingerprint() -> str:
    """Hash the baseline inputs together with the current platform."""
    digest = hashlib.sha256(
        f"{platform.system().lower()}:{platform.machine().lower()}".encode()
    )
    for relative_path in FINGERPRINT_SOURCES:
        digest.update(relative_path.encode())
        digest.update((ROOT / relative_path).read_bytes())
    return digest.hexdigest()[:16]


def data_digest(data: pd.DataFrame) -> str:
    """Digest a data set independently of the pandas version that built it."""
    digest = hashlib.sha256()
    for column in sorted(data.columns):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(np.float64)
        digest.update(str(column).encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()[:16]


def _label(value: Any) -> str:
    if isinstance(value, tuple):
        return ":".join(str(part) for part in value)
    return str(value)


def _flatten(name: str, value: Any) -> dict[str, Any]:
    """Render one quantity as flat ``name|label`` entries."""
    if isinstance(value, pd.DataFrame):
        value = value.stack()
    if isinstance(value, pd.Series):
        return {
            f"{name}|{_label(label)}": item
            for label, item in value.sort_index().items()
        }
    if np.ndim(value) == 0:
        return {name: value}
    return {
        f"{name}|{index}": item for index, item in enumerate(np.asarray(value).ravel())
    }


def _as_float(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class Baseline:
    """Record or compare one case's results against the pinned release."""

    def __init__(self, recorded: dict[str, Any], *, recording: bool, node_id: str):
        self._recorded = recorded
        self._recording = recording
        self._node_id = node_id

    def fml(self, fml: str) -> str:
        """Spell a formula the way the pyfixest doing the fitting understands it."""
        return fixed_effect_interactions_to_legacy(fml) if self._recording else fml

    def check(
        self,
        name: str,
        value: Any,
        *,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        reason: str | None = None,
    ) -> None:
        """Compare a numeric quantity, or record it while recording."""
        if (rtol, atol) != (DEFAULT_RTOL, DEFAULT_ATOL) and not reason:
            raise ValueError(f"{name}: a widened tolerance needs an explicit reason")
        actual = _as_float(_flatten(name, value))
        if self._recording:
            self._recorded.update(actual)
            return
        keys = sorted(actual)
        expected = _as_float(self._expect(name, keys))
        message = f"{name} differs from pyfixest {RELEASE_VERSION}"
        np.testing.assert_allclose(
            [actual[key] for key in keys],
            [expected[key] for key in keys],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{message} ({reason})" if reason else message,
        )

    def check_exact(self, name: str, value: Any) -> None:
        """Compare a structural quantity that must match exactly."""
        entry = {name: _jsonable(value)}
        if self._recording:
            self._recorded.update(entry)
            return
        expected = self._expect(name, [name])
        assert entry == expected, (
            f"{name} differs from pyfixest {RELEASE_VERSION}: "
            f"{entry[name]!r} != {expected[name]!r}"
        )

    def skip(self, name: str, *, reason: str) -> None:
        """Document a quantity with no comparable counterpart in the release."""
        if not reason:
            raise ValueError(f"{name}: a skipped quantity needs an explicit reason")

    def _expect(self, name: str, keys: list[str]) -> dict[str, Any]:
        missing = [key for key in keys if key not in self._recorded]
        if missing:
            pytest.fail(
                f"{name} is missing from the recorded pyfixest {RELEASE_VERSION} "
                f"baseline for {self._node_id}. The cache is stale; re-record it "
                f"with `{RECORD_COMMAND}`."
            )
        return {key: self._recorded[key] for key in keys}


@pytest.fixture(scope="session")
def _release_cache(
    request: pytest.FixtureRequest,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Load the recorded baseline, or collect a new one while recording."""
    recording = bool(request.config.getoption(RECORD_OPTION))
    if recording:
        cases: dict[str, dict[str, Any]] = {}
        yield cases
        if request.session.testsfailed:
            print("\nRecording failed; the baseline was not written.")
        else:
            _write_cache(cases)
        return

    # A missing or stale baseline is a setup state, not a regression, so the
    # suite skips rather than failing: `pixi run test-release-contract` records
    # it first and therefore never lands here.
    if not CACHE_PATH.exists():
        pytest.skip(
            f"No pyfixest {RELEASE_VERSION} baseline for this platform. "
            f"Record it with `{RECORD_COMMAND}`."
        )
    with gzip.open(CACHE_PATH, "rt") as handle:
        payload = json.load(handle)
    if payload.get("fingerprint") != fingerprint():
        pytest.skip(
            f"The recorded pyfixest {RELEASE_VERSION} baseline is stale. "
            f"Re-record it with `{RECORD_COMMAND}`."
        )
    cases = payload["cases"]
    collected = {item.nodeid for item in request.session.items}
    if collected and not collected & set(cases):
        pytest.skip(
            "No collected test matches the recorded baseline's node ids, which "
            "usually means the recording pytest generated different parametrize "
            f"ids than this one. Re-record with `{RECORD_COMMAND}`."
        )
    yield cases


def _write_cache(cases: dict[str, dict[str, Any]]) -> None:
    payload = {
        "fingerprint": fingerprint(),
        "release": {
            "pyfixest": importlib.metadata.version("pyfixest"),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "platform": platform.system(),
            "architecture": platform.machine(),
        },
        "cases": cases,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as handle:
        # Non-finite values round-trip through Python's JSON encoder as
        # NaN/Infinity literals; only Python ever reads this cache.
        json.dump(payload, handle, sort_keys=True)
    temporary.replace(CACHE_PATH)
    print(f"\nRecorded {len(cases)} cases to {CACHE_PATH.relative_to(ROOT)}")


@pytest.fixture
def baseline(
    request: pytest.FixtureRequest, _release_cache: dict[str, dict[str, Any]]
) -> Baseline:
    """Record or compare the current test's results against the release."""
    recording = bool(request.config.getoption(RECORD_OPTION))
    node_id = request.node.nodeid
    if recording:
        recorded = _release_cache.setdefault(node_id, {})
    elif node_id in _release_cache:
        recorded = _release_cache[node_id]
    else:
        pytest.fail(
            f"{node_id} is missing from the recorded pyfixest {RELEASE_VERSION} "
            f"baseline. The cache is stale; re-record it with `{RECORD_COMMAND}`."
        )
    return Baseline(recorded, recording=recording, node_id=node_id)
