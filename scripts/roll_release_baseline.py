"""Roll the release contract on to a newer pyfixest release.

The pinned release is the reference `tests/test_release_contract.py` compares
against. Rolling it is how the suite regains the comparisons that the
documented differences in that file currently skip, so the natural moment is
just after tagging a release.

    pixi run roll-release-baseline          # newest release tag
    pixi run roll-release-baseline 0.61.0   # a specific release
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._release_baseline import (  # noqa: E402
    RELEASE_MANIFEST,
    RELEASE_VERSION,
    newest_release_tag,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="release to pin; defaults to the newest release tag in this checkout",
    )
    target = parser.parse_args().version or newest_release_tag()
    if target is None:
        raise SystemExit(
            "No version given and no release tag found. Pass one explicitly."
        )
    if target == RELEASE_VERSION:
        print(f"Already pinned to pyfixest {target}.")
        return

    manifest = RELEASE_MANIFEST.read_text()
    rolled = re.sub(r'(?m)^pyfixest = "==[^"]+"$', f'pyfixest = "=={target}"', manifest)
    if rolled == manifest:
        raise SystemExit(f"Could not find the pyfixest pin in {RELEASE_MANIFEST}.")
    RELEASE_MANIFEST.write_text(rolled)
    print(f"Pinned pyfixest {RELEASE_VERSION} -> {target}; relocking.")

    subprocess.run(
        ["pixi", "lock", "--manifest-path", str(RELEASE_MANIFEST)], check=True
    )
    subprocess.run(
        [
            "pixi",
            "run",
            "--locked",
            "--clean-env",
            "--manifest-path",
            str(RELEASE_MANIFEST),
            "record",
        ],
        check=True,
    )
    print(
        "Recorded. Now re-run the suite and remove the documented differences in "
        "tests/test_release_contract.py that this release has caught up with."
    )


if __name__ == "__main__":
    main()
