"""Reject new Python comments in the package and prune retired exceptions."""

import argparse
import io
import json
import subprocess
import tokenize
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "comment_baseline.json"


def _package_comments() -> dict[str, list[tokenize.TokenInfo]]:
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "pyfixest"], cwd=ROOT
    )
    comments = {}
    for name in files.decode().split("\0"):
        path = ROOT / name
        if path.suffix not in {".py", ".pyi"} or not path.is_file():
            continue
        tokens = tokenize.tokenize(io.BytesIO(path.read_bytes()).readline)
        comments[name] = [token for token in tokens if token.type == tokenize.COMMENT]
    return comments


def main() -> int:
    """Check the current files or remove unused baseline entries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prune", action="store_true", help="Remove retired exceptions."
    )
    args = parser.parse_args()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    remaining = {}
    found_new = False

    for path, tokens in _package_comments().items():
        allowed = Counter(baseline.get(path, {}))
        retained: Counter[str] = Counter()
        for token in tokens:
            comment = token.string.rstrip()
            if retained[comment] < allowed[comment]:
                retained[comment] += 1
            else:
                print(f"{path}:{token.start[0]}: new comment: {comment}")
                found_new = True
        if retained:
            remaining[path] = dict(retained)

    if args.prune:
        BASELINE.write_text(
            json.dumps(remaining, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif remaining != baseline:
        print(
            "Retired comment exceptions. "
            "Run pixi run python scripts/check_comments.py --prune"
        )
        return 1

    return int(found_new)


if __name__ == "__main__":
    raise SystemExit(main())
