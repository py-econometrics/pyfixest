"""Gate docs deployment using the paginated GitHub releases API response."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def main() -> None:
    """Publish only the highest published stable version, ignoring API order."""
    # `gh api --paginate --slurp` supplies a list of release-list pages.
    pages = json.load(sys.stdin)
    stable_versions = {}
    for page in pages:
        for release in page:
            if release["draft"] or release["prerelease"]:
                continue
            tag = release["tag_name"]
            match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag)
            if match:
                stable_versions[tag] = tuple(int(part) for part in match.groups())

    if not stable_versions:
        sys.exit("No published stable version found; refusing docs deployment.")

    tag = os.environ["GITHUB_REF"].removeprefix("refs/tags/")
    publish = stable_versions.get(tag) == max(stable_versions.values())
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"publish={str(publish).lower()}\n")
    print(f"Docs deployment for {tag}: {'publish' if publish else 'skip'}")


if __name__ == "__main__":
    main()
