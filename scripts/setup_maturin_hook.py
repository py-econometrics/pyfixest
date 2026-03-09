"""Install the maturin import hook for automatic Rust extension rebuilds.

pyfixest includes a Rust extension (pyfixest.core._core_impl) built with maturin.
During development, we use maturin-import-hook to automatically rebuild the
extension whenever `import pyfixest` detects that .rs source files have changed.
This avoids having to manually run `maturin develop` after every Rust change.

How it works:
  This script writes a sitecustomize.py into the environment's site-packages.
  Python executes sitecustomize.py on startup (before any user code), so the
  hook is active in every Python session without any manual setup.

  The hook is configured to:
  - Build in release mode with symbols stripped (for performance)
  - Exclude .pixi/ from source file scanning (it contains nested Python
    environments that would confuse the file searcher)

This script is run by the `_setup` pixi task, which all test tasks depend on.
It is cached via a sentinel file so it only runs once per environment.

See Also
--------
  - maturin-import-hook docs: https://github.com/PyO3/maturin-import-hook
  - Python sitecustomize: https://docs.python.org/3/library/site.html
"""

import site
from pathlib import Path

SITECUSTOMIZE_CONTENT = """\
# <maturin_import_hook>
try:
    import maturin_import_hook
    import os
    import sys
    from maturin_import_hook.project_importer import DefaultProjectFileSearcher
    from maturin_import_hook.settings import MaturinSettings

    # maturin needs CONDA_PREFIX (or VIRTUAL_ENV) to locate the environment.
    # When launched directly from PyCharm (not via pixi run), neither is set.
    if not os.environ.get("CONDA_PREFIX") and not os.environ.get("VIRTUAL_ENV"):
        os.environ["CONDA_PREFIX"] = sys.prefix

    excluded_dirs = DefaultProjectFileSearcher.DEFAULT_SOURCE_EXCLUDED_DIR_NAMES | {".pixi"}
    file_searcher = DefaultProjectFileSearcher(source_excluded_dir_names=excluded_dirs)

    maturin_import_hook.install(
        settings=MaturinSettings(release=True, strip=True, color=True),
        file_searcher=file_searcher,
        enable_project_importer=True,
        enable_rs_file_importer=True,
    )
except Exception as e:
    raise RuntimeError(
        f"{e}\\n>> ERROR in managed maturin_import_hook installation. "
        "Remove the sitecustomize.py in your environment's site-packages.\\n",
    )
# </maturin_import_hook>
"""


def main() -> None:
    site_packages = Path(site.getsitepackages()[0])
    target = site_packages / "sitecustomize.py"

    # Remove existing hook content if present
    if target.exists():
        content = target.read_text()
        if "<maturin_import_hook>" in content:
            import re

            content = re.sub(
                r"# <maturin_import_hook>.*?# </maturin_import_hook>\n?",
                "",
                content,
                flags=re.DOTALL,
            )
            if content.strip():
                target.write_text(content)
            else:
                target.unlink()

    # Write new content
    if target.exists():
        with open(target, "a") as f:
            f.write("\n" + SITECUSTOMIZE_CONTENT)
    else:
        target.write_text(SITECUSTOMIZE_CONTENT)

    print(f"maturin import hook installed to {target}")


if __name__ == "__main__":
    main()
