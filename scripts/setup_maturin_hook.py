"""Install the maturin import hook with .pixi excluded from source scanning."""

import site
from pathlib import Path

SITECUSTOMIZE_CONTENT = """\
# <maturin_import_hook>
try:
    import maturin_import_hook
    from maturin_import_hook.project_importer import DefaultProjectFileSearcher
    from maturin_import_hook.settings import MaturinSettings

    excluded_dirs = DefaultProjectFileSearcher.DEFAULT_SOURCE_EXCLUDED_DIR_NAMES | {".pixi"}
    file_searcher = DefaultProjectFileSearcher(source_excluded_dir_names=excluded_dirs)

    maturin_import_hook.install(
        settings=MaturinSettings(release=True, strip=True, color=True, uv=True),
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
