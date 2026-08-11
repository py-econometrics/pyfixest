from pathlib import Path

from scripts.setup_maturin_hook import PyfixestProjectFileSearcher


def test_maturin_file_searcher_ignores_test_artifacts(tmp_path: Path) -> None:
    files = {
        "src/lib.rs",
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        ".coverage",
        "coverage.xml",
        "tests/output.tex",
        ".pixi/generated.rs",
    }
    for filename in files:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    installed_extension = tmp_path / "pyfixest.so"
    installed_extension.touch()
    source_paths = PyfixestProjectFileSearcher().get_source_paths(
        tmp_path, [], installed_extension
    )

    assert {path.relative_to(tmp_path).as_posix() for path in source_paths} == {
        "src/lib.rs",
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
    }
