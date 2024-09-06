"""Script to run all notebooks example notebooks.

Taken from: https://github.com/pymc-labs/pymc-marketing/blob/main/scripts/run_notebooks/runner.py
"""

import logging
from pathlib import Path

import papermill
from joblib import Parallel, delayed
from tqdm import tqdm

KERNEL_NAME: str = "python3"
DOCS = Path("docs")
NOTEBOOKS: list[Path] = [
    # DOCS / "compare-fixest-pyfixest.ipynb",  # needs R
    DOCS / "difference-in-differences.ipynb",
    DOCS / "marginaleffects.ipynb",
    DOCS / "quickstart.ipynb",
    DOCS / "replicating-the-effect.ipynb",
    # DOCS / "stargazer.ipynb",  # failing notebook
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_cwd_from_notebook_path(notebook_path: Path) -> str:
    return str(notebook_path).rsplit("/", 1)[0]


def run_notebook(notebook_path: Path) -> None:
    cwd = get_cwd_from_notebook_path(notebook_path)
    logging.info(f"Running notebook: {notebook_path.name}")
    papermill.execute_notebook(
        input_path=str(notebook_path),
        output_path=None,
        kernel_name=KERNEL_NAME,
        cwd=cwd,
    )


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting notebook runner")
    logging.info(f"Notebooks to run: {NOTEBOOKS}")
    Parallel(n_jobs=-1)(
        delayed(run_notebook)(notebook_path) for notebook_path in tqdm(NOTEBOOKS)
    )

    logging.info("Notebooks run successfully!")
