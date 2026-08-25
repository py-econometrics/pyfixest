from pathlib import Path

import pytest

from scripts.reference.compare_fixest import main


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "case_path",
    [
        Path("scripts/reference/cases/feols-smoke.toml"),
        Path("scripts/reference/cases/fepois-smoke.toml"),
    ],
)
def test_fixest_reference_cli_smoke(case_path, capsys):
    assert main([str(case_path)]) == 0
    assert "Overall: PASS" in capsys.readouterr().out
