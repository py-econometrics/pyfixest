from scripts.agent.change_scope import classify_changes, classify_path, risk_flags


def test_classify_path_maps_estimation_api_to_numerical_risk_domain():
    assert classify_path("pyfixest/estimation/api/feols.py") == {
        "api",
        "numerical",
        "python",
    }


def test_classify_path_maps_special_test_domains():
    assert classify_path("tests/test_hac_vs_fixest.py") == {
        "hac",
        "python",
        "reference",
        "tests",
    }


def test_classify_path_falls_back_conservatively():
    assert classify_path("unexpected/new-tool.conf") == {"unknown"}


def test_risk_flags_cover_public_core_and_reference_changes():
    assert risk_flags("pyfixest/estimation/api/feols.py") == {"public-api"}
    assert risk_flags("pyfixest/estimation/formula/parse.py") == {
        "formula-semantics",
        "shared-estimation-core",
    }
    assert risk_flags("tests/data/reference.csv") == {"stored-reference"}


def test_classify_changes_is_sorted_and_deduplicated():
    scope = classify_changes(
        base="parent",
        merge_base="abc123",
        files=[
            "tests/test_hac_vs_fixest.py",
            "pyfixest/estimation/api/feols.py",
            "tests/test_hac_vs_fixest.py",
        ],
    )

    assert scope.files == (
        "pyfixest/estimation/api/feols.py",
        "tests/test_hac_vs_fixest.py",
    )
    assert scope.domains == ("api", "hac", "numerical", "python", "reference", "tests")
    assert scope.risks == ("public-api",)
