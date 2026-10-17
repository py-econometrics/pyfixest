"""Reference-generation regressions; run in the optional docs environment."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

quartodoc = pytest.importorskip("quartodoc")
yaml = pytest.importorskip("yaml")
blueprint = quartodoc.blueprint
layout = quartodoc.layout

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def renderer():
    spec = importlib.util.spec_from_file_location(
        "pyfixest_docs_renderer", ROOT / "docs" / "_renderer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Renderer()


def _document(name: str):
    plan = layout.Layout(
        package="pyfixest",
        sections=[layout.Section(title="Reference", contents=[name])],
    )
    page = blueprint(plan, parser="numpy").sections[0].contents[0]
    return page.contents[0]


def test_property_summary_links_have_rendered_targets(renderer):
    """A state property link must reach its description, not an absent ID."""
    page = renderer.render(_document("estimation.state.EstimationOptions"))
    links = set(re.findall(r"\]\(#(pyfixest\.[^)]+)\)", page))
    anchors = set(re.findall(r"\{\s*#([^\s}]+)", page))
    assert links, "The fixture must exercise property navigation"
    assert links <= anchors, f"Missing property targets: {links - anchors}"
    assert "Access as `fit.options.fixef_tol`." in page
    assert "Access as `EstimationOptions.fixef_tol`." not in page


def test_state_classmethods_keep_class_access(renderer):
    page = renderer.render(_document("estimation.state.ObservationWeights"))
    assert "ObservationWeights.unweighted(" in page
    assert "fit.observation_weights.is_weighted" in page


def test_keyword_only_signature_preserves_call_contract(renderer):
    signature = renderer.signature(_document("estimation.state.EstimationOptions"))
    assert "EstimationOptions(\n    *," in signature


def test_top_level_anchor_survives_quarto_title_promotion(renderer):
    page = renderer.render(_document("estimation.models.feols_.Feols.predict"))
    assert '<span id="pyfixest.estimation.models.feols_.Feols.predict"></span>' in page


def test_registered_page_names_do_not_collide_by_case():
    """ssc() and Ssc must share a page on macOS and Windows too."""
    config = yaml.safe_load((ROOT / "docs" / "_quarto.yml").read_text())
    paths = []
    for section in config["quartodoc"]["sections"]:
        for entry in section["contents"]:
            if isinstance(entry, str):
                paths.append(entry)
            elif entry.get("kind") == "page":
                paths.append(entry["path"])
            else:
                paths.append(entry["name"])
    folded = [path.casefold() for path in paths]
    assert len(folded) == len(set(folded)), "Reference filenames collide by case"


def test_first_example_precedes_parameters_without_duplication(renderer):
    from quartodoc.renderers import MdRenderer

    document = _document("estimation.models.feols_.Feols.predict")
    original = MdRenderer().render(document.obj.docstring)
    page = renderer.render(document)
    blocks = r"(?ms)^```\{python\}[^\n]*\n.*?^```[ \t]*$"
    assert re.findall(blocks, page) == re.findall(blocks, original)
    assert page.index("Quick example") < page.index("Parameters")
    assert page.count("Quick example") == 1
    assert "fit.predict(" in page


def test_result_pages_link_to_the_canonical_method(renderer):
    plan = layout.Layout(
        package="pyfixest",
        sections=[
            layout.Section(
                title="Results",
                contents=[
                    {"name": "estimation.models.feols_.Feols", "members": ["predict"]},
                    "estimation.models.feols_.Feols.predict",
                ],
            )
        ],
    )
    shaped = blueprint(plan, parser="numpy")
    renderer.summarize(shaped)
    page = renderer.render(shaped.sections[0].contents[0])
    assert "estimation.models.feols_.Feols.predict.qmd#" in page
    assert "### fit.predict()" not in page
    assert "lookup_demeaned_data" not in page
    assert "Feols(" not in page


def test_alphabetic_index_includes_container_methods(renderer):
    plan = layout.Layout(
        package="pyfixest",
        sections=[
            layout.Section(
                title="Results",
                contents=[
                    {
                        "name": "estimation.FixestMulti_.FixestMulti",
                        "members": ["fetch_model"],
                    }
                ],
            )
        ],
    )
    index = renderer.summarize(blueprint(plan, parser="numpy"))
    assert "All functions and methods" in index
    assert "FixestMulti.fetch_model" in index


def test_estimator_parameter_groups_preserve_every_parameter(renderer):
    document = _document("estimation.api.feols.feols")
    page = renderer.render(document)
    parameters = next(
        section.value
        for section in document.obj.docstring.parsed
        if section.kind.value == "parameters"
    )
    assert "### Model and data {.doc-section .doc-section-parameters}" in page
    assert "### Inference" in page
    assert "### Storage" in page
    for parameter in parameters:
        assert (
            len(re.findall(rf"^\| {re.escape(parameter.name)}\s*\|", page, re.M)) == 1
        )


def test_reference_checker_catches_broken_destinations(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pyfixest_reference_checker", ROOT / "scripts" / "check_reference.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    target = tmp_path / "result.qmd"
    target.write_text("# Result {#pyfixest.result}\n")
    page = tmp_path / "index.qmd"
    page.write_text(
        "[Valid HTML route](result.html#pyfixest.result)\n"
        "[Missing property](result.qmd#pyfixest.result.missing)\n"
        "[Missing page](absent.qmd)\n"
        "```python\n'[Example text](not-a-link.qmd)'\n```\n"
    )
    issues = checker.check_links(tmp_path, [page])
    assert len(issues) == 2
    assert any("missing anchor" in issue for issue in issues)
    assert any("missing target" in issue for issue in issues)


def test_reference_checker_rejects_anchors_only_in_examples(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pyfixest_reference_checker", ROOT / "scripts" / "check_reference.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    target = tmp_path / "result.qmd"
    target.write_text(
        "# Result {#actual}\n```markdown\n# Example {#example-only}\n```\n"
    )
    page = tmp_path / "index.qmd"
    page.write_text(
        "[Valid](result.qmd#actual)\n"
        "[Automatic slug](result.qmd#result)\n"
        "[Example ID](result.qmd#example-only)\n"
    )
    issues = checker.check_links(tmp_path, [page])
    assert len(issues) == 2
    assert all("missing anchor" in issue for issue in issues)


def test_rendered_reference_checker_requires_actual_html_ids(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pyfixest_reference_checker", ROOT / "scripts" / "check_reference.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    reference = tmp_path / "reference"
    reference.mkdir()
    target = reference / "result.html"
    target.write_text('<h1 class="title">Result</h1>')
    index = reference / "index.html"
    index.write_text('<a href="result.html#pyfixest.result">Result</a>')
    assert len(checker.check_rendered_links(tmp_path)) == 1
    target.write_text('<h1 class="title">Result</h1><span id="pyfixest.result"></span>')
    assert checker.check_rendered_links(tmp_path) == []


def test_parameter_code_preserves_formula_pipes(renderer):
    page = renderer.render(_document("estimation.api.feols.feols"))
    assert '`"Y ~ X1 + X2 | FE1 + FE2"`' in page
    assert r"\| FE1" not in page


def test_reference_checker_allows_documented_variadic_parameters():
    import griffe

    spec = importlib.util.spec_from_file_location(
        "pyfixest_reference_checker", ROOT / "scripts" / "check_reference.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    package = griffe.visit(
        "example",
        filepath=Path("example.py"),
        code='''
def combine(*args):
    """Combine values.

    Parameters
    ----------
    *args : tuple
        Values to combine.
    """
''',
        docstring_parser="numpy",
    )
    assert checker.check_parameters(package, ["combine"]) == []


def test_summary_code_preserves_formula_pipes_and_regex_backslashes(renderer):
    description = r"Formula ``Y ~ X | f1 + f2``; regex `r'\|'`; plain A | B."
    row = renderer._summary_row("[Result](result.qmd)", description)
    assert "``Y ~ X | f1 + f2``" in row
    assert r"`r'\|'`" in row
    assert "plain A | B" not in row
