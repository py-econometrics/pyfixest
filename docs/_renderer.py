"""Render a task-oriented reference while preserving object IDs and URLs.

Callable labels show the package or fitted-result access used in examples.
Shared methods link to one canonical page; property summaries have rendered
detail targets. The first executable example moves before parameter tables,
and long estimator parameter tables are grouped without changing docstrings.

Wired in via ``_quarto.yml``:

    quartodoc:
      renderer:
        style: _renderer.py

quartodoc loads the file relative to the working directory of ``quartodoc
build``. The ``docs-build`` task runs it from ``docs/`` (``cd docs && quartodoc
build ...``), so this file sits next to ``_quarto.yml``. quartodoc instantiates
a class named exactly ``Renderer``.
"""

from __future__ import annotations

import re

from plum import dispatch
from quartodoc import ast as qast
from quartodoc import layout
from quartodoc._griffe_compat import dataclasses as dc
from quartodoc._griffe_compat import docstrings as ds
from quartodoc.renderers import MdRenderer
from quartodoc.renderers.base import sanitize

import pyfixest

_RESULT_CLASSES = {
    "Feols",
    "Feiv",
    "Fepois",
    "Feglm",
    "Felogit",
    "Feprobit",
    "Fegaussian",
    "Quantreg",
    "FixestMulti",
    "LPDID",
    "GelbachDecomposition",
}
_INSTANCE_ACCESS = {
    "EstimationOptions": "fit.options",
    "GlmEstimationOptions": "fit.options",
    "QuantregEstimationOptions": "fit.options",
    "ModelDescription": "fit.model",
    "ObservationWeights": "fit.observation_weights",
    "DroppedRowCounts": "fit.sample_info.dropped_by_stage",
    "EstimationSample": "fit.sample_info",
    "FixedEffectCounts": "fit.fixef_counts",
    "CollinearityCheck": "fit.collinearity",
    "WithinLinearData": "fit.within_data",
    "WithinIvData": "fit.within_data",
    "GlmWorkingState": "fit.working_state",
    "FittedValues": "fit.fitted_values",
    "SandwichComponents": "fit.sandwich",
    "VcovSpec": "fit.variance_covariance.spec",
    "VarianceCovariance": "fit.variance_covariance",
    "FirstStage": "fit.first_stage",
    "FirstStageDiagnostics": "fit.first_stage.diagnostics",
    "WaldTest": "fit.wald",
    "Capabilities": "fit.capabilities",
    "RitestStatistics": "fit.ritest_statistics",
    "CoefficientTable": "fit.coeftable",
    "FitStatistics": "fit.fitstat",
    "ModelMatrix": "fit.model_matrix",
    "FixedEffectEstimates": "fit.fixef_estimates",
    "FixedEffect": "fit.fixef_estimates.fixed_effects[formulaic_name]",
    "QuantregSolution": "fit.solution",
    "SampleSplit": "fit.model.sample_split",
    "InferenceDist": "fit.model.inference_dist",
    "RetentionPolicy": "fit.options.retention",
    "EventStudyDesign": "fit.event_study_design",
    "Ssc": "fit.options.ssc",
    "GelbachResults": "decomposition.results",
    "Formula": "fit.model.parsed_formula",
    "DegreesOfFreedomCounts": "counts",
    "SmallSampleCorrection": "correction",
    "Preconditioner": "fit.preconditioner",
}
_PARAMETER_GROUPS = {
    "Model and data": {
        "fml",
        "data",
        "family",
        "quantile",
        "weights",
        "weights_type",
        "offset",
        "drop_intercept",
        "context",
    },
    "Inference": {"vcov", "vcov_kwargs", "ssc"},
    "Multiple estimation": {"split", "fsplit", "multi_method"},
    "Computation": {
        "method",
        "tol",
        "maxiter",
        "demeaner",
        "fixef_rm",
        "collin_tol",
        "solver",
        "iwls_tol",
        "iwls_maxiter",
        "separation_check",
        "accelerate",
        "seed",
    },
    "Storage": {"copy_data", "store_data", "lean"},
    "Compatibility options": {"use_compression", "reps"},
}


class Renderer(MdRenderer):
    """MdRenderer that displays short object names instead of import paths."""

    style = "pyfixest_short_name"

    def _render_table(self, rows, headers, style) -> str:
        table = super()._render_table(rows=rows, headers=headers, style=style)
        # quartodoc escapes every pipe, but Pandoc already protects pipes in
        # inline code and would display the added backslash literally.
        return re.sub(
            r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)",
            lambda match: match.group(0).replace(r"\|", "|"),
            table,
        )

    def _summary_row(self, link, description):
        # Summary tables need escaping only outside inline code. Keep literal
        # regex backslashes as well as formula pipes intact inside code spans.
        parts = []
        start = 0
        for match in re.finditer(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", description):
            parts.append(
                sanitize(description[start : match.start()], allow_markdown=True)
            )
            parts.append(match.group(0))
            start = match.end()
        parts.append(sanitize(description[start:], allow_markdown=True))
        return f"| {link} | {''.join(parts)} |"

    @dispatch
    def render(self, el: layout.DocClass) -> str:
        """Link shared methods once and render property detail targets."""
        separate = [member for member in el.members if self._method_page(member)]
        embedded = [member for member in el.members if member not in separate]
        body = super().render(el.copy(update={"members": embedded}))
        if separate:
            heading = "#" * (self.crnt_header_level + 1)
            rows = "\n".join(self.summarize(member) for member in separate)
            body += (
                f"\n\n{heading} Methods\n\n| Method | Purpose |\n| --- | --- |\n{rows}"
            )
        attributes = [member for member in embedded if member.obj.is_attribute]
        with self._increment_header(1 if el.flat else 2):
            details = [self.render(member) for member in attributes]
        return "\n\n".join([body, *details])

    @dispatch
    def render(self, el: dc.Parameters) -> list[str]:  # noqa: F811
        """Preserve a leading keyword-only marker in dataclass signatures."""
        parts = super().render(el)
        if len(el) and el[0].kind == dc.ParameterKind.keyword_only:
            parts.insert(0, "*")
        return parts

    @dispatch
    def signature(self, el: layout.Doc) -> str:
        """Show callable access and suppress internal result constructors."""
        if el.obj.is_class and el.obj.name in _RESULT_CLASSES | {"Preconditioner"}:
            return ""
        signature = super().signature(el)
        if el.obj.is_function:
            signature = signature.replace(
                f"{el.obj.name}(", f"{self._call_name(el)}(", 1
            )
        return signature

    @dispatch
    def render(self, el: dc.Docstring) -> str:  # noqa: F811
        """Put one runnable example before parameter tables without copying it."""
        sections = list(qast.transform(el.parsed))
        examples = next(
            (section for section in sections if section.kind.value == "examples"), None
        )
        quick = ""
        remaining = ""
        if examples is not None:
            rendered = self.render(examples)
            block = re.search(r"(?ms)^```\{python\}[^\n]*\n.*?^```[ \t]*$", rendered)
            if block:
                quick = rendered[: block.end()]
                remaining = rendered[block.end() :].strip()

        parts = []
        parent = el.parent
        moved = bool(quick)
        quick_added = False
        for section in sections:
            if (
                parent
                and parent.is_class
                and parent.name in _RESULT_CLASSES
                and section.kind.value == "parameters"
            ):
                continue
            if section is examples and moved:
                if remaining:
                    parts.append(f"{self.render_header(section)}\n\n{remaining}")
                continue
            if section.kind.value == "text":
                parts.append(self.render(section))
                if moved and not quick_added:
                    parts.append(
                        f"{'#' * self.crnt_header_level} Quick example\n\n{quick}"
                    )
                    quick_added = True
                continue
            if (
                parent
                and parent.name in {"feols", "fepois", "feglm", "quantreg"}
                and section.kind.value == "parameters"
            ):
                groups = []
                assigned = set()
                for title, names in _PARAMETER_GROUPS.items():
                    values = [
                        parameter
                        for parameter in section.value
                        if parameter.name in names
                    ]
                    if values:
                        assigned.update(parameter.name for parameter in values)
                        groups.append(
                            f"{'#' * (self.crnt_header_level + 1)} {title} {{.doc-section .doc-section-parameters}}\n\n{self.render(ds.DocstringSectionParameters(values))}"
                        )
                other = [
                    parameter
                    for parameter in section.value
                    if parameter.name not in assigned
                ]
                if other:
                    groups.append(self.render(ds.DocstringSectionParameters(other)))
                body = "\n\n".join(groups)
            else:
                body = self.render(section)
            parts.append(f"{self.render_header(section)}\n\n{body}")
        return "\n\n".join(parts)

    def _call_name(self, el: layout.Doc) -> str:
        obj = el.obj
        parent = getattr(obj, "parent", None)
        if parent and parent.is_class:
            if "classmethod" not in obj.labels and parent.name in _INSTANCE_ACCESS:
                return f"{_INSTANCE_ACCESS[parent.name]}.{obj.name}"
            if parent.name == "FixestMulti":
                return f"models.{obj.name}"
            if parent.name == "GelbachDecomposition":
                return f"decomposition.{obj.name}"
            if parent.name in _RESULT_CLASSES or parent.name == "SaturatedEventStudy":
                return f"fit.{obj.name}"
            if obj.is_attribute:
                return f"instance.{obj.name}"
            return f"{parent.name}.{obj.name}"
        if obj.name == "factor_interaction":
            return "i"
        if obj.name in pyfixest.__all__:
            return f"pf.{obj.name}"
        return obj.name

    def _method_page(self, el: layout.Doc) -> tuple[str, str] | None:
        if not el.obj.is_function:
            return None
        return getattr(self, "_method_pages", {}).get(el.obj.canonical_path)

    def _short_name(self, el: layout.Doc) -> str:
        obj = el.obj
        if obj.is_function:
            return f"{self._call_name(el)}()"
        parent = getattr(obj, "parent", None)
        parent_kind = getattr(getattr(parent, "kind", None), "value", None)
        if parent_kind == "class":
            return self._call_name(el)
        return obj.name

    @dispatch
    def render_header(self, el: layout.Doc) -> str:
        """Render a page header using the short name, keeping the anchor."""
        anchor = f"{{ #{el.obj.path} }}"
        heading = f"{'#' * self.crnt_header_level} {self._short_name(el)} {anchor}"
        if self.crnt_header_level == 1:
            # Quarto promotes the first H1 into the title and drops its ID.
            # A separate anchor survives, including on combined object pages.
            heading = f'# {self._short_name(el)}\n\n<span id="{el.obj.path}"></span>'
        obj = el.obj
        if obj.name == "factor_interaction":
            access = "Use `i(...)` inside a model formula."
        elif obj.parent and obj.parent.is_class:
            suffix = "(...)" if obj.is_function else ""
            access = f"Access as `{self._call_name(el)}{suffix}`."
        elif obj.is_function and obj.name in pyfixest.__all__:
            access = f"Access as `pf.{obj.name}(...)` after `import pyfixest as pf`."
        else:
            module, _, name = obj.path.rpartition(".")
            if name in _RESULT_CLASSES - {"Feglm", "LPDID", "GelbachDecomposition"}:
                module = "pyfixest.estimation"
            access = f"Import with `from {module} import {name}`."
        return f"{heading}\n\n{access}"

    @dispatch
    def summarize(
        self, el: layout.Doc, path: str | None = None, shorten: bool = False
    ) -> str:
        """Render an index-table row using the short name."""
        name = self._short_name(el)
        method_page = self._method_page(el) if path is None else None
        if method_page:
            page, anchor = method_page
            link = f"[{name}]({page}.qmd#{anchor})"
        elif path is None:
            link = f"[{name}](#{el.anchor})"
        else:
            link = f"[{name}]({path}.qmd#{el.anchor})"
        return self._summary_row(link, self.summarize(el.obj))

    @dispatch
    def summarize(self, el: layout.Layout) -> str:  # noqa: F811
        """Add task discovery and an alphabetic view of the registered API."""
        self._method_pages = {}
        entries = []
        registered = []
        for section in el.sections:
            for page in section.contents:
                for doc in page.contents:
                    if not isinstance(doc, layout.Doc):
                        continue
                    registered.append((page.path, doc))
                    if doc.obj.is_function:
                        self._method_pages[doc.obj.canonical_path] = (
                            page.path,
                            doc.anchor,
                        )
        seen = set()
        for path, doc in registered:
            for item in [doc, *getattr(doc, "members", ())]:
                target = self._method_page(item)
                item_path, anchor = target or (path, item.anchor)
                if (item_path, anchor) in seen:
                    continue
                seen.add((item_path, anchor))
                name = self._short_name(item)
                if (
                    item.obj.parent
                    and item.obj.parent.is_class
                    and item.obj.parent.name not in {"Feols"}
                ):
                    name += f" ({item.obj.parent.name})"
                entries.append((name, item_path, anchor, item))
        sections = super().summarize(el)
        rows = "\n".join(
            self._summary_row(f"[{name}]({path}.qmd#{anchor})", self.summarize(doc.obj))
            for name, path, anchor, doc in sorted(
                entries, key=lambda entry: entry[0].casefold()
            )
        )
        return (
            "{{< include /reference-guide/_tasks.qmd >}}\n\n"
            + sections
            + "\n\n## All functions and methods {#all-functions-and-methods}\n\n"
            + "| Name | Purpose |\n| --- | --- |\n"
            + rows
        )
