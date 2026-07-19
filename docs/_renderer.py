"""Custom quartodoc renderer that labels objects by their short name.

Upstream quartodoc (0.11.x) renders the full import path
(e.g. ``estimation.api.feols.feols``) in page headers, the reference index,
and the sidebar, because they all read ``layout.Doc.name`` directly and there
is no config option to change it. This subclass shortens those labels to the
object name (``feols``), keeping the ``Class.method`` form for methods so they
stay unambiguous. Anchors, cross-links, and signatures are left untouched.

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

from plum import dispatch
from quartodoc import layout
from quartodoc.renderers import MdRenderer


class Renderer(MdRenderer):
    """MdRenderer that displays short object names instead of import paths."""

    style = "pyfixest_short_name"

    def _short_name(self, el: layout.Doc) -> str:
        obj = el.obj
        parent = getattr(obj, "parent", None)
        parent_kind = getattr(getattr(parent, "kind", None), "value", None)
        if parent_kind == "class":
            # e.g. Feols.tidy, so methods keep their class for context
            return f"{parent.name}.{obj.name}"
        return obj.name

    @dispatch
    def render_header(self, el: layout.Doc) -> str:
        """Render a page header using the short name, keeping the anchor."""
        anchor = f"{{ #{el.obj.path} }}"
        return f"{'#' * self.crnt_header_level} {self._short_name(el)} {anchor}"

    @dispatch
    def summarize(
        self, el: layout.Doc, path: str | None = None, shorten: bool = False
    ) -> str:
        """Render an index-table row using the short name."""
        name = self._short_name(el)
        if path is None:
            link = f"[{name}](#{el.anchor})"
        else:
            link = f"[{name}]({path}.qmd#{el.anchor})"
        return self._summary_row(link, self.summarize(el.obj))
