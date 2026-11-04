"""MkDocs hooks: README links point at GitHub (for GitHub and PyPI); on the site, keep readers on the site."""

import re

from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url

_REPO = "https://github.com/rhnfzl/SqueakyCleanText/"
_HREF = re.compile(r'href="' + re.escape(_REPO) + r'(?:blob|tree)/main/([^"#]+)(#[^"]*)?"')
# repo path -> docs/ page; anything under docs/ maps to itself
_PAGES = {"CHANGELOG.md": "changelog.md", "benchmarks": "benchmarks.md", "benchmarks/README.md": "benchmarks.md"}


def on_page_markdown(markdown: str, page: Page, **kwargs: object) -> str:
    # stub pages that only include another file: the edit button opens that file
    if "edit_url" in page.meta:
        page.edit_url = page.meta["edit_url"]
    return markdown


def on_page_content(html: str, page: Page, files: Files, **kwargs: object) -> str:
    # runs after pymdownx.snippets, so README text included into a page is covered too
    def local(m: re.Match[str]) -> str:
        path = m.group(1)
        src = path.removeprefix("docs/") if path.startswith("docs/") else _PAGES.get(path)
        target = files.get_file_from_path(src) if src else None
        if target is None:
            return m.group(0)
        return 'href="{}{}"'.format(get_relative_url(target.url, page.url), m.group(2) or "")

    return _HREF.sub(local, html)
