"""Build the static GitHub Pages copy of the landing page.

    python scripts/build_pages.py --out /tmp/pardadb-pages

PardaDB serves the landing page itself, so the page links to /dashboard and calls /api/*. GitHub Pages
has neither: this copy makes asset paths relative, points the DEMO buttons at the install section, and
adds a banner saying the live demo runs on the visitor's machine. The live-engine section already falls
back to a recorded real run when /api is not reachable, so nothing else changes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent.parent / "pardadb" / "dashboard"
REPO_URL = "https://github.com/deadhearth01/pardadb"

BANNER = (
    '<div role="note" style="position:relative;z-index:60;padding:10px 16px;text-align:center;'
    'font:600 13.5px/1.4 ui-monospace,Menlo,Consolas,monospace;background:var(--accent);color:var(--on-accent)">'
    "This is the static preview. The live demo (dashboard, real queries) runs on your machine: "
    '<a href="#install" style="color:inherit;text-decoration:underline">install in one command</a> '
    f'· <a href="{REPO_URL}" style="color:inherit;text-decoration:underline">source on GitHub</a></div>'
)

# (old, new, expected count or None for "at least one")
REWRITES: list[tuple[str, str, int | None]] = [
    ("url(/assets/", "url(assets/", None),
    ('href="/assets/', 'href="assets/', None),
    ('href="/dashboard"', 'href="#install"', None),
]


def static_landing(html: str) -> str:
    """Return the landing page rewritten for a static host."""
    for old, new, expected in REWRITES:
        count = html.count(old)
        if count == 0 or (expected is not None and count != expected):
            raise ValueError(f"landing page changed: expected {old!r} in the page")
        html = html.replace(old, new)
    if 'id="install"' not in html:
        raise ValueError("landing page has no #install section for the DEMO buttons to point at")
    body = html.index("<body")
    body_end = html.index(">", body) + 1
    return html[:body_end] + BANNER + html[body_end:]


def build(out: Path) -> None:
    """Write index.html, assets/ and .nojekyll into `out` (replacing what is there)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    page = static_landing((DASHBOARD / "landing.html").read_text(encoding="utf-8"))
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copytree(DASHBOARD / "assets", out / "assets")
    (out / ".nojekyll").write_text("")  # serve files as they are, no Jekyll processing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="output folder (replaced)")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
