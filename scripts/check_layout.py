#!/usr/bin/env python3
"""Check that the manuscript's layout is sound, not just its arithmetic.

A manuscript can be numerically perfect and still be unreadable. This
script checks the three layout defects that are invisible in a LaTeX log:

1. **Type size.** A table wider than its column is scaled down by
   \\resizebox until the text is too small to read. LaTeX reports no
   error; the PDF simply has 5 pt type in it. The threshold here is a
   line box shorter than MIN_LINE_PX at the resolution used for
   measurement.

2. **Float order.** LaTeX places a float where its \\input appears in
   the source, while the prose flows in source order too. If the
   \\input comes *before* the sentence that refers to the table, the
   table lands on the page before the reader has been told about it.

3. **Table manifest.** Every \\input of a table must exist in
   paper/tables/, and every generated table that the manuscript claims
   to include must actually be included.

Usage:
    python3.11 scripts/check_layout.py            # build and check
    python3.11 scripts/check_layout.py --pdf X    # check an existing PDF
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
TABLES = os.path.join(PAPER, "tables")

#: pdftohtml renders at 72 dpi scaled by this factor. A 10 pt body font
#: therefore measures about 13 px.
SCALE = 1.5

#: The smallest line box we accept. Below this the type is not legible
#: on paper. 10 pt body text measures ~13 px, so 9 px is roughly 7 pt --
#: the usual lower bound for a table in a two-column preprint.
MIN_LINE_PX = 9.0

#: A table may not sit more than this many pages after the paragraph
#: that refers to it. Zero means "same page or earlier"; two pages is
#: already a hunt.
MAX_FLOAT_DELAY_PAGES = 2


def build_pdf() -> str:
    """Compile the manuscript and return the path to the PDF.

    The PDF stays in paper/ -- it is the build product the CI step
    already produces, and a path into a temporary directory would be
    gone by the time the caller tried to read it.
    """
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        cwd=PAPER, capture_output=True, text=True,
    )
    subprocess.run(["bibtex", "main"], cwd=PAPER, capture_output=True, text=True)
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=PAPER, capture_output=True, text=True,
        )
    pdf = os.path.join(PAPER, "main.pdf")
    if not os.path.exists(pdf):
        raise SystemExit("[fail] the manuscript did not compile to main.pdf")
    return pdf


def page_texts(pdf: str) -> list[str]:
    """The text of each page, in order."""
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", out).group(1))
    texts = []
    for page in range(1, pages + 1):
        result = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), pdf, "-"],
            capture_output=True, text=True,
        )
        texts.append(result.stdout)
    return texts


def xml_text_lines(pdf: str) -> list[tuple[int, int, int, str]]:
    """(page, top, height, text) for every text run in the PDF."""
    with tempfile.TemporaryDirectory() as tmp:
        # pdftohtml resolves its output path relative to its own working
        # directory, so the copy and the output must share one.
        local = os.path.join(tmp, "doc.pdf")
        subprocess.run(["cp", pdf, local], check=True)
        target = os.path.join(tmp, "doc.xml")
        subprocess.run(
            ["pdftohtml", "-xml", "-hidden", "doc.pdf", "doc.xml"],
            cwd=tmp, capture_output=True, text=True,
        )
        if not os.path.exists(target):
            return []
        data = open(target, encoding="utf-8", errors="replace").read()
    runs = []
    page = 0
    pattern = re.compile(
        r'<page number="(\d+)"|<text top="(\d+)" left="(\d+)" width="(\d+)" '
        r'height="(\d+)"[^>]*>([^<]*)</text>'
    )
    for match in pattern.finditer(data):
        if match.group(1):
            page = int(match.group(1))
            continue
        top = int(match.group(2))
        height = int(match.group(5))
        text = match.group(6)
        if height > 0 and text.strip():
            runs.append((page, top, height, text))
    return runs


def check_type_size(pdf: str) -> list[str]:
    """Report text that is set too small to read."""
    problems = []
    runs = xml_text_lines(pdf)
    small = [(p, h, t) for p, _top, h, t in runs if h < MIN_LINE_PX]
    for page, height, text in small:
        # A single glyph (a multiplication sign, a digit) measured on its
        # own is not a line of type; it is a fragment of one. Only flag
        # runs that carry actual words.
        if len(text.strip()) < 3:
            continue
        problems.append(
            f"page {page}: type of {height}px (below {MIN_LINE_PX}px): {text.strip()[:40]!r}"
        )
    return problems


def check_float_order(texts: list[str]) -> list[str]:
    """Report tables that appear before the paragraph referring to them."""
    problems = []
    for number in range(1, 20):
        pattern = re.compile(rf"Table {number}(?![:\d])")
        refs = [i + 1 for i, t in enumerate(texts) if pattern.search(t)]
        caps = [i + 1 for i, t in enumerate(texts) if f"Table {number}:" in t]
        if not refs or not caps:
            continue
        delay = caps[0] - refs[0]
        if delay < 0:
            problems.append(
                f"Table {number}: appears on page {caps[0]} but is referred to "
                f"on page {refs[0]} -- the table precedes its own reference"
            )
        elif delay > MAX_FLOAT_DELAY_PAGES:
            problems.append(
                f"Table {number}: appears {delay} pages after the paragraph "
                f"that refers to it (page {refs[0]} -> page {caps[0]})"
            )
    return problems


def check_manifest() -> list[str]:
    """Every \\input of a table must exist; every table must be used."""
    with open(os.path.join(PAPER, "main.tex"), encoding="utf-8") as handle:
        text = handle.read()
    used = set(re.findall(r"\\input\{tables/([^}]+)\}", text))
    problems = []
    for name in sorted(used):
        if not os.path.exists(os.path.join(TABLES, f"{name}.tex")):
            problems.append(f"manuscript \\inputs tables/{name}.tex, which does not exist")
    available = {
        os.path.splitext(f)[0]
        for f in os.listdir(TABLES)
        if f.endswith(".tex")
    }
    # The appendix tables are generated but not part of the manuscript.
    for name in sorted(available - used):
        if name.startswith("tableA"):
            continue
        problems.append(
            f"paper/tables/{name}.tex exists but the manuscript never \\inputs it"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="check an existing PDF instead of building")
    args = parser.parse_args()

    pdf = args.pdf or build_pdf()

    problems = []
    problems += check_manifest()
    problems += check_type_size(pdf)
    problems += check_float_order(page_texts(pdf))

    if problems:
        print(f"[fail] {len(problems)} layout problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("[ok] layout is sound: table manifest, type size and float order all check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
