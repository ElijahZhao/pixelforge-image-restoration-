#!/usr/bin/env python3
"""Build the Preprints.org submission archive.

The official Instructions for Authors state:

    Files should be submitted in Microsoft Word or LaTeX format. For LaTeX
    files, ensure that all files necessary, including the .bib file, if
    applicable, to recreate the PDF are included in a zip or similar
    archive.

So the archive has to contain everything needed to rebuild the PDF, and
nothing that would make the rebuild fail. This script assembles it and then
*proves* the claim: it unpacks the archive into a temporary directory and
compiles the manuscript there, from scratch, with no access to the rest of
the repository. If that works, the archive is sufficient.

Usage:
    python3.11 scripts/make_submission_package.py
    python3.11 scripts/make_submission_package.py --check   # rebuild and compare
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
OUT = os.path.join(REPO, "dist", "preprints_submission.zip")

#: Files copied into the archive, relative to paper/.
TEX_FILES = ["main.tex", "refs.bib"]

#: Figures the manuscript includes. Listed explicitly so that a stale
#: figure in the archive cannot be shipped silently: the list is the
#: contract, and the script checks each file exists.
FIGURE_FILES = [
    "figures/fig1_protocol_matrix.pdf",
    "figures/fig2_sampling_noise.pdf",
]

#: Tables pulled in with \input. Their .tex twins are needed; the .md
#: versions are not.
#:
#: The list is derived from the manuscript rather than maintained by hand:
#: a table that the manuscript \inputs but the archive omits fails to
#: compile; a table the archive carries but the manuscript never uses is
#: dead weight that suggests the two have drifted apart. The verification
#: step below checks both directions.
TABLE_FILES = [
    "tables/table1_neighbours.tex",
    "tables/table2_protocols.tex",
    "tables/table3_setup.tex",
    "tables/table4_task_ratios.tex",
    "tables/table5_sign_flip.tex",
    "tables/table6_literature.tex",
    "tables/table7_checklist.tex",
]

README = """\
Preprints.org submission archive
================================

This archive contains the LaTeX source of

    The Protocol Is the Result: How Evaluation Choices Flip the Verdict on
    Lightweight Image Restoration

and everything needed to rebuild it.

Contents
--------
  main.tex                     the manuscript
  refs.bib                     27 references (BibTeX)
  figures/                     the two figures included by the manuscript
  tables/                      the tables included by \\input
  README.md                    this file

How to build
------------
    pdflatex main
    bibtex main
    pdflatex main
    pdflatex main

(The manuscript is set two-column; the PDF is main.pdf.)

Notes
-----
* The graphical abstract is submitted separately, as required by the
  platform's instructions: it must be a JPG or PNG and may not be a
  screenshot of the textual abstract.
* The per-image values behind every reported protocol cell are released
  in the accompanying repository under results/per_image/, with the
  aggregated matrix and the scripts that regenerate the tables.
"""


def check_table_manifest(main_tex: str) -> list[str]:
    """Report mismatches between the manuscript's \\input and the manifest.

    Two failure modes matter, and both are invisible in a normal build:

    * the manuscript \\inputs a table the archive does not carry -- the
      archive then fails to compile, but only after submission;
    * the archive carries a table the manuscript never \\inputs -- a sign
      that the two lists have drifted apart, which is exactly the class of
      failure this paper is about.

    Returns a list of human-readable problems; empty means the lists agree.
    """
    with open(main_tex, encoding="utf-8") as handle:
        text = handle.read()
    used = {
        name.strip()
        for name in re.findall(r"\\input\{tables/([^}]+)\}", text)
    }
    listed = {
        os.path.splitext(os.path.basename(name))[0]
        for name in TABLE_FILES
    }
    problems = []
    for name in sorted(used - listed):
        problems.append(f"manuscript \\inputs tables/{name}.tex, which the archive omits")
    for name in sorted(listed - used):
        problems.append(f"archive carries tables/{name}.tex, which the manuscript never \\inputs")
    return problems


def collect() -> list[tuple[str, str]]:
    """Return (source path, path inside the archive) pairs."""
    items: list[tuple[str, str]] = []
    problems = check_table_manifest(os.path.join(PAPER, "main.tex"))
    if problems:
        raise SystemExit("[fail] table manifest disagrees with the manuscript:\n  "
                         + "\n  ".join(problems))
    for name in TEX_FILES + TABLE_FILES + FIGURE_FILES:
        src = os.path.join(PAPER, name)
        if not os.path.exists(src):
            raise SystemExit(f"[fail] required file missing: paper/{name}")
        items.append((src, name))
    return items


def build_archive(target: str) -> None:
    os.makedirs(os.path.dirname(target), exist_ok=True)
    items = collect()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            zf.write(src, arc)
        zf.writestr("README.md", README)


def verify_archive(target: str) -> None:
    """Unpack and compile the archive in isolation.

    This is the only way to know the archive is sufficient: any file it
    misses shows up here as a compile error, not as a surprise after
    submission.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(target) as zf:
            zf.extractall(tmp)

        for step in (
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ["bibtex", "main"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ):
            proc = subprocess.run(step, cwd=tmp, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = "\n".join(proc.stdout.splitlines()[-25:])
                raise SystemExit(f"[fail] {step[0]} failed in the unpacked archive:\n{tail}")

        log = open(os.path.join(tmp, "main.log"), encoding="utf-8", errors="replace").read()
        problems = []
        if any(line.startswith("! ") for line in log.splitlines()):
            problems.append("LaTeX errors are present in main.log")
        if "undefined" in log.lower():
            problems.append("unresolved references in main.log")
        if "Overfull" in log:
            problems.append("overfull boxes in main.log")
        pdf = os.path.join(tmp, "main.pdf")
        if not os.path.exists(pdf):
            problems.append("main.pdf was not produced")
        if problems:
            raise SystemExit("[fail] " + "; ".join(problems))

        pages = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
        for line in pages.stdout.splitlines():
            if line.startswith("Pages:"):
                print(f"  archive builds cleanly: main.pdf ({line.strip()})")
                break


def digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="rebuild and verify the archive, then compare")
    args = parser.parse_args()

    if args.check and os.path.exists(OUT):
        before = digest(OUT)

    build_archive(OUT)
    size = os.path.getsize(OUT)
    print(f"built {os.path.relpath(OUT, REPO)} ({size / 1024:.0f} KiB)")

    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
    print(f"  {len(names)} entries: {', '.join(sorted(names))}")

    verify_archive(OUT)

    if args.check and os.path.exists(OUT):
        after = digest(OUT)
        print(f"  sha256 {after[:16]}...")
        if before != after:
            print("  [warn] archive contents changed since the last build")

    return 0


if __name__ == "__main__":
    sys.exit(main())
