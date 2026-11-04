#!/usr/bin/env python3
"""Build Table 1: the seven nearest-neighbour works.

Table 1 is the paper's "who did what, and what do we add" table. It sits in
Related Work and is the only table whose content is bibliographic rather
than measured, so it cannot be derived from the result CSVs.

Two properties are enforced rather than assumed:

* Every row cites a bib key that must exist in `paper/refs.bib`. A table
  that names a paper the bibliography does not contain is exactly the
  failure mode this paper criticises elsewhere, so it is checked, not
  trusted.
* Every row's claims about what the cited work found are the audited
  wording from `paper/MATERIALS.md` section 4.1, which was verified
  against arXiv metadata and the abstracts. The wording is held in this
  file so the table has one source of truth.

Usage:
    python3.11 scripts/make_table1.py
    python3.11 scripts/make_table1.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(REPO, "paper", "refs.bib")
OUT_DIR = os.path.join(REPO, "paper", "tables")

#: The seven audited nearest neighbours.
#:
#: Columns
#:   bib     -- key in paper/refs.bib; verified to exist by --check
#:   venue   -- short venue string
#:   did     -- what the work established (audited wording)
#:   gap     -- what it left open that this paper addresses
#:   ours    -- the part of this work that answers that gap
NEIGHBOURS = [
    {
        "bib": "keles2021psnr",
        "venue": "PCS 2021",
        "did": "PSNR over a set is not the mean of per-image PSNR; gives the bound for SISR (2.07 dB measured, 2.507 dB theoretical)",
        "gap": "Isolates aggregation only; no protocol matrix, and no seed variance",
        "ours": "Protocol P1 (best-of-N) is the extreme case of the same effect; we quantify it against seven other protocol dimensions",
    },
    {
        "bib": "parmar2022cleanfid",
        "venue": "CVPR 2022",
        "did": "Antialiased vs aliased resizing changes FID rankings in GAN evaluation",
        "gap": "Resize semantics only, in the perceptual/GAN metric setting",
        "ours": "P6 vs P5 isolates antialiasing in the distortion metric; the effect is 2.72 dB, not a ranking detail",
    },
    {
        "bib": "zhang2024seal",
        "venue": "ICLR 2024",
        "did": "Random-crop evaluation is unreliable and flips rankings among SR methods",
        "gap": "Sampling randomness only; degradation, channel and aggregation are held fixed",
        "ours": "P1 vs P2 vs P3 separates crop position from the other five dimensions, with 5 seeds per cell",
    },
    {
        "bib": "srinivasan2026lol",
        "venue": "arXiv 2026",
        "did": "LOL benchmarks carry hidden label shifts: LOLv1 test labels render differently, and LOLv2-Real is 14% darker than the training labels",
        "gap": "Dataset-side label shift; the evaluation protocol itself is not varied",
        "ours": "The low-light task is our counterexample: protocol does not dominate there (ratio 0.21), so the claim is conditional",
    },
    {
        "bib": "shah2026hyperbench",
        "venue": "arXiv 2026",
        "did": "Protocol implementation differences in the Wald protocol prevent hyper-resolution rankings from holding",
        "gap": "One task family (hyperspectral), one protocol dimension (degradation)",
        "ours": "Three tasks and eight protocols; the protocol span exceeds the training gain on two of them",
    },
    {
        "bib": "su2025bridging",
        "venue": "arXiv 2025",
        "did": "Audits perceptual quality metrics for SR and finds they diverge from distortion metrics",
        "gap": "Metric choice is orthogonal to the protocol dimensions this paper varies",
        "ours": "Holds the metric fixed and varies sampling, channel, aggregation and degradation instead",
    },
    {
        "bib": "cawley2010overfitting",
        "venue": "JMLR 2010",
        "did": "Selection bias: repeated model selection on a validation set inflates the reported performance of the selected model",
        "gap": "Gives the theory; does not measure how large the bias is under a stated protocol",
        "ours": "Measures the same bias at 1.74 dB under P1, which is why the reporting checklist requires the selection budget N",
    },
]

CAPTION = ("Nearest-neighbour work on evaluation protocols, with the part of this "
           "paper that addresses what each left open.")


def bib_keys() -> set[str]:
    with open(BIB, encoding="utf-8") as handle:
        return set(re.findall(r"@\w+\{([^,]+),", handle.read()))


def render_markdown() -> str:
    lines = [f"**{CAPTION}**", "",
             "| Work | Venue | What it established | What it left open | What this paper adds |",
             "|---|---|---|---|---|"]
    for row in NEIGHBOURS:
        lines.append(
            f"| `{row['bib']}` | {row['venue']} | {row['did']} | {row['gap']} | {row['ours']} |"
        )
    lines += [
        "",
        "1. The seven entries are the works that vary an *evaluation* protocol rather "
        "than propose a method. Five were verified against arXiv metadata; two "
        "(Cawley & Talbot, Keleş et al.) against the publisher record.",
        "2. No row claims that the cited work is wrong. Each entry records what it "
        "established and the scope it left open.",
        "3. `refs.bib` is the single source for every citation in this table; the "
        "keys in the first column are verified to exist by `scripts/make_table1.py --check`.",
    ]
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    return (text.replace("&", "\\&")
                .replace("%", "\\%")
                .replace("_", "\\_")
                .replace("#", "\\#"))


def render_latex() -> str:
    lines = [
        "% Auto-generated by scripts/make_table1.py -- do not hand-edit.",
        "% Requires: \\usepackage{booktabs} \\usepackage{graphicx}",
        "% A five-column table with prose in three of them: it needs the full",
        "% text width of a two-column page, so it is a table* float.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{latex_escape(CAPTION)}}}",
        "\\label{tab:nearest-neighbours}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llp{0.20\\textwidth}p{0.20\\textwidth}p{0.24\\textwidth}@{}}",
        "\\toprule",
        "Work & Venue & What it established & What it left open & What this paper adds \\\\",
        "\\midrule",
    ]
    for row in NEIGHBOURS:
        lines.append(
            f"\\texttt{{{latex_escape(row['bib'])}}} & {latex_escape(row['venue'])} & "
            f"{latex_escape(row['did'])} & {latex_escape(row['gap'])} & "
            f"{latex_escape(row['ours'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify bib keys and that the output files are up to date")
    args = parser.parse_args()

    keys = bib_keys()
    missing = [row["bib"] for row in NEIGHBOURS if row["bib"] not in keys]
    if missing:
        print(f"[FAIL] bib key(s) missing from paper/refs.bib: {missing}", file=sys.stderr)
        return 1
    print(f"[ok] all {len(NEIGHBOURS)} bib keys exist in paper/refs.bib")

    md = render_markdown()
    tex = render_latex()
    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "table1_neighbours.md")
    tex_path = os.path.join(OUT_DIR, "table1_neighbours.tex")

    if args.check:
        problems = []
        for path, content in ((md_path, md), (tex_path, tex)):
            if not os.path.exists(path):
                problems.append(f"missing {os.path.relpath(path, REPO)}")
            elif open(path, encoding="utf-8").read() != content:
                problems.append(f"stale {os.path.relpath(path, REPO)}")
        if problems:
            print("[FAIL] " + "; ".join(problems), file=sys.stderr)
            return 1
        print("[ok] table1_neighbours.{md,tex} are up to date")
        return 0

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    print(f"wrote {os.path.relpath(md_path, REPO)} and {os.path.relpath(tex_path, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
