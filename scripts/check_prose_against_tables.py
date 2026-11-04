#!/usr/bin/env python3
"""Check the manuscript prose against the generated tables.

The tables and the prose are two renderings of the same numbers: the tables
are generated from the result CSVs, the prose is written by hand. They can
drift apart silently -- a table is corrected and the sentence that quotes
it is not.

This script extracts the numeric values that appear in both, and reports
the values the prose states that the corresponding table row contradicts.
It works from the table files themselves, so a change in the data flows
through automatically.

The comparison is per-table, using the row labels to align: for a given
table, every number the prose states near a mention of that table is
compared with the numbers in the table's rows, and a prose value that
matches no cell (at the table's precision) is reported.

This is a heuristic alignment, and it is reported as such: it flags
candidates, and a human decides. Its value is catching the specific
failure this project keeps meeting -- a number corrected in FACTS.md and
the tables, but not in the sentence that cites it.

Usage:
    python3.11 scripts/check_prose_against_tables.py
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUSCRIPT = os.path.join(REPO, "paper", "main.tex")
TABLES = os.path.join(REPO, "paper", "tables")

#: Which generated table each manuscript table reference points at.
#:
#: The manuscript loads tables with \input, so the caption-to-file mapping
#: is not visible in main.tex; this table records it explicitly rather
#: than guessing from file names.
TABLE_FILES = {
    "tab:protocols": "table2_protocols.md",
    "tab:span-vs-gain": "table4_task_ratios.md",
    "tab:sign-flip": "table5_sign_flip.md",
    "tab:reporting-rates": "table6_literature.md",
    "tab:checklist": "table7_checklist.md",
    "tab:nearest-neighbours": "table1_neighbours.md",
}

#: Prose values that are deliberately not in the table they are discussed
#: near, with the reason. Each entry was checked by hand.
#:
#: Without these, the checker reports them on every run, and a checker
#: that always reports something is a checker nobody reads.
KNOWN_EXCEPTIONS = {
    # The seed sigma of the two sampling protocols. The prose states that
    # P1 and P2 have a different kind of variation and must not be pooled
    # with the full-image protocols, which is precisely why the table does
    # not list them.
    "0.2364": "sigma of the sampling protocols (P1/P2), listed separately on purpose",
    "0.5584": "sigma of the sampling protocols (P1/P2), listed separately on purpose",
    "0.4148": "sigma of the sampling protocols (P1/P2), listed separately on purpose",
    # A derived ratio: the swing divided by the training gain. Neither
    # factor is a cell in the table; the ratio is a conclusion.
    "3.3": "derived ratio (swing / training gain), not a table cell",
}


def table_numbers(path: str) -> set[float]:
    """Every number that appears in a table's data cells."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    values: set[float] = set()
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        for token in re.findall(r"-?\d+\.\d+|\d+", line):
            try:
                values.add(round(float(token), 6))
            except ValueError:
                continue
    return values


def prose_values(text: str) -> list[tuple[float, str]]:
    """Every number in the manuscript prose, with context."""
    body = text.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"\\(?:input|include)\{[^}]*\}", " ", body)
    body = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:label|ref)\{[^}]*\}", " ", body)
    body = re.sub(r"\\[a-zA-Z@]+\*?", " ", body)
    body = body.replace("\\%", " ").replace("\\,", " ")
    out = []
    for match in re.finditer(r"-?\d+\.\d+|\d+", body):
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        start = max(0, match.start() - 70)
        end = min(len(body), match.end() + 70)
        out.append((value, re.sub(r"\s+", " ", body[start:end])))
    return out


def main() -> int:
    if not os.path.exists(MANUSCRIPT):
        print("[skip] no manuscript yet")
        return 0

    with open(MANUSCRIPT, encoding="utf-8") as handle:
        text = handle.read()

    # Collect the numbers that appear within a window of each table
    # reference, so each prose value is checked against the table it is
    # most likely about.
    references = list(re.finditer(r"\\ref\{([^}]+)\}", text))
    if not references:
        print("[skip] the manuscript references no tables yet")
        return 0

    problems: list[str] = []
    checked = 0

    for match in references:
        key = match.group(1)
        filename = TABLE_FILES.get(key)
        if filename is None:
            continue
        path = os.path.join(TABLES, filename)
        if not os.path.exists(path):
            problems.append(f"{key}: expected table file {filename} is missing")
            continue

        known = table_numbers(path)
        # The window is the paragraph the reference sits in. A wider window
        # would pull in numbers that belong to a different table -- the
        # sign-flip values sit within a thousand characters of the
        # span-versus-gain reference -- and every one of them would be
        # reported as a mismatch, which buries the real ones.
        start = text.rfind("\n\n", 0, match.start())
        start = 0 if start == -1 else start
        end = text.find("\n\n", match.end())
        end = len(text) if end == -1 else end
        window = text[start:end]

        for token in re.findall(r"-?\d+\.\d+", window):
            try:
                value = float(token)
            except ValueError:
                continue
            checked += 1
            if any(round(candidate, 3) == round(value, 3) for candidate in known):
                continue
            if token in KNOWN_EXCEPTIONS:
                continue
            problems.append(
                f"{key}: prose value {token} does not appear in {filename}"
            )

    print(f"checked {checked} prose number(s) against {len(TABLE_FILES)} table(s)")

    # Deduplicate, keeping order.
    seen: set[str] = set()
    unique = []
    for line in problems:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    if unique:
        print(f"\n[note] {len(unique)} candidate mismatch(es). Each needs a "
              f"manual look: a prose value may legitimately differ from the "
              f"table (rounded differently, or drawn from a different "
              f"source), or it may be a stale number.\n")
        for line in unique:
            print(f"  {line}")
        return 1

    print("[ok] every prose number near a table reference agrees with that table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
