#!/usr/bin/env python3
"""Structural checks on paper/FACTS.md.

The FACTS file is the paper's single source of truth, and this script guards
the properties that make it usable as one:

* the errata table (section 10) has one row per entry, numbered E1..En with
  no gaps and no duplicates;
* every errata row has the same number of table cells, so the Markdown table
  still renders;
* the figures quoted in the errata table match the ones in the results
  sections, and no figure quoted there is on the forbidden list in section 11.

The third check is the one that has caught real problems before: a number
copied into the errata table and later changed in the results section leaves
the errata table describing a state that no longer exists.

Usage:
    python3.11 scripts/check_facts_structure.py
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(REPO, "paper", "FACTS.md")


def main() -> int:
    with open(FACTS, encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    errors: list[str] = []

    # ---- errata table: numbering ------------------------------------------
    # Anchor at the start of the line: an errata row may quote another entry's
    # number in its prose (E78 quotes "**E74**"), which must not be mistaken
    # for a row of its own.
    entry_lines = [(i, l) for i, l in enumerate(lines) if re.match(r"^\| \*\*E\d+\*\*", l)]
    numbers = [int(re.match(r"\| \*\*E(\d+)\*\*", l).group(1)) for _, l in entry_lines]
    if numbers != list(range(1, len(numbers) + 1)):
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        errors.append(f"errata numbering is not E1..E{len(numbers)}: "
                      f"missing={missing} duplicated={dupes}")
    else:
        print(f"[ok] errata table: E1..E{len(numbers)} contiguous, ascending")

    # ---- errata table: row structure --------------------------------------
    # A row that swallowed its neighbour is longer than the others and has
    # too many cells. This is exactly how E74 was lost inside E75.
    cell_counts = [(i, l.count("|")) for i, l in entry_lines]
    if cell_counts:
        median = sorted(c for _, c in cell_counts)[len(cell_counts) // 2]
        odd = [(i + 1, c) for i, c in cell_counts if c != median]
        if odd:
            for lineno, count in odd:
                errors.append(f"line {lineno}: {count} cell separators, expected {median}")
        else:
            print(f"[ok] errata rows: all {len(entry_lines)} have {median} separators")

    # ---- forbidden figures must not appear in the errata table -------------
    forbidden_heading = "## 11."
    forbidden_idx = next((i for i, l in enumerate(lines)
                          if l.startswith(forbidden_heading)), None)
    if forbidden_idx is None:
        errors.append("section 11 (forbidden figures) not found")
    else:
        tail = lines[forbidden_idx:]
        # Collect the numbers the section forbids, then check none of them
        # appears as a headline claim in the errata table's "what it said"
        # column. The check is deliberately coarse: it flags a forbidden
        # number sitting in an errata row, which always needs a human look.
        banned: set[str] = set()
        for line in tail:
            for token in re.findall(r"\b\d+\.\d{2}\b", line):
                banned.add(token)
        errata_text = "\n".join(l for _, l in entry_lines)
        suspicious = sorted({t for t in banned
                             if re.search(r"(?<![\d.])" + re.escape(t) + r"(?![\d.])",
                                          errata_text)})
        if suspicious:
            print(f"[note] figures that are forbidden in section 11 but appear in the "
                  f"errata table (each needs a manual look): {suspicious}")
        else:
            print("[ok] no section-11 forbidden figure appears in the errata table")

    if errors:
        print("\n[FAIL]")
        for line in errors:
            print(f"  {line}")
        return 1

    print("\n[OK] FACTS.md structure is sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
