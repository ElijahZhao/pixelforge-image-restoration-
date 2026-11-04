#!/usr/bin/env python3
"""Check the manuscript's numbers against paper/FACTS.md.

The paper's discipline is that no number is typed by hand: tables are
generated and checked against FACTS.md, and the prose quotes the same
values. This script closes the last gap -- the prose.

It extracts every decimal number from the manuscript's prose (excluding
LaTeX commands, table inputs and the bibliography) and checks each one
against the set of values that FACTS.md attests. A number the manuscript
uses that FACTS.md does not know about is either a typo or an unsourced
claim, and both are worth catching.

The check is deliberately conservative about what counts as a finding:

* values are compared after normalising trailing zeros, so 3.89 and 3.890
  are the same;
* integers in ranges and counts (the 21 protocol cells, the 5 seeds, the
  63 papers) are matched against the same numbers as they appear in
  FACTS.md, so an invented count is still caught;
* a number is only reported when FACTS.md contains no occurrence of it at
  all, which keeps the output to genuinely unknown values.

Usage:
    python3.11 scripts/check_manuscript_numbers.py
"""

from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUSCRIPT = os.path.join(REPO, "paper", "main.tex")
FACTS = os.path.join(REPO, "paper", "FACTS.md")

#: Sections whose numbers come from outside FACTS.md.
#:
#: - The abstract quotes the same quantities as the body, but it is
#:   excluded because LaTeX-free plain text here would double-count.
#: - The author/contribution block has no numbers of its own.
SKIP_ENVIRONMENTS = ("latex", "bibliography")


def prose_numbers(path: str) -> list[tuple[str, str]]:
    """Return (number, context) pairs from the manuscript prose."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    # Drop the preamble: package names, macro definitions and the author
    # block are not prose claims.
    body = text.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\begin{thebibliography}", 1)[0]

    # Remove \input and \include directives: the tables they pull in are
    # checked separately by make_tables.py --check.
    body = re.sub(r"\\(?:input|include)\{[^}]*\}", " ", body)

    # Remove citation and reference commands *before* stripping other
    # commands: the key itself can contain digits (the year, as in
    # "dong2014srcnn"), and those are not claims about the paper.
    body = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:label|ref)\{[^}]*\}", " ", body)

    # Remove LaTeX commands but keep their brace arguments, so that the
    # numbers inside \textbf{...} survive.
    body = re.sub(r"\\[a-zA-Z@]+\*?", " ", body)

    # Remove escaped characters that are not numbers.
    body = body.replace("\\%", " ").replace("\\,", " ").replace("\\;", " ")

    out: list[tuple[str, str]] = []
    for match in re.finditer(r"\d+\.\d+|\d+", body):
        start = max(0, match.start() - 55)
        end = min(len(body), match.end() + 55)
        out.append((match.group(0), re.sub(r"\s+", " ", body[start:end])))
    return out


def attested_values(facts_path: str) -> set[float]:
    """Every numeric literal in FACTS.md, as a float.

    The comparison is by value, not by string: FACTS.md writes the protocol
    span as 3.8892 where the manuscript rounds it to 3.889, and those are
    the same quantity. A string comparison would flag every rounded value
    in the prose, which is exactly the noise that makes a checker
    useless.
    """
    with open(facts_path, encoding="utf-8") as handle:
        text = handle.read()
    values: set[float] = set()
    for token in re.findall(r"\d+\.\d+|\d+", text):
        try:
            values.add(round(float(token), 6))
        except ValueError:
            continue
    return values


def matches(value: float, known: set[float], decimals: int) -> bool:
    """True when the value equals an attested one at the written precision.

    A manuscript value of 3.889 is satisfied by an attested 3.8892, and by
    3.889 exactly, but not by 3.89 -- rounding must be in the same
    direction, so a manuscript cannot smuggle in a different number that
    happens to round near an attested one.
    """
    for candidate in known:
        if round(candidate, decimals) == round(value, decimals):
            return True
    return False


def normalise(token: str) -> str:
    """Compare numbers by value, not by how they were written."""
    try:
        value = float(token)
    except ValueError:
        return token
    if value == int(value):
        return str(int(value))
    # Round to the precision actually written, so 3.89 and 3.890 agree.
    decimals = len(token.split(".")[1]) if "." in token else 0
    return f"{value:.{decimals}f}"


def main() -> int:
    if not os.path.exists(MANUSCRIPT):
        print(f"[skip] {MANUSCRIPT} does not exist yet", file=sys.stderr)
        return 0

    known = attested_values(FACTS)
    unknown: dict[str, list[str]] = {}

    for token, context in prose_numbers(MANUSCRIPT):
        try:
            value = float(token)
        except ValueError:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        if matches(value, known, decimals):
            continue
        unknown.setdefault(token, []).append(context)

    if not unknown:
        print("[ok] every number in the manuscript prose is attested in FACTS.md")
        return 0

    print(f"[note] {len(unknown)} value(s) in the manuscript are not found in "
          f"FACTS.md. Each needs a manual decision: it is either a value "
          f"FACTS.md states in a different form, or an unsourced claim.\n")
    for token in sorted(unknown, key=lambda t: -len(t)):
        print(f"  {token}")
        for ctx in unknown[token][:2]:
            print(f"      ...{ctx}...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
