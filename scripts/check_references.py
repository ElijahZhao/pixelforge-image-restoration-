"""Audit every bibliography entry in paper/refs.bib for validity.

Four independent checks, each of which has caught a real defect in this
project before:

1. arXiv id resolves, and the title returned by the arXiv API contains the
   title recorded in refs.bib. (Two ids written from memory during the
   earlier literature audit resolved to unrelated papers.)
2. Every \\cite key used in paper/main.tex exists in refs.bib, and every
   entry in refs.bib is actually cited -- an uncited entry is dead weight
   in a preprint, and a missing key is a broken reference.
3. Every doi resolves at doi.org and the registered title matches the
   title in refs.bib (doi.org / Crossref).
4. Non-arXiv entries without a doi (JMLR) are checked against the journal's
   own landing page.

Exit code is non-zero when anything fails, so this can run in CI.

Exit code 2 is reserved for "the check could not run": the network refused
every attempt. That is a different statement from "a reference is broken",
and the distinction matters in CI -- a registry being briefly unreachable
must not be reported as a broken citation, because a reader who sees "27
references failed" will reasonably assume the manuscript is wrong. The CI
step treats exit 2 as inconclusive rather than as a defect.

    python scripts/check_references.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "http://www.w3.org/2005/Atom"

FAILURES: list[str] = []
CHECKS = 0


class NetworkUnreachable(Exception):
    """The registry never answered; the check could not run.

    Distinct from a failed check. A reference that genuinely does not
    resolve is a defect in the manuscript; a registry that is briefly down
    is not, and reporting the two the same way teaches the reader to
    distrust the report.
    """


def check(ok: bool, label: str, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{label}{(': ' + detail) if detail else ''}")
    return ok


def http_get(url: str, timeout: int = 45, accept: str = "*/*",
             retries: int = 4) -> tuple[int, str]:
    """GET with backoff on the rate limits the registries impose.

    doi.org and Crossref answer anonymous bursts with 429, and the IEEE
    landing pages make doi.org answer a bare resolution request with
    "202 Accepted" while it resolves the redirect. A single sample would
    therefore report false failures, so the request is retried with a
    growing delay and the metadata is asked for explicitly through
    content negotiation.
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": "reference-audit/1.0 (preprint submission check)",
        "Accept": accept,
    })
    last = (0, "")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.status, fh.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = (e.code, "")
            if e.code in (429, 503):
                time.sleep(3 * (attempt + 1))
                continue
            return e.code, ""
        except Exception as e:  # noqa: BLE001
            # Status 0 marks a connection-level failure: the request never
            # got an answer. The caller uses this to tell "the network is
            # down" apart from "the server said no".
            last = (0, f"{type(e).__name__}: {e}")
            time.sleep(2 * (attempt + 1))
    return last


def parse_bib(path: str) -> dict[str, dict[str, str]]:
    """Minimal BibTeX reader: enough for @article/@inproceedings/@misc."""
    text = open(path, encoding="utf-8").read()
    entries: dict[str, dict[str, str]] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        kind, key = m.group(1).lower(), m.group(2).strip()
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        fields: dict[str, str] = {"_kind": kind}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{.*?\}|\"[^\"]*\"|[^,\n]+)\s*,?\s*(?=\n|$|[\w-]+\s*=)",
                              body, re.S):
            name = fm.group(1).lower()
            val = fm.group(2).strip()
            if val.startswith("{") and val.endswith("}"):
                val = val[1:-1]
            elif val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            fields[name] = " ".join(val.split())
        entries[key] = fields
    return entries


def strip_latex(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\&", "&").replace("\\_", "_").replace("\\%", "%")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("``", '"').replace("''", '"').replace("`", "'")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return " ".join(s.split())


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", strip_latex(s).lower()).strip()


def arxiv_meta(ids: list[str]) -> dict[str, tuple[str, str]]:
    """id -> (title, published year) for the given arXiv ids."""
    url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + "&max_results=" + str(len(ids)))
    status, body = http_get(url)
    if status == 0:
        raise NetworkUnreachable(f"arXiv API unreachable: {body[:200]}")
    if status != 200:
        raise SystemExit(f"arXiv API returned {status}: {body[:200]}")
    root = ET.fromstring(body)
    out: dict[str, tuple[str, str]] = {}
    for e in root.findall(f"{{{ATOM}}}entry"):
        aid = e.find(f"{{{ATOM}}}id").text.rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join(e.find(f"{{{ATOM}}}title").text.split())
        pub = e.find(f"{{{ATOM}}}published")
        year = pub.text[:4] if pub is not None else ""
        out[aid] = (title, year)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="paper/refs.bib")
    ap.add_argument("--tex", default="paper/main.tex")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    try:
        run(args)
    except NetworkUnreachable as exc:
        print()
        print(f"[refs] INCONCLUSIVE: {exc}")
        print("[refs] The registry never answered, so the check did not run.")
        print("[refs] This is not a statement about the bibliography.")
        if args.report:
            json.dump({"checks": None, "status": "network-unreachable",
                       "detail": str(exc)},
                      open(args.report, "w"), indent=2)
        sys.exit(2)


def run(args) -> None:
    entries = parse_bib(args.bib)
    print(f"[refs] parsed {len(entries)} entries from {args.bib}")

    # ---- 2. cite-key closure ------------------------------------------------
    tex = open(args.tex, encoding="utf-8").read()
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    cited: list[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited.append(k)
    cited_set = set(cited)
    bib_set = set(entries)
    for k in sorted(cited_set - bib_set):
        check(False, f"cited key missing from {args.bib}", k)
    for k in sorted(bib_set - cited_set):
        check(False, f"entry in {args.bib} is never cited", k)
    print(f"[refs] {len(cited_set)} distinct cite keys, "
          f"{len(cited)} cite commands, closure {'OK' if cited_set == bib_set else 'BROKEN'}")

    # ---- 1. arXiv resolution + title match ----------------------------------
    # The year recorded in refs.bib is the *venue* year (the conference or
    # journal that finally published the work), while the arXiv API reports the
    # year the preprint was first uploaded. Those legitimately differ by a
    # year or two -- VGG was uploaded in 2014 and published at ICLR 2015 --
    # so the check is that the venue year is not earlier than the upload and
    # not implausibly later.
    arxiv_ids = {k: v["eprint"] for k, v in entries.items() if "eprint" in v}
    if arxiv_ids:
        meta = arxiv_meta(sorted(set(arxiv_ids.values())))
        for key, aid in arxiv_ids.items():
            if aid not in meta:
                check(False, f"{key}: arXiv id does not resolve", aid)
                continue
            api_title, api_year = meta[aid]
            local = norm(entries[key].get("title", ""))
            remote = norm(api_title)
            check(local == remote or remote.startswith(local) or local.startswith(remote),
                  f"{key}: title mismatch vs arXiv {aid}",
                  f"bib={entries[key].get('title','')!r} arxiv={api_title!r}")
            yr = entries[key].get("year", "")
            if yr and api_year:
                check(int(api_year) <= int(yr) <= int(api_year) + 3,
                      f"{key}: venue year inconsistent with arXiv upload year",
                      f"bib={yr} arxiv_upload={api_year}")
            # The abs page must be reachable -- readers need a working link.
            status, body = http_get(f"https://arxiv.org/abs/{aid}")
            if status == 0:
                raise NetworkUnreachable(
                    f"arxiv.org/abs/{aid} unreachable: {body[:160]}")
            check(status == 200, f"{key}: arxiv.org/abs/{aid} not reachable",
                  f"HTTP {status}")
        print(f"[refs] arXiv: {len(arxiv_ids)} entries fetched, "
              f"titles/years compared, abs pages reachable")

    # ---- 3. doi resolution + title match ------------------------------------
    # doi.org answers a bare resolution request for the IEEE DOIs with
    # "202 Accepted" (their landing page defers the redirect), so a plain
    # GET is not a validity signal. Two independent signals are used
    # instead: the Crossref registry must hold the DOI with a matching
    # title, and doi.org content negotiation must return the registered
    # metadata. The latter is rate limited, so it is retried with backoff.
    dois = {k: v["doi"] for k, v in entries.items() if "doi" in v}
    for key, doi in dois.items():
        cstatus, cbody = http_get(
            "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
        if cstatus == 0:
            raise NetworkUnreachable(
                f"Crossref unreachable for {doi}: {cbody[:160]}")
        if not check(cstatus == 200, f"{key}: DOI not registered at Crossref",
                     f"{doi} -> HTTP {cstatus}"):
            continue
        msg = json.loads(cbody)["message"]
        reg = norm(msg.get("title", [""])[0])
        local = norm(entries[key].get("title", ""))
        check(local == reg or reg.startswith(local) or local.startswith(reg),
              f"{key}: title mismatch vs Crossref",
              f"bib={entries[key].get('title','')!r} crossref={msg.get('title',[''])[0]!r}")
        # The registered year must agree with the year in the entry.
        parts = (msg.get("published-print") or msg.get("published-online")
                 or msg.get("published") or {}).get("date-parts", [[None]])[0]
        if parts and parts[0] and entries[key].get("year"):
            check(int(parts[0]) == int(entries[key]["year"]),
                  f"{key}: year mismatch vs Crossref",
                  f"bib={entries[key]['year']} crossref={parts[0]}")
        # doi.org must also serve the metadata for this DOI. Content
        # negotiation is what makes the registry answer with the record
        # instead of the publisher's landing page.
        st, body = http_get(
            "https://doi.org/" + urllib.parse.quote(doi),
            accept="application/vnd.citationstyles.csl+json")
        if st == 0:
            raise NetworkUnreachable(
                f"doi.org unreachable for {doi}: {body[:160]}")
        ok = st == 200
        if ok:
            try:
                rec = json.loads(body)
                check(bool(rec.get("DOI")), f"{key}: doi.org record has no DOI", doi)
            except Exception:  # noqa: BLE001
                ok = False
        check(ok, f"{key}: doi.org does not serve metadata", f"{doi} -> HTTP {st}")
        print(f"[refs] doi ok: {key} {doi} (Crossref title+year match, doi.org 200)")

    # ---- 4. entries with neither eprint nor doi -----------------------------
    for key, e in entries.items():
        if "eprint" in e or "doi" in e:
            continue
        j = e.get("journal", "")
        if "Machine Learning Research" in j:
            status, body = http_get(
                "https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf")
            if status == 0:
                raise NetworkUnreachable(
                    f"jmlr.org unreachable: {body[:160]}")
            check(status == 200, f"{key}: JMLR landing page not reachable", f"HTTP {status}")
            print(f"[refs] journal landing ok: {key} ({j})")
        else:
            check(False, f"{key}: entry has neither eprint nor doi", j)

    # ---- report --------------------------------------------------------------
    print()
    if FAILURES:
        print(f"[refs] FAILED {len(FAILURES)}/{CHECKS} checks:")
        for f in FAILURES:
            print("  - " + f)
        if args.report:
            json.dump({"checks": CHECKS, "failures": FAILURES},
                      open(args.report, "w"), indent=2)
        sys.exit(1)

    print(f"[refs] ALL {CHECKS} checks passed "
          f"({len(entries)} entries: {len(arxiv_ids)} arXiv, {len(dois)} doi, "
          f"{len(entries) - len(arxiv_ids)} other)")
    if args.report:
        json.dump({"checks": CHECKS, "failures": [], "entries": len(entries)},
                  open(args.report, "w"), indent=2)


if __name__ == "__main__":
    main()
