#!/usr/bin/env python3
"""Verify that per-image CSVs are the true source of the paper's numbers.

The archive (`archive_private/`) holds several generations of evaluation
output: an early `kernel-output-v4` sweep, a failed repository attempt
(`kernel-output-protocol-repo-v1-failed`), superseded protocol repos, and the
runs that finally produced `results/matrix_runs/*/protocol_matrix.json`.

Publishing all of it would publish numbers that do not back the paper. This
script decides, per per-image CSV, whether it agrees with one of the six
authoritative matrix jobs -- by recomputing the per-protocol mean PSNR from
the CSV and comparing it with the `psnr_mean` recorded in the JSON.

A CSV is accepted only if some (job, protocol, seed) triple matches to
within 0.001 dB. The result is the manifest used to build the public data
release, so the mapping is evidence-based rather than name-based.

Usage:
    python3.11 scripts/verify_per_image_mapping.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO, "archive_private")
MATRIX_ROOT = os.path.join(REPO, "results", "matrix_runs")
TOL = 1e-3

#: The archive directories that produced an authoritative matrix job, mapped
#: to the job they produced. Every other archived generation (the failed
#: repository attempt, the superseded sweeps, the older protocol repos) is
#: deliberately absent.
#:
#: The mapping is explicit rather than a substring test, because the archive
#: directory names do not always match the job names: the x2 bicubic
#: baseline lives in `local-baseline-bicubic-sr-x2`, and `lowlight_unet_x2`
#: under `full-v3` is a superseded run that must NOT be confused with the
#: authoritative `lowlight_unet_x1` under `full-v4`.
ARCHIVE_DIR_TO_JOB = {
    "kernel-output-protocol-full-v2/generator_x4_x4": "generator_x4_x4",
    "kernel-output-protocol-full-v2/srcnn_x2_x2": "srcnn_x2_x2",
    "kernel-output-protocol-full-v3/bicubic_sr_x4": "bicubic_sr_x4",
    "kernel-output-protocol-full-v4/identity_ll": "identity_ll",
    "kernel-output-protocol-full-v4/lowlight_unet_x1": "lowlight_unet_x1",
    "local-baseline-bicubic-sr-x2": "bicubic_sr_x2",
}

#: Column names that hold a model-side PSNR. The archived sweep used
#: `psnr_model` / `psnr_bicubic`; the protocol runs used plain `psnr`.
PSNR_COLUMNS = ("psnr", "psnr_model")


def csv_mean_psnr(path: str) -> float | None:
    """Mean PSNR over the rows of a per-image CSV.

    Two subtleties, both learned from the official aggregation path
    (`scripts/aggregate_matrix.py` and `scripts/align_p1_inf.py`):

    * Non-finite values are dropped, not propagated. A bicubic upsampler can
      reconstruct a flat patch exactly, so MSE == 0 and PSNR == +inf. The
      authoritative matrix reports the mean over the 95 *finite* images for
      P1; carrying the infinities into the mean yields `inf` and matches
      nothing.
    * The column name varies between generations of the archive.

    Returns None if the file is unreadable or has no usable PSNR column, so
    a malformed file is reported rather than silently skipped.
    """
    total = 0.0
    count = 0
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            field = next((c for c in PSNR_COLUMNS if c in (reader.fieldnames or [])), None)
            if field is None:
                return None
            for row in reader:
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    continue
                total += value
                count += 1
    except (OSError, ValueError, KeyError):
        return None
    if count == 0:
        return None
    return total / count


def load_matrix_index() -> dict[tuple[str, str], float]:
    """(checkpoint_key, protocol) -> psnr_mean from the authoritative JSONs."""
    index: dict[tuple[str, str], float] = {}
    for json_path in sorted(glob.glob(os.path.join(MATRIX_ROOT, "*", "protocol_matrix.json"))):
        job = os.path.basename(os.path.dirname(json_path))
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        for ckpt, payload in data.get("checkpoints", {}).items():
            for proto, stats in payload.get("protocols", {}).items():
                index[(job, ckpt, proto)] = float(stats["psnr_mean"])
    return index


def parse_cell_from_name(path: str) -> tuple[str, str] | None:
    """Read (checkpoint_key, protocol) out of an archived file name.

    The archived file names encode the cell they belong to:

        sr_generator_scale4_best__s42__P3.csv -> ("sr_generator_scale4_best__s42", "P3")
        bicubic__P1.csv                        -> ("bicubic", "P1")

    Matching on the value alone is not enough. P3 and P4 differ only by a
    4-pixel shave, so their means sit within ~0.01 dB of each other and a
    tolerance-based match alone assigns the same cell to two files. The name
    pins the cell; the value then confirms the file really holds that cell's
    data.
    """
    stem = os.path.basename(path)[:-len(".csv")]
    if "__" not in stem:
        return None
    checkpoint, _, protocol = stem.rpartition("__")
    if not protocol.startswith("P"):
        return None
    return checkpoint, protocol


def job_of(path: str) -> str | None:
    """Infer the matrix job from the archive path.

    Only directories listed in `ARCHIVE_DIR_TO_JOB` produce an authoritative
    `results/matrix_runs/<job>/`; every other archived generation (failed
    repos, superseded sweeps) has no job and therefore no cell to claim.
    """
    normalised = path.replace(os.sep, "/")
    for archive_dir, job in ARCHIVE_DIR_TO_JOB.items():
        if f"/{archive_dir}/" in normalised:
            return job
    return None


def classify(csv_path: str, index: dict[tuple[str, str, str], float]) -> tuple[str, str]:
    """Return (status, detail) for one per-image CSV.

    status is one of:
      match    -- the file's name says cell X and its content reproduces X
      stale    -- parsable, but either the name points at no authoritative
                  cell or the content disagrees with the cell it claims
      broken   -- unreadable or has no usable PSNR column
    """
    mean = csv_mean_psnr(csv_path)
    if mean is None:
        return "broken", "unreadable or no psnr column"

    job = job_of(csv_path)
    parsed = parse_cell_from_name(csv_path)
    if job is None or parsed is None:
        return "stale", f"mean={mean:.4f}; no authoritative job/cell in the name"
    checkpoint, protocol = parsed
    expected = index.get((job, checkpoint, protocol))
    if expected is None:
        return "stale", f"mean={mean:.4f}; {job}/{checkpoint}/{protocol} is not a matrix cell"
    if abs(mean - expected) > TOL:
        return "stale", (f"mean={mean:.4f} != matrix {expected:.4f} "
                         f"for {job}/{checkpoint}/{protocol}")
    return "match", f"{job}/{checkpoint}/{protocol}"


def main() -> int:
    index = load_matrix_index()
    print(f"authoritative matrix cells: {len(index)}")

    candidates = sorted(
        glob.glob(os.path.join(ARCHIVE, "**", "per_image", "*.csv"), recursive=True)
    )
    # The three flat per_image_*.csv files from kernel-output-v4 are the
    # baseline sweep; they are protocol-aggregated differently and are
    # checked separately below.
    flat = sorted(glob.glob(os.path.join(ARCHIVE, "**", "per_image_*.csv"), recursive=True))

    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in candidates:
        status, detail = classify(path, index)
        buckets[status].append((os.path.relpath(path, REPO), detail))

    print(f"per_image/*.csv candidates: {len(candidates)}")
    for status in ("match", "stale", "broken"):
        print(f"  {status:>6}: {len(buckets[status])}")

    matched = buckets["match"]
    print("\n--- matched files by job ---")
    by_job: dict[str, int] = defaultdict(int)
    for rel, detail in matched:
        by_job[detail.split("/")[0]] += 1
    for job, count in sorted(by_job.items()):
        print(f"  {job:>22}: {count}")

    # One-to-one check. Two archived CSVs matching the same cell would mean
    # the release contains duplicates of one number and the mapping is not
    # uniquely determined -- publishing on that basis would be arbitrary.
    cell_hits: dict[str, list[str]] = defaultdict(list)
    for rel, detail in matched:
        cell_hits[detail].append(rel)
    collisions = {cell: files for cell, files in cell_hits.items() if len(files) > 1}
    print(f"\ncells covered: {len(cell_hits)} / {len(index)}")
    if collisions:
        print(f"!! {len(collisions)} cell(s) matched by more than one file:")
        for cell, files in sorted(collisions.items()):
            print(f"   {cell} <- {files}")
    else:
        print("one-to-one: every matched file owns a distinct cell")

    if buckets["broken"]:
        print("\n--- broken ---")
        for rel, detail in buckets["broken"]:
            print(f"  {rel}: {detail}")

    # The flat files from kernel-output-v4 carry the baseline sweep that the
    # matrix jobs call `bicubic_*` / `identity_ll`. Report their means so
    # they can be matched by hand against the JSON if needed.
    print("\n--- flat per_image_*.csv (baseline sweep, kernel-output-v4) ---")
    for path in flat:
        means = {}
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = [f for f in (reader.fieldnames or []) if f.startswith("psnr")]
                totals = dict.fromkeys(fields, 0.0)
                counts = dict.fromkeys(fields, 0)
                for row in reader:
                    for field in fields:
                        raw = (row.get(field) or "").strip()
                        if not raw:
                            continue
                        value = float(raw)
                        if not math.isfinite(value):
                            continue
                        totals[field] += value
                        counts[field] += 1
            for field in fields:
                means[field] = (totals[field] / counts[field]) if counts[field] else None
        except (OSError, ValueError):
            means = {"error": True}
        rendered = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in means.items()
        )
        hit = [cell for cell, exp in index.items()
               if any(isinstance(v, float) and abs(v - exp) <= TOL for v in means.values())]
        verdict = f"matches {hit}" if hit else "matches no authoritative cell"
        print(f"  {os.path.relpath(path, REPO)}: {rendered}  -> {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
