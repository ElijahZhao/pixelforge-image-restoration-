#!/usr/bin/env python3
"""Rebuild the paper's headline numbers from the public per-image CSVs alone.

This is the promise the Data Availability statement makes to a reader: "the
per-image values behind every protocol cell are released, and re-averaging
them reproduces the table". The script checks that promise mechanically.

For every authoritative matrix cell it re-averages the matching per-image
CSV and compares the result with the `psnr_mean` recorded in
`results/matrix_runs/*/protocol_matrix.json`. It then re-derives the
seed-level means and the cross-seed spread for each (task, protocol) pair,
which is what Table 3 and Appendix Table A1 report.

Exit status is non-zero if any cell fails to reproduce, so the release can
never be published in a state that contradicts the paper.

Usage:
    python3.11 scripts/verify_public_release.py
"""

from __future__ import annotations

import csv
import glob
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(REPO, "results", "per_image")
MATRIX_ROOT = os.path.join(REPO, "results", "matrix_runs")
TOL = 1e-3

#: task label -> (matrix job, checkpoint stem)
TASK_JOBS = {
    "SR x4": ("generator_x4_x4", "sr_generator_scale4_best__s"),
    "SR x2": ("srcnn_x2_x2", "sr_srcnn_scale2_best__s"),
    "Low-light": ("lowlight_unet_x1", "lowlight_srcnn_scale2_best__s"),
}


def read_csv_psnr(path: str) -> list[float]:
    """Finite PSNR values from a per-image CSV."""
    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("psnr") or "").strip()
            if not raw:
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
    return values


def main() -> int:
    if not os.path.isdir(RELEASE):
        print(f"[fail] public release directory missing: {RELEASE}", file=sys.stderr)
        return 1

    failures: list[str] = []
    checked = 0

    # ---- level 1: every released file reproduces its matrix cell -----------
    for json_path in sorted(glob.glob(os.path.join(MATRIX_ROOT, "*", "protocol_matrix.json"))):
        job = os.path.basename(os.path.dirname(json_path))
        with open(json_path, encoding="utf-8") as handle:
            import json
            data = json.load(handle)
        for ckpt, payload in data.get("checkpoints", {}).items():
            for proto, stats in payload.get("protocols", {}).items():
                rel = f"{job}__{ckpt}__{proto}.csv"
                path = os.path.join(RELEASE, rel)
                if not os.path.exists(path):
                    failures.append(f"missing release file for {job}/{ckpt}/{proto}")
                    continue
                values = read_csv_psnr(path)
                if not values:
                    failures.append(f"{rel}: no finite PSNR values")
                    continue
                recomputed = statistics.fmean(values)
                expected = float(stats["psnr_mean"])
                checked += 1
                if abs(recomputed - expected) > TOL:
                    failures.append(
                        f"{rel}: recomputed {recomputed:.4f} != matrix {expected:.4f}"
                    )

    print(f"cell-level check: {checked} cells re-averaged from released CSVs")

    # ---- level 2: cross-seed spread for each (task, protocol) --------------
    print("\ncross-seed means (what Table 3 / Table A1 report):")
    for task, (job, ckpt_stem) in TASK_JOBS.items():
        json_path = os.path.join(MATRIX_ROOT, job, "protocol_matrix.json")
        with open(json_path, encoding="utf-8") as handle:
            import json
            data = json.load(handle)
        protocols = sorted(
            {p for ck in data["checkpoints"].values() for p in ck["protocols"]},
            key=lambda s: int(s[1:]),
        )
        for proto in protocols:
            means = []
            for ckpt, payload in data["checkpoints"].items():
                if not ckpt.startswith(ckpt_stem):
                    continue
                rel = f"{job}__{ckpt}__{proto}.csv"
                path = os.path.join(RELEASE, rel)
                if not os.path.exists(path):
                    continue
                means.append(statistics.fmean(read_csv_psnr(path)))
            if len(means) < 2:
                continue
            spread = statistics.stdev(means)
            # Cross-check against the JSON's own seed spread.
            json_spreads = [
                float(p["protocols"][proto]["psnr_std"])
                for ck, p in data["checkpoints"].items()
                if ck.startswith(ckpt_stem)
            ]
            flag = ""
            if json_spreads and abs(spread - statistics.fmean(json_spreads)) > 0.5:
                flag = "  <-- differs from per-protocol std (expected: that is image std)"
            print(f"  {task:11} {proto}: mean={statistics.fmean(means):8.4f}  "
                  f"seed-spread={spread:.4f}  n_seeds={len(means)}{flag}")

    if failures:
        print(f"\n[FAIL] {len(failures)} problem(s):")
        for line in failures[:20]:
            print(f"  {line}")
        return 1

    print("\n[OK] every released per-image CSV reproduces its matrix cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
