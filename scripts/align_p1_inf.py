#!/usr/bin/env python3
"""Realign the P1 (best-of-60) gain against the bicubic baseline's censored images.

Why this script exists
----------------------
`scripts/protocol_matrix.py` excludes non-finite PSNR values from the aggregate
and records the count in ``n_inf`` (a bicubic upsampler can reconstruct a flat
patch *exactly*, so MSE == 0 and PSNR == +inf).  That is the right thing to do
for the baseline's own mean, but it silently changes the *sample*: P1's
``psnr_mean`` for a bicubic baseline is an average over 95 images while the
model's is an average over 100.  Subtracting the two mixes two different image
sets and shifts the reported gain.

This script rebuilds the comparison on the intersection of images -- the 95
where the baseline produced a finite PSNR -- for both sides, and writes the
naive (mixed) and aligned numbers side by side so the difference is auditable.

Scope: P1 only.  P2-P8 have ``n_inf == 0`` for every baseline in this project
(checked in ``results/matrix_runs/*/protocol_matrix.json``); the paired
low-light baseline (identity) also has ``n_inf == 0``, so nothing else is
affected.

Usage
-----
    python scripts/align_p1_inf.py \
        --runs-root results/matrix_runs \
        --per-image-root archive_private/06-实验脚本与输出 \
        --out results/p1_inf_alignment.csv

The per-image CSVs live in the private archive (they are the kernel outputs);
the *result* of this script is small and belongs in the public repo, so a
reader can check the paper's P1 numbers without re-running the matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys

#: task key -> (model job dir, model per-image prefix, baseline job dir,
#:              baseline per-image prefix, baseline per-image dir override)
JOBS = {
    "generator_x4": {
        "model_job": "generator_x4_x4",
        "model_prefix": "sr_generator_scale4_best",
        "base_job": "bicubic_sr_x4",
        "base_prefix": "bicubic",
        "base_dir": "kernel-output-protocol-full-v3/bicubic_sr_x4/per_image",
        "model_dir": "kernel-output-protocol-full-v2/generator_x4_x4/per_image",
    },
    "srcnn_x2": {
        "model_job": "srcnn_x2_x2",
        "model_prefix": "sr_srcnn_scale2_best",
        "base_job": "bicubic_sr_x2",
        "base_prefix": "bicubic",
        "base_dir": "local-baseline-bicubic-sr-x2/per_image",
        "model_dir": "kernel-output-protocol-full-v2/srcnn_x2_x2/per_image",
    },
}
SEEDS = (42, 43, 44, 45, 46)
PROTOCOL = "P1"


def load_per_image(path: str) -> dict[str, float]:
    with open(path, newline="") as fh:
        return {row["image"]: float(row["psnr"]) for row in csv.DictReader(fh)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default="results/matrix_runs")
    ap.add_argument("--per-image-root",
                    default="archive_private/06-实验脚本与输出")
    ap.add_argument("--out", default="results/p1_inf_alignment.csv")
    args = ap.parse_args(argv)

    rows: list[dict[str, object]] = []
    for task, spec in JOBS.items():
        base_csv = os.path.join(args.per_image_root, spec["base_dir"],
                                f"{spec['base_prefix']}__{PROTOCOL}.csv")
        if not os.path.exists(base_csv):
            print(f"[skip] {task}: missing {base_csv}", file=sys.stderr)
            continue
        base = load_per_image(base_csv)
        inf_images = sorted(k for k, v in base.items() if not math.isfinite(v))
        finite = {k: v for k, v in base.items() if math.isfinite(v)}

        seeds: dict[int, dict[str, float]] = {}
        for s in SEEDS:
            p = os.path.join(args.per_image_root, spec["model_dir"],
                             f"{spec['model_prefix']}__s{s}__{PROTOCOL}.csv")
            if not os.path.exists(p):
                print(f"[skip] {task}: missing {p}", file=sys.stderr)
                break
            seeds[s] = load_per_image(p)
        if len(seeds) != len(SEEDS):
            continue

        common = set(finite)
        for m in seeds.values():
            common &= set(m)

        base_aligned = statistics.mean([finite[k] for k in common])
        model_aligned_per_seed = [statistics.mean([seeds[s][k] for k in common])
                                  for s in SEEDS]
        model_aligned = statistics.mean(model_aligned_per_seed)
        model_full = statistics.mean(
            [statistics.mean(seeds[s].values()) for s in SEEDS])

        # Cross-check against the aggregate JSON so a mismatch is loud.
        js = json.load(open(os.path.join(args.runs_root, spec["base_job"],
                                         "protocol_matrix.json")))
        ck = next(iter(js["checkpoints"].values()))
        rec = ck["protocols"][PROTOCOL]
        if rec.get("n_inf") != len(inf_images):
            print(f"[warn] {task}: n_inf in JSON ({rec.get('n_inf')}) != "
                  f"counted from CSV ({len(inf_images)})", file=sys.stderr)
        if abs((rec.get("psnr_mean") or 0) - statistics.mean(finite.values())) > 5e-4:
            print(f"[warn] {task}: baseline mean does not reproduce from CSV",
                  file=sys.stderr)

        # Round the two means first, then subtract: the gain column must be
        # reproducible as `model_mean_aligned - baseline_mean_aligned` from the
        # values printed in this same row (otherwise the two CSVs disagree in
        # the 4th decimal for purely cosmetic reasons).
        base_r = round(base_aligned, 4)
        model_r = round(model_aligned, 4)
        rows.append({
            "task": task,
            "protocol": PROTOCOL,
            "n_images_total": len(base),
            "n_images_aligned": len(common),
            "n_inf_baseline": len(inf_images),
            "inf_images": "|".join(inf_images),
            "baseline_mean_aligned": f"{base_r:.4f}",
            "model_mean_aligned": f"{model_r:.4f}",
            "gain_aligned": f"{model_r - base_r:+.4f}",
            "model_mean_full": f"{model_full:.4f}",
            "gain_naive_mixed": f"{round(model_full, 4) - base_r:+.4f}",
            "seeds": " ".join(str(s) for s in SEEDS),
        })

    if not rows:
        print("nothing to do (no per-image CSV found)", file=sys.stderr)
        return 1

    cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out}  ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['task']:>6}  n={r['n_images_aligned']}/{r['n_images_total']}"
              f"  inf={r['n_inf_baseline']}"
              f"  base={r['baseline_mean_aligned']}"
              f"  model={r['model_mean_aligned']}"
              f"  gain={r['gain_aligned']:>8}"
              f"   (naive {r['gain_naive_mixed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
