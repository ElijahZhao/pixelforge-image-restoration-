#!/usr/bin/env python3
"""Rebuild the three Figure-1 tables from the raw matrix JSONs.

Why this script exists
----------------------
`results/protocol_matrix_fig1_long.csv`, `..._wide.csv` and
`model_vs_bicubic_gain.csv` are the numbers the paper quotes. Until now they
were assembled by hand from the kernel outputs, which means a reader could
check them only by trusting us. This script makes the path mechanical:

    results/matrix_runs/*/protocol_matrix.json   (raw, one dir per kernel job)
        -> long   (one row per checkpoint x protocol)
        -> wide   (mean/std over the five seeds)
        -> gain   (model minus the same-protocol baseline)

Anyone can rerun it and diff the result against the committed CSVs.

Aggregation conventions (must match `results/seeds/aggregate.md`)
-----------------------------------------------------------------
* ``psnr_mean`` is the arithmetic mean over the N seeds.
* ``psnr_std``  is the **sample** standard deviation (ddof=1), not the
  population one. With n=5 the two differ by ~10%, which is larger than every
  seed effect in this project, so the choice is not cosmetic.
* ``psnr_min`` / ``psnr_max`` are the extremes over the *seeds' means*, not over
  images. Image-level spread lives at 4-6 dB and would misrepresent the error
  bar by two orders of magnitude.

Baselines
---------
A job directory is a baseline when its checkpoints are named ``bicubic`` or
``identity``. The reference is *not* always bicubic: in the paired low-light
task there is no upscaling step, so the honest do-nothing line is the capture
itself, and the table says ``identity``.

Usage
-----
    python scripts/aggregate_matrix.py                       # defaults below
    python scripts/aggregate_matrix.py --runs-root DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

#: (task, scale, paired, model_job_dir, baseline_job_dir)
#: Paths are relative to ``--runs-root``; a missing baseline is allowed and
#: emits empty gain columns rather than failing, so a half-finished matrix can
#: still be inspected.
DEFAULT_JOBS = [
    ("generator_x4", 4, False, "generator_x4_x4", "bicubic_sr_x4"),
    ("srcnn_x2", 2, False, "srcnn_x2_x2", "bicubic_sr_x2"),
    ("lowlight_unet", 1, True, "lowlight_unet_x1", "identity_ll"),
]

BASELINE_NAMES = {"bicubic", "identity", "bicubic_sr", "identity_ll"}

LONG_COLS = ["task", "scale", "paired", "dataset", "checkpoint", "protocol",
             "label", "degrade", "metric", "sampling", "agg", "psnr", "ssim",
             "n_images"]
WIDE_COLS = ["task", "scale", "paired", "protocol", "label", "degrade", "metric",
             "sampling", "agg", "n_images", "n_seeds", "psnr_mean", "psnr_std",
             "psnr_min", "psnr_max"]
GAIN_COLS = ["task", "protocol", "reference", "model_mean", "reference_psnr",
             "gain", "n_seeds", "n_aligned", "gain_naive_mixed"]


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _is_baseline(name: str) -> bool:
    return name in BASELINE_NAMES or name.startswith("bicubic") or \
        name.startswith("identity")


def collect(job_dir: str, baseline_dir: str | None):
    """Return (per_checkpoint_rows, baseline_rows, n_images, dataset)."""
    model_rows, base_rows = [], []
    n_images = None
    dataset = None
    for label, path in (("model", job_dir), ("base", baseline_dir)):
        if not path:
            continue
        js = os.path.join(path, "protocol_matrix.json")
        if not os.path.isfile(js):
            print(f"[warn] missing {js}", file=sys.stderr)
            continue
        d = _load(js)
        for ck, rec in d.get("checkpoints", {}).items():
            ds = rec.get("dataset") or d.get("config", {}).get("data", "")
            if dataset is None and not _is_baseline(ck):
                dataset = ds
            for proto, m in rec.get("protocols", {}).items():
                row = {
                    "dataset": ds, "checkpoint": ck, "protocol": proto,
                    "label": m.get("label", ""), "degrade": m.get("degrade", ""),
                    "metric": m.get("metric", ""), "sampling": m.get("sampling", ""),
                    "agg": m.get("agg", ""), "psnr": m.get("psnr_mean"),
                    "ssim": m.get("ssim_mean"),
                    "n_images": m.get("n_images"),
                }
                if n_images is None:
                    n_images = m.get("n_images")
                (base_rows if _is_baseline(ck) else model_rows).append((ck, row))
    return model_rows, base_rows, n_images, dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="results/matrix_runs",
                    help="directory holding one subdirectory per kernel job")
    ap.add_argument("--out", default="results", help="where the CSVs go")
    ap.add_argument("--p1-alignment", default="results/p1_inf_alignment.csv",
                    help="P1 gain realigned on the images where the baseline is "
                         "finite (produced by scripts/align_p1_inf.py); "
                         "pass '' to fall back to the naive mixed count")
    args = ap.parse_args()

    # P1's bicubic baseline can reconstruct flat patches exactly (PSNR = +inf),
    # so its aggregate covers 95 of 100 images while the model's covers 100.
    # `align_p1_inf.py` rebuilds both sides on the 95-image intersection; this
    # table overrides the P1 gain when present. Missing file is not an error --
    # the naive value is still emitted in `gain_naive_mixed`.
    aligned: dict[str, dict] = {}
    if args.p1_alignment and os.path.exists(args.p1_alignment):
        with open(args.p1_alignment, newline="") as fh:
            aligned = {r["task"]: r for r in csv.DictReader(fh)}
        print(f"[info] P1 gains realigned from {args.p1_alignment} "
              f"({', '.join(sorted(aligned))})")
    else:
        print("[warn] no P1 alignment table: P1 gain stays on the mixed count "
              "(baseline 95 images vs model 100)", file=sys.stderr)

    long_rows, wide_rows, gain_rows = [], [], []
    for task, scale, paired, job, base_job in DEFAULT_JOBS:
        mrows, brows, n_images, dataset = collect(
            os.path.join(args.runs_root, job),
            os.path.join(args.runs_root, base_job) if base_job else None)
        if not mrows:
            print(f"[warn] no model rows for {task}", file=sys.stderr)
            continue

        for ck, row in mrows:
            long_rows.append({
                "task": task, "scale": scale, "paired": int(paired),
                "dataset": row["dataset"] or dataset, "checkpoint": ck,
                "protocol": row["protocol"], "label": row["label"],
                "degrade": row["degrade"], "metric": row["metric"],
                "sampling": row["sampling"], "agg": row["agg"],
                "psnr": f"{float(row['psnr']):.4f}" if row["psnr"] is not None else "",
                "ssim": f"{float(row['ssim']):.4f}" if row["ssim"] is not None else "",
                "n_images": row["n_images"] if row["n_images"] is not None else n_images,
            })

        # ---- wide: mean / sample-std / min / max over the seed means --------
        by_proto: dict[str, list] = {}
        meta: dict[str, dict] = {}
        for ck, row in mrows:
            by_proto.setdefault(row["protocol"], []).append(float(row["psnr"]))
            meta.setdefault(row["protocol"], row)
        for proto in sorted(by_proto, key=lambda p: int(p[1:])):
            vals = by_proto[proto]
            m = meta[proto]
            wide_rows.append({
                "task": task, "scale": scale, "paired": int(paired), "protocol": proto,
                "label": m["label"], "degrade": m["degrade"], "metric": m["metric"],
                "sampling": m["sampling"], "agg": m["agg"],
                "n_images": m["n_images"] if m["n_images"] is not None else n_images,
                "n_seeds": len(vals),
                "psnr_mean": f"{st.mean(vals):.4f}",
                "psnr_std": f"{st.stdev(vals):.4f}" if len(vals) > 1 else "",
                "psnr_min": f"{min(vals):.4f}",
                "psnr_max": f"{max(vals):.4f}",
            })

        # ---- gain: model mean minus the same-protocol baseline --------------
        base_by_proto: dict[str, float] = {}
        base_name = None
        for ck, row in brows:
            base_by_proto[row["protocol"]] = float(row["psnr"])
            base_name = base_name or ck
        if not base_by_proto:
            print(f"[warn] no baseline for {task}: gain columns stay empty",
                  file=sys.stderr)
        for proto in sorted(by_proto, key=lambda p: int(p[1:])):
            model_mean = st.mean(by_proto[proto])
            ref = base_by_proto.get(proto)
            n_aligned: object = ""
            naive: object = ""
            if proto == "P1" and task in aligned:
                # Report the like-for-like comparison: both sides averaged over
                # the images where the baseline produced a finite PSNR. The
                # 100-image model mean is kept in `gain_naive_mixed` so the size
                # of the correction stays visible.
                a = aligned[task]
                naive = a["gain_naive_mixed"]
                model_mean = float(a["model_mean_aligned"])
                ref = float(a["baseline_mean_aligned"])
                n_aligned = a["n_images_aligned"]
            gain_rows.append({
                "task": task, "protocol": proto,
                # `paired` decides the semantics, not the checkpoint name: the
                # archived low-light baseline JSON still calls its key
                # "bicubic" because that string was hardcoded before the
                # paired path existed. In paired mode nothing is upscaled, so
                # the reference is "do nothing" -- an identity, not a resampler.
                "reference": "identity" if paired else "bicubic",
                "model_mean": f"{model_mean:.4f}",
                "reference_psnr": f"{ref:.4f}" if ref is not None else "",
                "gain": f"{model_mean - ref:.4f}" if ref is not None else "",
                "n_seeds": len(by_proto[proto]),
                "n_aligned": n_aligned,
                "gain_naive_mixed": naive,
            })

    os.makedirs(args.out, exist_ok=True)
    for name, cols, rows in (
            ("protocol_matrix_fig1_long.csv", LONG_COLS, long_rows),
            ("protocol_matrix_fig1_wide.csv", WIDE_COLS, wide_rows),
            ("model_vs_bicubic_gain.csv", GAIN_COLS, gain_rows)):
        path = os.path.join(args.out, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[i] {path}: {len(rows)} rows")

    # A one-screen summary, so a wrong number is visible without opening a CSV.
    print("\n=== gain summary (model - baseline, same protocol) ===")
    for r in gain_rows:
        g = r["gain"] or "  —  "
        flag = "  <-- sign flip" if r["gain"] and float(r["gain"]) < 0 else ""
        print(f"  {r['task']:14s} {r['protocol']} {g:>8s}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
