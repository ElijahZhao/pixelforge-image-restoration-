#!/usr/bin/env python3
"""Aggregate a seed matrix into mean +- std + 95% CI.

Reads the per-run training CSVs produced by

    python train/train.py ... --seed S --tag seeds/s{S}

(locally, or pulled from the Stage-4 Kaggle kernel `pixelforge-seed-matrix`,
whose output tree is ``runs/<model>/s<seed>/train_log_*.csv``), and prints one
row per model with the spread across seeds.

Coordinate-system warning
-------------------------
The numbers in these CSVs are the **training-time validation** metric:
centre-crop (``--crop_size``, default 96/128), per-image mean, LR built by the
training-side degradation function. They are NOT the full-image literature
protocol (Y channel + shave). So this table answers "how much does the seed
move the number", not "what is the model's performance". The paper's main
table is produced by re-evaluating these same checkpoints under P1..P6
(Stage 5). Mixing the two is exactly the aggregation-口径 mistake this project
already paid for once (11.85 dB between two aggregation choices).

Best-of-N warning
-----------------
``--epochs`` selection during training already picks the best epoch by
val_psnr, i.e. every run is a best-of-N. Reporting the mean of those maxima is
honest only if it is labelled as such; the pre-registered rule for the paper
is: report **final-epoch mean** as the headline, and ``best`` only as an upper
bound with its range. Both are printed; neither is silently preferred.

Usage
-----
    python scripts/aggregate_seeds.py --runs-root results
    python scripts/aggregate_seeds.py --runs-root kaggle_output/runs \
        --runs-root results --out results/seeds_summary.csv --md results/seeds_summary.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

# Two-sided 95% t quantiles for small n (df = n-1). With 5 seeds t=2.776; a
# normal approximation (z=1.96) would understate the interval width by ~29%
# (1 - 1.96/2.776) at n=5, so every CI in the paper uses this table.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}


def parse_csv(path):
    """Return (meta, rows) from a train_log CSV; ``# k=v`` lines become meta."""
    meta, rows = {}, []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("#"):
                body = row[0].lstrip("#").strip()
                if "=" in body:
                    k, v = body.split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            if row[0] == "epoch":
                continue
            try:
                rows.append({"epoch": int(row[0]), "train_loss": float(row[1]),
                             "val_psnr": float(row[2]),
                             "val_ssim": float(row[3])})
            except (ValueError, IndexError):
                continue
    return meta, rows


def run_key_from_name(path):
    """train_log_sr_srcnn_scale2.csv -> ('sr', 'srcnn', 2)."""
    stem = os.path.basename(path)[len("train_log_"):-len(".csv")]
    parts = stem.rsplit("_scale", 1)
    if len(parts) != 2:
        return None
    head, scale = parts
    task = "lowlight" if head.startswith("lowlight") else "sr"
    model = head[len(task) + 1:]
    try:
        return task, model, int(scale)
    except ValueError:
        return None


def seed_from_path(path):
    """.../seeds/s43/train_log_*.csv -> 43 (None if the path has no seed dir)."""
    for part in reversed(os.path.dirname(path).split(os.sep)):
        if len(part) > 1 and part[0] == "s" and part[1:].isdigit():
            return int(part[1:])
    return None


def collect(roots):
    """{run_key: {seed: info}} over every CSV under the given roots."""
    table = defaultdict(dict)
    problems = []
    for root in roots:
        if not os.path.isdir(root):
            problems.append(f"not a directory: {root}")
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "train_log_*.csv"),
                                     recursive=True)):
            key = run_key_from_name(path)
            if key is None:
                problems.append(f"unparsable filename: {path}")
                continue
            seed = seed_from_path(path)
            if seed is None:
                problems.append(f"no seed directory in path: {path}")
                continue
            meta, rows = parse_csv(path)
            if not rows:
                problems.append(f"no data rows: {path}")
                continue
            # The CSV names its own seed. If that disagrees with the directory,
            # one of them is lying and every statistic below is suspect -- the
            # v1.4.0 guard exists precisely to make this impossible.
            if meta.get("seed") and meta["seed"] != str(seed):
                problems.append(
                    f"{path}: directory says s{seed} but CSV header says "
                    f"seed={meta['seed']}")
            if seed in table[key]:
                problems.append(f"duplicate seed {seed} for {key}: {path}")
                continue
            table[key][seed] = {
                "path": path, "meta": meta, "n_epochs": len(rows),
                "final_psnr": rows[-1]["val_psnr"], "final_ssim": rows[-1]["val_ssim"],
                "best_psnr": max(r["val_psnr"] for r in rows),
                "best_ssim": max(r["val_ssim"] for r in rows),
                "best_epoch": max(rows, key=lambda r: r["val_psnr"])["epoch"],
                "last_epoch": rows[-1]["epoch"],
            }
    return table, problems


def stats(values):
    n = len(values)
    if n == 0:
        return {}
    mean = statistics.mean(values)
    out = {"n": n, "mean": mean, "min": min(values), "max": max(values)}
    if n > 1:
        sd = statistics.stdev(values)  # ddof=1: these are samples, not a population
        out["std"] = sd
        t = T95.get(n, 1.96)
        half = t * sd / (n ** 0.5)
        out["ci95"] = (mean - half, mean + half)
        out["t"] = t
    return out


def fmt(s, nd=4):
    if not s:
        return "—"
    if "std" in s:
        ci = f" [{s['ci95'][0]:.{nd}f}, {s['ci95'][1]:.{nd}f}]"
        return f"{s['mean']:.{nd}f} ± {s['std']:.{nd}f}{ci}"
    return f"{s['mean']:.{nd}f} (n=1, 无 std)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", action="append", required=True,
                    help="directory to scan (repeatable); e.g. results, or a "
                         "pulled kaggle output containing runs/")
    ap.add_argument("--out", default="", help="write a tidy CSV here")
    ap.add_argument("--md", default="", help="write a Markdown table here")
    ap.add_argument("--expect-seeds", type=int, default=5,
                    help="flag runs with fewer seeds than this (default 5)")
    args = ap.parse_args(argv)

    table, problems = collect(args.runs_root)
    if not table:
        print("no train_log CSV found under: " + ", ".join(args.runs_root))
        for p in problems:
            print("  ! " + p)
        return 2

    rows_out, md = [], []
    md.append("| 模型 | n | final-epoch PSNR (均值 ± std [95% CI]) "
              "| best-of-N PSNR (均值 ± std [95% CI]) | best 范围 |")
    md.append("|---|---|---|---|---|")
    for key in sorted(table):
        task, model, scale = key
        seeds = table[key]
        fin = stats([v["final_psnr"] for v in seeds.values()])
        best = stats([v["best_psnr"] for v in seeds.values()])
        name = f"{task}/{model}/×{scale}"
        n_ep = sorted({v["n_epochs"] for v in seeds.values()})
        if len(n_ep) > 1 or n_ep[0] < 2:
            problems.append(f"{name}: uneven epoch counts across seeds {n_ep}")
        if len(seeds) < args.expect_seeds:
            problems.append(f"{name}: only {len(seeds)} seed(s), "
                            f"expected {args.expect_seeds}")
        md.append(f"| {name} | {len(seeds)} | {fmt(fin)} | {fmt(best)} "
                  f"| {best['min']:.4f} – {best['max']:.4f} |")
        rows_out.append({
            "run": name, "task": task, "model": model, "scale": scale,
            "n_seeds": len(seeds), "seeds": " ".join(str(s) for s in sorted(seeds)),
            "epochs": n_ep[0] if len(n_ep) == 1 else "mixed",
            "final_psnr_mean": round(fin["mean"], 4),
            "final_psnr_std": round(fin.get("std", 0.0), 4),
            "final_psnr_ci95_low": round(fin["ci95"][0], 4) if "ci95" in fin else "",
            "final_psnr_ci95_high": round(fin["ci95"][1], 4) if "ci95" in fin else "",
            "best_psnr_mean": round(best["mean"], 4),
            "best_psnr_std": round(best.get("std", 0.0), 4),
            "best_psnr_min": round(best["min"], 4),
            "best_psnr_max": round(best["max"], 4),
        })

    print("\n".join(md))
    print()
    print("口径: 训练内验证（中心裁剪 / per-image mean / 训练侧退化算子），"
          "不是全图 Y+shave 文献口径；主表由阶段 5 用 P1–P6 重评这些 checkpoint。")
    print("best-of-N: 每个 run 的 best 是 --epochs 内按 val_psnr 挑的最大值，"
          "均值只能作为上界报告，headline 用 final-epoch 均值。")
    if problems:
        print("\n注意:")
        for p in problems:
            print("  ! " + p)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {args.out}")
    if args.md:
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
