"""Aggregation gap: mean(PSNR_i) vs PSNR of the mean MSE.

The two are not the same number and the gap is not a constant -- it grows as the
per-image scores spread out.  On a single 96x96 crop each image contributes one
number, so the between-image spread is the between-crop spread and the gap blows
up; on the full image the spread shrinks and so does the gap.

A note on what "the mean MSE" means here.  This script averages the per-image
MSEs with **equal weight**, which is the common way people re-aggregate a table
of per-image PSNR.  A textbook "global MSE" would instead weight each image by
its pixel count (total error / total pixels).  DIV2K val is not a fixed size --
long edge 2040, short edge 816 to 2040, a 2.5x spread in area -- so the two are
not identical, but they differ by only 0.02-0.05 dB: large images are not
systematically the hard ones.  We keep the equal-weight version because LOL has
no local copy to measure, and because 0.05 dB is two orders of magnitude below
the effect being reported.

Reads the per-image CSVs archived with the kernel runs, so the numbers here come
from the same 15 seed checkpoints the paper reports -- not from the deployment
weights in `serve/models/`, which are a different set of trainings.

    python scripts/aggregation_gap.py --out results/aggregation_gap.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st

TASKS = {
    "generator_x4": (
        "archive_private/06-实验脚本与输出/kernel-output-protocol-full-v2/generator_x4_x4/per_image",
        "sr_generator_scale4_best",
    ),
    "srcnn_x2": (
        "archive_private/06-实验脚本与输出/kernel-output-protocol-full-v2/srcnn_x2_x2/per_image",
        "sr_srcnn_scale2_best",
    ),
    "lowlight_unet": (
        "archive_private/06-实验脚本与输出/kernel-output-protocol-full-v4/lowlight_unet_x1/per_image",
        "lowlight_srcnn_scale2_best",
    ),
}
SEEDS = (42, 43, 44, 45, 46)


def load_psnr(path: str) -> dict[str, float]:
    with open(path, newline="") as fh:
        return {r["image"]: float(r["psnr"]) for r in csv.DictReader(fh)}


def image_areas(directory: str) -> dict[str, int]:
    """Pixel counts for the DIV2K val images, keyed by file name.

    Returns an empty dict when the images are not on disk (they are not
    redistributable, so a fresh clone will not have them).  Callers must handle
    that: the area-weighted column simply stays blank.
    """
    try:
        from PIL import Image
    except ImportError:
        return {}
    out = {}
    for f in glob.glob(os.path.join(directory, "*.png")):
        w, h = Image.open(f).size
        out[os.path.basename(f)] = w * h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/aggregation_gap.csv")
    ap.add_argument("--div2k-val", default="data/div2k/val",
                    help="DIV2K val pngs, used only for the area-weighted check")
    args = ap.parse_args()

    areas = image_areas(args.div2k_val)
    total_area = sum(areas.values()) or 0

    rows = []
    for task, (pdir, prefix) in TASKS.items():
        protos = sorted(
            {os.path.basename(f).split("__")[-1][:-4] for f in glob.glob(pdir + "/*.csv")},
            key=lambda p: int(p[1:]),
        )
        for proto in protos:
            per_seed_mean, per_seed_global, gaps = [], [], []
            per_seed_area = []
            n_inf = 0
            for s in SEEDS:
                f = os.path.join(pdir, f"{prefix}__s{s}__{proto}.csv")
                if not os.path.exists(f):
                    continue
                d = load_psnr(f)
                vals = list(d.values())
                finite = [v for v in vals if math.isfinite(v)]
                n_inf += len(vals) - len(finite)
                if not finite:
                    continue
                # mean of per-image PSNR
                m = st.mean(finite)
                # PSNR of the mean MSE (equal weights -- see the module docstring
                # for why this is not the pixel-weighted global MSE, and why that
                # choice is worth 0.05 dB).
                g = -10.0 * math.log10(st.mean([10.0 ** (-v / 10.0) for v in finite]))
                per_seed_mean.append(m)
                per_seed_global.append(g)
                gaps.append(m - g)
                # the same thing with pixel-count weights, when we have sizes
                covered = [k for k in d if k in areas and math.isfinite(d[k])]
                if total_area and len(covered) == len(finite):
                    wa = -10.0 * math.log10(
                        sum(areas[k] / total_area * 10.0 ** (-d[k] / 10.0) for k in covered))
                    per_seed_area.append(m - wa)
            if not gaps:
                continue
            rows.append({
                "task": task,
                "protocol": proto,
                "n_seeds": len(gaps),
                "n_images": len(load_psnr(
                    os.path.join(pdir, f"{prefix}__s{SEEDS[0]}__{proto}.csv"))),
                "n_inf": n_inf,
                "mean_psnr": f"{st.mean(per_seed_mean):.4f}",
                "psnr_of_mean_mse": f"{st.mean(per_seed_global):.4f}",
                "jensen_gap": f"{st.mean(gaps):+.4f}",
                "jensen_gap_std": f"{st.stdev(gaps):.4f}" if len(gaps) > 1 else "",
                "jensen_gap_areaweighted": (
                    f"{st.mean(per_seed_area):+.4f}" if per_seed_area else ""),
            })

    rows.sort(key=lambda r: (list(TASKS).index(r["task"]), int(r["protocol"][1:])))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")
    for r in rows:
        print(f"  {r['task']:14s} {r['protocol']}  "
              f"mean(PSNR)={r['mean_psnr']}  PSNR(meanMSE)={r['psnr_of_mean_mse']}  "
              f"gap={r['jensen_gap']} ± {r['jensen_gap_std']}  inf={r['n_inf']}")


if __name__ == "__main__":
    main()
