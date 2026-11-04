#!/usr/bin/env python3
"""Figure 2: how many samples per image does a PSNR number need?

The figure exists to answer one question a reviewer will ask: *"if single-crop
validation is noisy, how noisy, and how do you know?"*

Data: bootstrap (5000 resamples) over the 28 disjoint 96x96 blocks of one
444x680 real photograph, evaluated with the SR x4 weights. For each sampling
budget n we draw n blocks, average their PSNR, and record the spread of that
average across resamples. `paper/data/fig2_sampling_noise.csv` holds the five
measured points; this script only draws them.

Two things the figure has to say without a caption:

  1. the noise falls as 1/sqrt(n) -- a statistical law, not an artifact of one
     unlucky experiment, so the grey reference curve is plotted on top;
  2. one sample per image (the common practice) carries ~1.6 dB of noise, which
     is ~5x the "+0.3 dB" improvements papers routinely claim. Resolving 0.3 dB
     needs >=16 samples per image, which for a 96x96 window means tiling most
     of the image -- i.e. full-image evaluation.

Honest scope, stated in the caption and repeated here: this is a *per-image*
noise curve from a single image, not a dataset-level confidence interval. A
100-image mean shrinks the noise by another factor of 10, which is why the
matrix in the paper uses full images rather than arguing from this curve.

Usage
-----
    python scripts/make_figure2.py                     # -> paper/figures/
    python scripts/make_figure2.py --out DIR --fmt pdf,png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: Same Okabe-Ito pair as Figure 1, so the two figures read as one family.
C_STD = "#0072B2"
C_PP = "#D55E00"
C_REF = "#999999"
C_CLAIM = "#009E73"
C_GRID = "#DDDDDD"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(HERE), "paper", "data",
                           "fig2_sampling_noise.csv")

#: What a paper typically claims. 0.3 dB is deliberately conservative: the
#: literature claims we audited range from +0.1 to +1 dB, and using the small
#: end makes the point harder to dismiss as cherry-picked.
CLAIM_DB = 0.3


def load(path):
    # utf-8-sig: the CSV is edited by hand and some editors prepend a BOM,
    # which would otherwise turn the first column name into "\ufeffn_samples"
    # and fail with a KeyError that points nowhere useful.
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    n = np.array([int(r["n_samples"]) for r in rows], dtype=float)
    std = np.array([float(r["std_db"]) for r in rows])
    pp = np.array([float(r["pp99_db"]) for r in rows])
    return n, std, pp


def draw(out_dir, fmts):
    # A single-column figure is 3.4 in wide. Matplotlib's 10 pt default fills it
    # and the result reads as a poster crop rather than a journal inset, so the
    # type scale is set explicitly here instead of globally (Figure 1 is a
    # two-column, six-panel figure and needs the other scale).
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "mathtext.fontset": "dejavusans",
    })

    n, std, pp = load(DEFAULT_CSV)
    order = np.argsort(n)
    n, std, pp = n[order], std[order], pp[order]

    fig, ax = plt.subplots(figsize=(3.45, 2.30))

    # 1/sqrt(n) reference through the first point. Fitted to n=1 only, not
    # least-squares over all five: the claim being made is "the law holds", and
    # a fit would hide a systematic departure from it.
    xs = np.linspace(1, 30.5, 200)
    ax.plot(xs, std[0] / np.sqrt(xs), "-", color=C_REF, lw=0.7, zorder=1,
            label=r"$1/\sqrt{n}$ reference")

    ax.plot(n, pp, "--s", color=C_PP, ms=2.8, lw=0.9, mfc="white", mew=0.8,
            zorder=3, label="99% peak-to-peak")
    ax.plot(n, std, "-o", color=C_STD, ms=3.2, lw=1.2, zorder=4,
            label="std. of the mean")

    # The claim line carries its own legend entry rather than an inline label:
    # at this width an inline string long enough to be informative collides
    # with the 1/sqrt(n) curve around n=12.
    ax.axhline(CLAIM_DB, color=C_CLAIM, lw=0.9, ls=(0, (4, 2)), zorder=2,
               label=f"typical claimed gain ({CLAIM_DB} dB)")

    # The gap at n=1 is the point of the figure: annotate the vertical distance
    # between one-crop noise and the effect being claimed.
    ax.annotate("", xy=(1, CLAIM_DB), xytext=(1, std[0]),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.7,
                                shrinkA=0, shrinkB=0))
    ax.text(2.3, 1.52,
            f"{std[0]:.2f} dB $= {std[0] / CLAIM_DB:.1f}\\times$ the claim",
            fontsize=6.2, color="#222222", va="center", ha="left")

    # Where the curve drops below the claim; what that costs in samples.
    ax.axvspan(16, 30.5, color=C_STD, alpha=0.07, zorder=0)
    ax.text(23.2, 3.05, "$\\geq$16 samples/image\nto resolve 0.3 dB",
            fontsize=6.0, color=C_STD, ha="center", va="center",
            linespacing=1.5)

    ax.set_xlabel("samples per image $n$")
    ax.set_ylabel("PSNR spread (dB)")
    ax.set_xlim(0.2, 30.5)
    ax.set_ylim(0, 5.95)
    ax.set_xticks([1, 4, 9, 16, 28])
    ax.grid(axis="y", color=C_GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Two columns: four rows of single-column entries ran out to x=12 and
    # pushed the ">=16" annotation out of its own shaded band.
    ax.legend(loc="upper right", frameon=False, handlelength=1.6,
              fontsize=5.6, ncol=2, columnspacing=0.9,
              borderpad=0.15, labelspacing=0.32, borderaxespad=0.2)

    fig.tight_layout(pad=0.25)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fmt in fmts:
        p = os.path.join(out_dir, f"fig2_sampling_noise.{fmt}")
        fig.savefig(p, dpi=400 if fmt == "png" else None, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE),
                                                  "paper", "figures"))
    ap.add_argument("--fmt", default="pdf,png")
    a = ap.parse_args()
    for p in draw(a.out, [f.strip() for f in a.fmt.split(",") if f.strip()]):
        print("wrote", p, f"{os.path.getsize(p) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
