#!/usr/bin/env python3
"""Figure 1: the protocol matrix.

One figure, two claims:

  1. **A protocol choice moves the reported PSNR by far more than the model
     does.** Within one task, the eight protocols span up to 19.8 dB; the
     model's own improvement over its same-protocol baseline is 0.8 dB.
  2. **A protocol choice can flip the sign of the conclusion.** On SR x4 the
     same weights beat bicubic by +0.82 dB when downsampling is antialiased
     (P5) and lose by -1.75 dB when it is not (P6). Nothing about the model
     changed; one default did.

Top row: absolute PSNR, model against its same-protocol reference.
Bottom row: the difference, which is what a paper would actually report.

P1 (best-of-60 random crops) is drawn off-scale on purpose: it sits far above
every other protocol (x4: 15.9-19.8 dB, x2: 12.5-15.6, low-light: 3.8-6.1 --
task-dependent, do not quote a single range), and letting it set the y-range
would squash the 0.8 dB effect the figure exists to show. Its value is printed
above the axis so nothing is hidden.

Usage
-----
    python scripts/make_figure1.py                 # -> paper/figures/
    python scripts/make_figure1.py --out DIR --fmt pdf,png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#: Okabe-Ito, which stays distinguishable under the common colour-vision
#: deficiencies. Model/baseline are also separated by marker shape and line
#: style, so the figure survives being printed in greyscale.
C_MODEL = "#0072B2"
C_BASE = "#D55E00"
C_NEG = "#D55E00"
C_POS = "#0072B2"
C_GRID = "#DDDDDD"

TASKS = [
    ("generator_x4", "SR $\\times$4 (SRResNet)", "bicubic"),
    ("srcnn_x2", "SR $\\times$2 (SRCNN)", "bicubic"),
    ("lowlight_unet", "Low-light (U-Net)", "identity"),
]

#: The axis carries P1..P8 only. Two-line descriptive ticks ("P6 no AA")
#: overlapped into an unreadable band at this column width, and a reader needs
#: the *identity* of the protocol, not a second copy of Table 1. The full
#: definitions live in that table and in the caption.
P_SHORT = {p: p for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")}


def read_wide(path: str):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_gain(path: str):
    with open(path) as f:
        return {(r["task"], r["protocol"]): r for r in csv.DictReader(f)}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,   # keep text selectable / editable in the PDF
        "ps.fonttype": 42,
    })


def panel_top(ax, task, title, ref_name, wide, gain, show_ylabel, show_xlabel):
    rows = [r for r in wide if r["task"] == task]
    protos = [r["protocol"] for r in rows]
    xs = list(range(len(protos)))
    model = [float(r["psnr_mean"]) for r in rows]
    # Seed spread. At n=5 this is 0.019-0.094 dB across tasks (x4 up to 0.036,
    # x2 up to 0.094) -- mostly thinner than the marker, so it is drawn but
    # deliberately not allowed to suggest precision the design does not have.
    mstd = [float(r["psnr_std"]) if r["psnr_std"] else 0.0 for r in rows]
    base = []
    for p in protos:
        g = gain.get((task, p))
        base.append(float(g["reference_psnr"]) if g and g["reference_psnr"] else float("nan"))

    base_label = "bicubic" if ref_name == "bicubic" else "identity (do nothing)"

    # P1 is off-scale by construction (see module docstring). Drawing it on the
    # same axis would (a) squash the 0.8 dB effect the figure exists to show and
    # (b) paint a diagonal line from 45 dB down to 26 dB that reads as a trend
    # rather than as one outlier protocol. So the polyline starts at P2 and P1
    # is marked separately at the top edge.
    p1 = protos.index("P1") if "P1" in protos else None
    keep = [i for i in xs if i != p1]
    kx = [xs[i] for i in keep]
    km = [model[i] for i in keep]
    kb = [base[i] for i in keep]
    ks = [mstd[i] for i in keep]
    finite_b = [v for v in kb if v == v]
    hi = max(km + finite_b) if finite_b else max(km)
    lo = min(km + finite_b) if finite_b else min(km)
    span = max(hi - lo, 0.5)

    ax.plot(kx, km, "-o", color=C_MODEL, ms=3.4, lw=1.3, zorder=3)
    if finite_b:
        ax.plot(kx, kb, "--s", color=C_BASE, ms=3.0, lw=1.1, mfc="white",
                mew=0.9, zorder=2)
    ax.errorbar(kx, km, yerr=ks, fmt="none", ecolor=C_MODEL, elinewidth=0.7,
                capsize=1.4, zorder=4)

    top = hi + span * 0.16
    if p1 is not None:
        v = model[p1]
        ax.plot([0], [top], marker="^", ms=4.4, color=C_MODEL, clip_on=False,
                zorder=6, mew=0)
        ax.text(0.16, top, f"P1 {v:.2f}", fontsize=6.4, color=C_MODEL,
                ha="left", va="center", zorder=6, fontweight="bold")
        ax.set_ylim(lo - span * 0.12, top + span * 0.10)
    else:
        ax.set_ylim(lo - span * 0.12, hi + span * 0.12)

    ax.set_xticks(xs)
    ax.set_xticklabels([P_SHORT.get(p, p) for p in protos] if show_xlabel else [])
    ax.set_xlim(-0.45, len(protos) - 0.55)
    ax.grid(axis="y", color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, pad=4)
    if show_ylabel:
        ax.set_ylabel("PSNR (dB)")
    if show_xlabel:
        ax.set_xlabel("protocol")


def panel_bottom(ax, task, gain, wide, show_ylabel, show_xlabel, ylim=None):
    rows = [r for r in wide if r["task"] == task]
    protos = [r["protocol"] for r in rows]
    vals, colors = [], []
    for p in protos:
        g = gain.get((task, p))
        v = float(g["gain"]) if g and g["gain"] else float("nan")
        vals.append(v)
        colors.append(C_NEG if (v == v and v < 0) else C_POS)

    xs = list(range(len(protos)))
    if any(v == v for v in vals):
        ax.bar(xs, [0 if v != v else v for v in vals], width=0.62, color=colors,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(0, color="black", lw=0.8, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([P_SHORT.get(p, p) for p in protos] if show_xlabel else [])
    ax.set_xlim(-0.45, len(protos) - 0.55)
    ax.grid(axis="y", color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    finite = [v for v in vals if v == v]
    if ylim is not None:
        # A fixed range shared by both SR panels. Letting each panel autoscale
        # made x4 look like it moved further than x2 while the opposite is true
        # (-1.75 vs -1.89), because the two axes had different limits.
        ax.set_ylim(*ylim)
    if not finite:
        ax.text(0.5, 0.62, "baseline not run", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="#888888",
                style="italic")
        ax.set_yticks([])
    if finite and ylim is None:
        span = max(finite) - min(finite)
        pad = max(span * 0.30, 0.25)
        ax.set_ylim(min(min(finite), 0) - pad, max(max(finite), 0) + pad)
    if finite:
        for x, v in zip(xs, vals):
            if v != v or abs(v) < 0.05:
                continue
            neg = v < 0
            # White-on-bar was tried and reverted: "−1.75" is wider than the
            # 0.62-wide bar, so the overhang fell outside the patch and the
            # white glyphs vanished against the page. Labels go outside, and
            # the shared y-range reserves room for them.
            ax.annotate(f"{v:+.2f}", (x, v), fontsize=5.6, ha="center",
                        va="top" if neg else "bottom",
                        xytext=(0, -1.6 if neg else 1.2),
                        textcoords="offset points", color="#333333")
    if show_ylabel:
        ax.set_ylabel("gain vs baseline (dB)")
    if show_xlabel:
        ax.set_xlabel("protocol")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", default="results/protocol_matrix_fig1_wide.csv")
    ap.add_argument("--gain", default="results/model_vs_bicubic_gain.csv")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--fmt", default="pdf,png")
    args = ap.parse_args()

    if not os.path.isfile(args.wide):
        print(f"missing {args.wide}", file=sys.stderr)
        return 1
    style()
    wide = read_wide(args.wide)
    gain = read_gain(args.gain)

    fig, axes = plt.subplots(
        2, len(TASKS), figsize=(7.0, 3.9),
        gridspec_kw=dict(height_ratios=[2.35, 1.0], hspace=0.12, wspace=0.22))

    for j, (task, title, ref) in enumerate(TASKS):
        panel_top(axes[0][j], task, f"({chr(97 + j)}) {title}", ref, wide, gain,
                  show_ylabel=(j == 0), show_xlabel=False)
        # The low-light reference is the raw capture, not bicubic, so its
        # gains live on a 10x larger scale and must keep its own axis.
        sr_ylim = (-2.95, 1.60) if task != "lowlight_unet" else None
        panel_bottom(axes[1][j], task, gain, wide,
                     show_ylabel=(j == 0), show_xlabel=True, ylim=sr_ylim)

    handles = [
        Line2D([], [], color=C_MODEL, marker="o", ms=3.4, lw=1.3, label="model"),
        Line2D([], [], color=C_BASE, marker="s", ms=3.0, lw=1.1, mfc="white",
               mew=0.9, label="same-protocol baseline"),
        # P1 needs its own legend entry: it is drawn as a bare triangle at the
        # top edge, and a reader who does not know that will read it as a stray
        # marker rather than as a protocol 4-20 dB off the scale (task-dependent).
        Line2D([], [], color=C_MODEL, marker="^", ms=4.0, lw=0,
               label="P1 (off-scale)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=2.0)

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, "fig1_protocol_matrix")
    for fmt in args.fmt.split(","):
        fmt = fmt.strip()
        if not fmt:
            continue
        path = f"{base}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"[i] wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
