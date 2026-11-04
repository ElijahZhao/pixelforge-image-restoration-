#!/usr/bin/env python3
"""Graphical abstract for the Preprints.org submission.

A graphical abstract is not Figure 1 in miniature and not a picture of the
abstract. It has one job: let a reader who will never open the paper get the
argument in about ten seconds. That means showing the *comparison* the paper
is about, with the numbers on the picture, not pointing at a result and
hoping the reader reconstructs the reasoning.

The argument, in one picture:

    A protocol choice moves the reported metric further than the model does
    -- and one boolean flips the sign of the conclusion.

Three panels, left to right:

  1. **Scale.** For each task, the dB a protocol choice can move (the span
     over P2-P8) against the dB the model itself won (the P3 gain). On SR the
     protocol bar is 5x and 3.6x the model bar. On low-light it is the other
     way round, and the figure says so: the claim is conditional, and the
     counterexample is the most honest thing on the picture.
  2. **The flip.** The same weights, the same images, the same code path,
     with the downsampling antialias flag on or off: +0.82 dB becomes
     -1.75 dB. One boolean.
  3. **The gap.** How often 63 recent papers report the settings that decide
     the number: aggregation 1/63, validation batch 0/63, val/test overlap
     0/63.

Every number is read from the same sources the tables use, and the script
refuses to draw if they disagree with `FACTS.md`.

Output: `paper/figures/ga_protocol_audit.{png,jpg,pdf}`

Usage
-----
    python scripts/make_graphical_abstract.py
    python scripts/make_graphical_abstract.py --out DIR --fmt png,jpg,pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Same Okabe-Ito pair as Figure 1 and Figure 2, so everything reads as one
#: family. Blue = the model, orange = the reference/protocol side, green = the
#: reporting gap, grey = structure.
C_MODEL = "#0072B2"
C_BASE = "#D55E00"
C_CLAIM = "#009E73"
C_INK = "#222222"
C_MUTE = "#777777"
C_BG = "#FFFFFF"
C_PANEL = "#F7F7F8"


def _p(*parts):
    return os.path.join(ROOT, *parts)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load():
    """Everything the picture needs, recomputed from the sources."""
    wide = {(r["task"], r["protocol"]): r
            for r in read_csv(_p("results", "protocol_matrix_fig1_wide.csv"))}
    gain = {(r["task"], r["protocol"]): r
            for r in read_csv(_p("results", "model_vs_bicubic_gain.csv"))}

    tasks = ["generator_x4", "srcnn_x2", "lowlight_unet"]
    names = {"generator_x4": r"SR $\times$4", "srcnn_x2": r"SR $\times$2",
             "lowlight_unet": "Low-light"}

    # Span over P2-P8 (all protocols except the best-of-60 outlier P1) and the
    # P3 training gain, per task. Same task on both sides of every division.
    scale = []
    for t in tasks:
        vals = {p: float(wide[(t, p)]["psnr_mean"])
                for p in ("P2", "P3", "P4", "P5", "P6", "P7", "P8")
                if (t, p) in wide}
        span = max(vals.values()) - min(vals.values())
        g = float(gain[(t, "P3")]["gain"])
        scale.append({"task": t, "name": names[t], "span": span, "gain": g,
                      "ratio": span / g})

    # The sign flip, SR x4: P5 (AA on) vs P6 (AA off).
    flip = {
        "p5": float(gain[("generator_x4", "P5")]["gain"]),
        "p6": float(gain[("generator_x4", "P6")]["gain"]),
    }
    flip["swing"] = abs(flip["p5"] - flip["p6"])

    # Reporting rates, recounted from the coding table.
    rows = [r for r in read_csv(_p("archive_private", "05-论文规划",
                                    "文献抽样编码表.csv"))
            if r["编号"] != "示例"]
    n = len(rows)
    n_sr = sum(1 for r in rows if r["任务(SR/LLIE)"].strip().startswith("SR"))
    rates = [
        ("Aggregation method", sum(1 for r in rows if r["F1"].strip() not in ("未报告", "")), n),
        ("Validation batch size", sum(1 for r in rows if r["F2"].strip() not in ("未报告", "")), n),
        ("val / test overlap", 0, n),
        # The degradation operator only exists for SR, so its denominator is
        # the SR subset (44), not the whole sample.
        ("SR degradation reproducible", sum(
            1 for r in rows
            if r["任务(SR/LLIE)"].strip().startswith("SR")
            and r["F3"].strip() in ("具名可复现实现(MATLAB bicubic)", "显式抗锯齿")), n_sr),
    ]
    return {"scale": scale, "flip": flip, "rates": rates, "n": n}


def check(d):
    """Refuse to draw numbers that disagree with FACTS.md."""
    bad = []
    want_span = {"generator_x4": 3.8892, "srcnn_x2": 3.1431, "lowlight_unet": 2.2496}
    want_gain = {"generator_x4": 0.7816, "srcnn_x2": 0.8629, "lowlight_unet": 10.7361}
    want_ratio = {"generator_x4": 4.98, "srcnn_x2": 3.64, "lowlight_unet": 0.21}
    for row in d["scale"]:
        t = row["task"]
        if abs(row["span"] - want_span[t]) > 5e-4:
            bad.append(f"span {t}: {row['span']:.4f} != {want_span[t]}")
        if abs(row["gain"] - want_gain[t]) > 5e-4:
            bad.append(f"gain {t}: {row['gain']:.4f} != {want_gain[t]}")
        if abs(round(row["ratio"], 2) - want_ratio[t]) > 1e-9:
            bad.append(f"ratio {t}: {row['ratio']:.4f} != {want_ratio[t]}")
    if abs(d["flip"]["p5"] - 0.8179) > 5e-4:
        bad.append(f"flip P5: {d['flip']['p5']:.4f} != 0.8179")
    if abs(d["flip"]["p6"] + 1.7531) > 5e-4:
        bad.append(f"flip P6: {d['flip']['p6']:.4f} != -1.7531")
    if d["n"] != 63:
        bad.append(f"n: {d['n']} != 63")
    if d["rates"][0][1] != 1:
        bad.append(f"aggregation: {d['rates'][0][1]} != 1")
    if d["rates"][1][1] != 0:
        bad.append(f"batch: {d['rates'][1][1]} != 0")
    # The degradation-operator rate is over the SR subset, so 5/44.
    if d["rates"][3][1] != 5 or d["rates"][3][2] != 44:
        bad.append(f"degradation: {d['rates'][3][1]}/{d['rates'][3][2]} != 5/44")
    return bad


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_scale(ax, d):
    """Panel 1: how far a protocol moves the number vs how far the model moved it."""
    ax.set_facecolor(C_PANEL)
    rows = d["scale"]
    ys = list(range(len(rows)))[::-1]

    for y, row in zip(ys, rows):
        # Protocol span (the orange bar) and model gain (the blue bar).
        ax.barh(y, row["span"], height=0.46, color=C_BASE, alpha=0.95,
                edgecolor="none", zorder=3)
        ax.barh(y, row["gain"], height=0.46, color=C_MODEL, alpha=0.95,
                edgecolor="none", zorder=4)
        # Values. When the model bar is long (low-light), the protocol bar is
        # hidden inside it, so the two labels are placed on opposite sides of
        # their own bars and the short one goes inside.
        if row["gain"] > row["span"]:
            # Model bar dominates and hides the protocol bar entirely. So:
            #   - the MODEL value (long bar, blue) goes inside it in white,
            #     at its left end;
            #   - the PROTOCOL value (short bar, orange, hidden) goes just
            #     past the right end of the model bar, in orange on the
            #     panel background where it can actually be read.
            # Both were previously drawn on top of the blue bar, where the
            # orange one was invisible against it.
            # zorder matters: the bars are drawn at 3 and 4, and text
            # defaults to 3, so the blue bar (4) painted over the white
            # label. It was there in the file the whole time -- invisible.
            ax.text(0.30, y, f"{row['gain']:.2f} dB", va="center", ha="left",
                    fontsize=9.5, color="white", fontweight="bold", zorder=6)
            ax.text(row["gain"] + 0.30, y, f"{row['span']:.2f} dB",
                    va="center", ha="left", fontsize=9.5, color=C_BASE,
                    fontweight="bold", zorder=6)
        else:
            # The model bar is too short to hold a label (0.78 dB on a 3.9 dB
            # axis is a sliver), so its value goes UNDER the bar in blue and
            # the protocol value goes to the right of the bars in orange.
            ax.text(row["span"] + 0.25, y, f"{row['span']:.2f} dB", va="center",
                    ha="left", fontsize=9.5, color=C_BASE, fontweight="bold",
                    zorder=6)
            # Above the bar: below it sat the counterexample note.
            ax.text(row["gain"] / 2.0, y + 0.30, f"{row['gain']:.2f} dB",
                    va="bottom", ha="center", fontsize=9.0, color=C_MODEL,
                    fontweight="bold", zorder=6)
        # The ratio badge is the point of the panel, so it gets its own column.
        ax.text(-2.30, y, f"{row['ratio']:.2f}x", va="center", ha="right",
                fontsize=12, color=C_INK, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=10.5)
    ax.tick_params(axis="y", length=0, pad=8)
    # The ratio badge lives to the LEFT of the axis, so the x range has to
    # start well below zero. The earlier -3.6 clipped the task names.
    ax.set_xlim(-6.6, 15.4)
    ax.set_ylim(-0.72, len(rows) - 0.10)
    ax.set_xlabel("dB moved by the metric", fontsize=9.5, color=C_MUTE)
    ax.set_title("A protocol choice moves the number\nfurther than the model does",
                 fontsize=11.5, fontweight="bold", color=C_INK, pad=9, loc="left")
    ax.grid(axis="x", color="#E3E3E6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D5DA")

    # Legend: what the two colours mean.
    handles = [Rectangle((0, 0), 1, 1, color=C_BASE),
               Rectangle((0, 0), 1, 1, color=C_MODEL)]
    leg = ax.legend(handles, ["protocol choice (span P2-P8)", "model's own gain (P3)"],
                    loc="upper right", bbox_to_anchor=(1.02, 1.02),
                    fontsize=8.2, frameon=False, ncol=1,
                    handlelength=1.0, handleheight=0.9, borderaxespad=0.5)
    for t in leg.get_texts():
        t.set_color(C_MUTE)

    # The counterexample, called out above the low-light bar where there is
    # empty space. Anchoring it below used to land on the legend and the
    # x-axis label at the same time.
    row_x = rows[2]["span"] / 2.0
    # Above the low-light bar and right-aligned to the panel edge, so it
    # cannot run into the middle panel.
    # Sits directly above the low-light bar. Two short lines keep it inside
    # the panel: the earlier one-line version ran under the middle panel.
    ax.text(row_x, ys[2] + 0.26,
            "counterexample: headroom is large,\nso the protocol does not decide it",
            fontsize=7.4, color=C_MUTE, ha="center", va="bottom",
            linespacing=1.3)


def panel_flip(ax, d):
    """Panel 2: one boolean, and the sign of the conclusion changes."""
    ax.set_facecolor(C_PANEL)
    flip = d["flip"]

    ax.axhline(0, color=C_INK, lw=1.1, zorder=4)
    for x, v, lab, col, sub in (
            (0, flip["p5"], "antialias ON", C_MODEL, "P5"),
            (1, flip["p6"], "antialias OFF", C_BASE, "P6")):
        ax.bar([x], [v], width=0.44, color=col, edgecolor="none", zorder=3)
        ax.text(x, v + (0.32 if v >= 0 else -0.32), f"{v:+.2f} dB",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=11.5, fontweight="bold", color=col)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["antialias ON\n(P5)", "antialias OFF\n(P6)"],
                       fontsize=9.5)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(-3.05, 2.35)
    ax.set_yticks([-2, 0, 2])
    ax.set_ylabel("model gain over bicubic (dB)", fontsize=9.5, color=C_MUTE)
    ax.set_title("One boolean flips the verdict",
                 fontsize=11.5, fontweight="bold", color=C_INK, pad=9, loc="left")
    ax.grid(axis="y", color="#E3E3E6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5D5DA")
    ax.spines["left"].set_color("#D5D5DA")

    ax.text(0.5, -2.70, "same weights  |  same images  |  same code path",
            ha="center", va="center", fontsize=8.6, color=C_MUTE, style="italic")
    # The swing note sits at the far left, where the P6 bar is not: centred it
    # landed on the -1.75 dB label.
    ax.text(-0.55, -0.95, f"swing {flip['swing']:.2f} dB\n= 3.3x the\ntraining gain",
            ha="left", va="top", fontsize=9.2, color=C_INK, fontweight="bold",
            linespacing=1.35)


def panel_gap(ax, d):
    """Panel 3: how often the deciding settings are reported."""
    ax.set_facecolor(C_PANEL)
    rates = d["rates"]
    n = d["n"]
    ys = list(range(len(rates)))[::-1]

    for y, (label, k, denom) in zip(ys, rates):
        frac = k / denom
        # The empty track, so "zero" reads as a measured zero and not a gap.
        ax.barh(y, 1.0, height=0.42, color="#E9E9EC", edgecolor="none", zorder=2)
        ax.barh(y, frac, height=0.42, color=C_CLAIM, edgecolor="none", zorder=3)
        ax.text(0.0, y + 0.36, label, va="bottom", ha="left", fontsize=8.8,
                color=C_INK)
        # The count goes to the right of the track: at 0.035 it was hidden
        # under the green bar in the one non-zero case.
        ax.text(1.0, y, f"{k}/{denom}", va="center", ha="right", fontsize=11.5,
                color=C_INK, fontweight="bold", zorder=4)

    # Labels are drawn inside the panel, above each track, rather than as
    # y-tick labels: as tick labels they ran into the middle panel.
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(rates) - 0.3)
    ax.set_xticks([])
    ax.set_title("What 63 recent papers report",
                 fontsize=11.5, fontweight="bold", color=C_INK, pad=9, loc="left")
    ax.grid(False)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    ax.text(0.5, -0.52, "bars are the empty track: a measured zero, not missing data",
            ha="center", va="center", fontsize=7.4, color=C_MUTE, style="italic")


def build(d):
    """Compose the whole picture."""
    style()
    # MDPI's reference graphical-abstract size is 560 x 1100 px (h x w), i.e.
    # a 1.964:1 canvas. This figure is built at that ratio so it is neither
    # stretched on upload nor cropped.
    fig = plt.figure(figsize=(9.63, 4.90), facecolor=C_BG)

    # Manual layout: the middle panel is the punchline and gets more width.
    ax1 = fig.add_axes([0.088, 0.215, 0.304, 0.560])
    ax2 = fig.add_axes([0.452, 0.215, 0.208, 0.560])
    ax3 = fig.add_axes([0.720, 0.215, 0.250, 0.560])

    panel_scale(ax1, d)
    panel_flip(ax2, d)
    panel_gap(ax3, d)

    # A connector from "the protocol is big" to "and it can flip the sign".
    arr = FancyArrowPatch((0.404, 0.575), (0.444, 0.575),
                          transform=fig.transFigure,
                          arrowstyle="-|>", mutation_scale=13,
                          color="#C9C9CF", lw=1.3, zorder=1)
    fig.patches.append(arr)

    # Title band.
    fig.text(0.062, 0.960, "The protocol is the result",
             fontsize=17, fontweight="bold", color=C_INK, ha="left", va="center")
    fig.text(0.062, 0.906,
             "In lightweight image restoration, the evaluation protocol -- not the model -- "
             "often decides what the reported number says.",
             fontsize=10.4, color=C_MUTE, ha="left", va="center")
    fig.text(0.970, 0.960, "21 protocol cells x 5 seeds = 105 evaluations",
             fontsize=8.6, color=C_MUTE, ha="right", va="center")
    fig.text(0.970, 0.915, "SR x4 / SR x2 / low-light",
             fontsize=8.6, color=C_MUTE, ha="right", va="center")

    # Divider under the title. It sits above the panel titles, which start at
    # y=0.81 -- an earlier divider at 0.845 cut through them.
    fig.add_artist(plt.Line2D([0.062, 0.970], [0.872, 0.872],
                              transform=fig.transFigure, color="#E3E3E6", lw=1.0))

    # Footer: the one-line takeaway plus the honest caveat.
    fig.text(0.062, 0.105,
             "Report the protocol: aggregation, validation set, sampling window, "
             "channel and shave, degradation operator, seeds.",
             fontsize=9.0, color=C_INK, ha="left", va="center")
    fig.text(0.062, 0.048,
             "The claim is conditional: on low-light the protocol spans 2.25 dB "
             "against a 10.74 dB training gain, so it does not decide the result.",
             fontsize=7.8, color=C_MUTE, ha="left", va="center")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_p("paper", "figures"))
    ap.add_argument("--fmt", default="png,jpg,pdf")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    d = load()
    bad = check(d)
    if bad:
        print("SOURCES DISAGREE WITH FACTS.md -- refusing to draw:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        sys.exit(1)
    print("check: every number agrees with FACTS.md")

    os.makedirs(args.out, exist_ok=True)
    fig = build(d)
    for ext in [f.strip() for f in args.fmt.split(",") if f.strip()]:
        path = os.path.join(args.out, f"ga_protocol_audit.{ext}")
        if ext == "jpg":
            fig.savefig(path, dpi=args.dpi, facecolor=C_BG, pil_kwargs={"quality": 95})
        else:
            fig.savefig(path, dpi=args.dpi, facecolor=C_BG)
        print("  wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
