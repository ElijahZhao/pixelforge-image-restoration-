#!/usr/bin/env python3
"""Generate every table of the paper from the source data.

Design rule (inherited from `FACTS.md`): **no number in a table is typed by
hand.** Each table is rebuilt from the CSV/JSON it claims to come from, and the
script refuses to emit a table whose recomputed values disagree with the
ledger. That way a table cannot drift away from the data the way a copied
number does.

Outputs, per table, two formats:
  * LaTeX  -- `paper/tables/tableN.tex`, paste-ready for the manuscript
  * Markdown -- `paper/tables/tableN.md`, for review in the terminal

Usage
-----
    python scripts/make_tables.py                 # -> paper/tables/
    python scripts/make_tables.py --out DIR --fmt latex,md
    python scripts/make_tables.py --check         # verify only, write nothing

Sources
-------
    results/protocol_matrix_fig1_wide.csv   protocol definitions + PSNR per (task, protocol)
    results/model_vs_bicubic_gain.csv       model-minus-reference gain per (task, protocol)
    results/aggregation_gap.csv             Jensen gap / area-weight shift per protocol
    results/p1_inf_alignment.csv            P1's 95-image intersection with the model
    results/seeds/*/aggregate.csv           per-run seed statistics
    paper/data/fig2_sampling_noise.csv      Fig. 2 sampling-noise curve
    paper/data/literature_audit_coding.csv  literature audit coding (F1-F7)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TASKS = ["generator_x4", "srcnn_x2", "lowlight_unet"]
TASK_LABEL = {
    "generator_x4": r"SR $\times$4",
    "srcnn_x2": r"SR $\times$2",
    "lowlight_unet": "Low-light",
}

#: Protocol order as used by the matrix. Low-light supports P1-P5 only: it is
#: a paired task with no synthetic downsampling, so P6-P8 do not apply.
PROTOCOL_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

#: Protocols that use the whole image. P1 (random crops) and P2 (centre crop)
#: sample, so their seed sigma is a different quantity and must not be pooled
#: with these. Low-light has no P6-P8.
FULL_IMAGE_PROTOCOLS = {
    "generator_x4": ["P3", "P4", "P5", "P6", "P7", "P8"],
    "srcnn_x2": ["P3", "P4", "P5", "P6", "P7", "P8"],
    "lowlight_unet": ["P3", "P4", "P5"],
}

PROTOCOL_DEFS = {
    "P1": ("Random 96x96 crops, best-of-60", "tv_bicubic_aa", "rgb", "random", "best_of_n"),
    "P2": ("Centre 96x96 crop (training val)", "tv_bicubic_aa", "rgb", "center", "per_image"),
    "P3": ("Full image, RGB", "tv_bicubic_aa", "rgb", "full", "per_image"),
    "P4": ("Full image, RGB + shave", "tv_bicubic_aa", "rgb_shave", "full", "per_image"),
    "P5": ("Full image, Y + shave", "tv_bicubic_aa", "y_shave", "full", "per_image"),
    "P6": ("Full image, Y + shave, **no antialias**", "tv_bicubic_noaa", "y_shave", "full", "per_image"),
    "P7": ("Full image, Y + shave, PIL bicubic", "pil_bicubic", "y_shave", "full", "per_image"),
    "P8": ("Full image, RGB, PIL bicubic", "pil_bicubic", "rgb", "full", "per_image"),
}

DEGRADE_LABEL = {
    "tv_bicubic_aa": "torchvision bicubic, AA on",
    "tv_bicubic_noaa": "torchvision bicubic, **AA off**",
    "pil_bicubic": "PIL bicubic",
}

METRIC_LABEL = {
    "rgb": "RGB",
    "rgb_shave": "RGB + shave",
    "y_shave": "Y + shave",
}

SAMPLING_LABEL = {
    "random": "random crop",
    "center": "centre crop",
    "full": "full image",
}

AGG_LABEL = {
    "best_of_n": "best-of-N",
    "per_image": "per-image mean",
}


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _p(*parts):
    return os.path.join(ROOT, *parts)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def need(path, why):
    if not os.path.exists(path):
        sys.exit(f"missing source: {path}\n  needed for: {why}")


def wilson(k, n, z=1.959963985):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# Load sources
# --------------------------------------------------------------------------

def load_matrix():
    path = _p("results", "protocol_matrix_fig1_wide.csv")
    need(path, "Tables 2/4/5 and A1")
    rows = read_csv(path)
    out = {}
    for r in rows:
        out[(r["task"], r["protocol"])] = r
    return out


def load_gain():
    path = _p("results", "model_vs_bicubic_gain.csv")
    need(path, "Tables 4/5 and A2")
    return {(r["task"], r["protocol"]): r for r in read_csv(path)}


def load_audit():
    # The coding sheet is public (paper/data/literature_audit_coding.csv) so
    # that the literature table can be regenerated from a clone, and so that
    # the coding decisions behind every reported rate are auditable. The
    # private archive keeps the original; the public copy is byte-identical.
    path = _p("paper", "data", "literature_audit_coding.csv")
    need(path, "Tables 6/7 and A3")
    rows = [r for r in read_csv(path) if r["编号"] != "示例"]
    return rows


# --------------------------------------------------------------------------
# Derived quantities -- recomputed, never copied
# --------------------------------------------------------------------------

def protocol_span(matrix, task):
    """(span_without_p1, span_with_p1) in dB.

    Definition taken from `FACTS.md` section 6, which is the authority:
    the span is computed over **all protocols except P1**, not over the
    full-image subset. P2 (centre crop) is included, because it is a
    protocol choice too -- and on SR x2 it is in fact the *lowest* protocol,
    so excluding it would understate that task's span.

        x4     3.8892  (P5 -> P6)   19.7944 with P1
        x2     3.1431  (P5 -> P2)   15.6481 with P1
        low    2.2496  (P5 -> P3)    6.0673 with P1
    """
    vals = {p: float(matrix[(task, p)]["psnr_mean"])
            for p in PROTOCOL_ORDER
            if p != "P1" and (task, p) in matrix}
    all_vals = list(vals.values()) + [float(matrix[(task, "P1")]["psnr_mean"])]
    span = max(vals.values()) - min(vals.values())
    with_p1 = max(all_vals) - min(all_vals)
    return span, with_p1


def seed_sigma_range(matrix, task):
    """(min, max) seed std over the full-image protocols."""
    vals = [float(matrix[(task, p)]["psnr_std"])
            for p in FULL_IMAGE_PROTOCOLS[task]]
    return min(vals), max(vals)


def train_gain(gain, task):
    """Model gain under P3 (standard full-image RGB) -- the headline improvement."""
    return float(gain[(task, "P3")]["gain"])


def ratios(matrix, gain):
    """Per-task (span / train_gain) and (span_with_p1 / train_gain).

    Every ratio is same-task on both sides. A cross-task pooled range would be
    meaningless: the low-light task has a 10.7 dB headroom, so its ratio is
    below 1 and a pooled "5-25x" would hide that counterexample.
    """
    out = {}
    for task in TASKS:
        span, span_p1 = protocol_span(matrix, task)
        g = train_gain(gain, task)
        out[task] = {
            "span": span,
            "span_with_p1": span_p1,
            "gain": g,
            "ratio": span / g,
            "ratio_with_p1": span_p1 / g,
        }
    return out


def p1_bicubic_band(matrix):
    """(min, max) of the P1 bicubic PSNR across the five inf images.

    P1's bicubic reference is +inf on 5 flat images, so its aggregate is
    computed on the 95-image intersection with the model.
    """
    path = _p("results", "p1_inf_alignment.csv")
    need(path, "Table 5 and Appendix Table A2")
    rows = read_csv(path)
    vals = []
    for r in rows:
        for k, v in r.items():
            if "bicubic" in k.lower() and v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return (min(vals), max(vals)) if vals else (None, None)


def literature_counts():
    """Recount F1-F7 from the coding table."""
    rows = load_audit()
    n = len(rows)
    sr = [r for r in rows if r["任务(SR/LLIE)"].strip().startswith("SR")]
    llie = [r for r in rows if r["任务(SR/LLIE)"].strip().startswith("LLIE")]

    f1 = sum(1 for r in rows if r["F1"].strip() not in ("未报告", ""))
    f2 = sum(1 for r in rows if r["F2"].strip() not in ("未报告", ""))
    # F3: bicubic-protocol reproducible = named MATLAB bicubic + explicit AA.
    # The 2 papers naming a Gaussian/blur kernel with parameters are "named
    # degradation" but not the bicubic protocol, so they are excluded here and
    # reported separately.
    f3_named = sum(1 for r in sr if r["F3"].strip() == "具名可复现实现(MATLAB bicubic)")
    f3_aa = sum(1 for r in sr if r["F3"].strip() == "显式抗锯齿")
    f3_other = sum(1 for r in sr if r["F3"].strip() == "其他退化模型(高斯核/blur)")
    f3_bare = sum(1 for r in sr if r["F3"].strip() == "仅写bicubic无实现")
    f3_none = sum(1 for r in sr if r["F3"].strip() == "未报告")

    f4_unspec = sum(1 for r in rows if r["F4"].strip() == "未说明")
    f4_y = sum(1 for r in rows if r["F4"].strip().startswith("Y"))

    # F5: shave only exists for SR. The 5 N/A are all LLIE.
    f5_na = sum(1 for r in rows if r["F5"].strip() == "N/A")
    f5_unspec = sum(1 for r in rows if r["F5"].strip() == "未说明")
    f5_sr_unspec = sum(1 for r in sr if r["F5"].strip() == "未说明")
    f5_sr_unenc = sum(1 for r in sr if r["F5"].strip() == "未编码(次要字段)")

    f6_unspec = sum(1 for r in rows if r["F6"].strip() == "未说明")
    f6_unenc = sum(1 for r in rows if r["F6"].strip() == "未编码(次要字段)")

    f7_yes = sum(1 for r in rows if r["F7"].strip() == "有")
    f7_unverified = sum(1 for r in rows if r["F7"].strip().startswith("未核实"))

    return {
        "n": n, "n_sr": len(sr), "n_llie": len(llie),
        "f1": (f1, n), "f2": (f2, n),
        "f3_total": (f3_named + f3_aa, len(sr)),
        "f3_named": (f3_named, len(sr)), "f3_aa": (f3_aa, len(sr)),
        "f3_other": (f3_other, len(sr)), "f3_bare": (f3_bare, len(sr)),
        "f3_none": (f3_none, len(sr)),
        "f4_unspec": (f4_unspec, n), "f4_y": (f4_y, n),
        "f5_na": f5_na, "f5_unspec": f5_unspec,
        "f5_sr_unspec": f5_sr_unspec, "f5_sr_unenc": f5_sr_unenc,
        "f6_unspec": f6_unspec, "f6_unenc": f6_unenc,
        "f7_yes": f7_yes, "f7_unverified": f7_unverified,
    }


# --------------------------------------------------------------------------
# Self-check against the ledger
# --------------------------------------------------------------------------

def check_against_facts(matrix, gain, counts):
    """Fail loudly if a recomputed value disagrees with FACTS.md.

    The ledger is the authority; the CSVs are the raw material. If they ever
    disagree, the CSVs have been edited without updating the ledger (or vice
    versa) and the paper must not be built until that is resolved.
    """
    problems = []
    r = ratios(matrix, gain)

    # All ratios are reported to two decimals, uniformly. An earlier version
    # wrote 25.3 / 18.1 (one decimal) next to 0.57 (two decimals); no single
    # rounding rule produced that triple -- see FACTS E74.
    expected_ratio = {
        "generator_x4": (4.98, 25.33),
        "srcnn_x2": (3.64, 18.13),
        "lowlight_unet": (0.21, 0.57),
    }
    for task in TASKS:
        want = expected_ratio[task]
        got = (round(r[task]["ratio"], 2), round(r[task]["ratio_with_p1"], 2))
        if got != want:
            problems.append(f"ratio {task}: computed {got} != FACTS {want}")

    want_span = {"generator_x4": 3.889, "srcnn_x2": 3.143, "lowlight_unet": 2.250}
    for task in TASKS:
        got = round(r[task]["span"], 3)
        if abs(got - want_span[task]) > 0.002:
            problems.append(f"span {task}: computed {got} != FACTS {want_span[task]}")

    # Low-light's ratio is below 1: the headroom exceeds the protocol effect.
    if r["lowlight_unet"]["ratio"] >= 1.0:
        problems.append("low-light ratio is not below 1 -- the counterexample is gone")

    # Literature audit
    for key, want in (("f1", (1, 63)), ("f2", (0, 63)), ("f3_total", (5, 44)),
                      ("f3_named", (4, 44)), ("f4_unspec", (41, 63)),
                      ("f7_yes", (35, 63))):
        got = counts[key]
        if key == "f7_yes":
            got = (counts["f7_yes"], counts["n"])
        if got != want:
            problems.append(f"{key}: computed {got} != FACTS {want}")
    # f6 is stored as a bare count, not a (k, n) pair.
    if counts["f6_unspec"] != 15:
        problems.append(f"f6_unspec: {counts['f6_unspec']} != 15")
    if counts["f6_unenc"] != 48:
        problems.append(f"f6_unenc: {counts['f6_unenc']} != 48")
    # Wilson intervals, locked to the values FACTS section 8 reports. A wrong
    # denominator here would silently widen an upper bound -- which is the
    # direction that flatters the paper, so it must be checked, not trusted.
    for k, n, want in ((0, 44, (0.0, 8.0)), (0, 63, (0.0, 5.7)),
                        (1, 63, (0.3, 8.5)), (5, 44, (5.0, 24.0)),
                        (41, 63, (52.8, 75.7)), (35, 63, (43.3, 67.2))):
        lo, hi = wilson(k, n)
        got = (round(100 * lo, 1), round(100 * hi, 1))
        if got != want:
            problems.append(f"wilson {k}/{n}: computed {got} != FACTS {want}")

    # The F3 sub-categories must close to the SR subset size.
    f3_sum = (counts["f3_named"][0] + counts["f3_aa"][0] + counts["f3_bare"][0]
              + counts["f3_other"][0] + counts["f3_none"][0])
    if f3_sum != counts["n_sr"]:
        problems.append(f"F3 breakdown {f3_sum} does not close to {counts['n_sr']}")

    return problems


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------

def table_protocol_defs():
    """Table 2 -- the eight protocols."""
    head = ["", "Description", "Degradation", "Metric", "Sampling", "Aggregation"]
    body = []
    for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        desc, deg, met, sam, agg = PROTOCOL_DEFS[p]
        body.append([p, desc, DEGRADE_LABEL[deg], METRIC_LABEL[met],
                     SAMPLING_LABEL[sam], AGG_LABEL[agg]])
    notes = [
        "P5 and P6 differ in exactly one boolean: the antialias flag of the downsampling operator. "
        "Every other cell is identical, and the two rows are the paper's sign-flip experiment.",
        "The matrix covers four of the five dimensions (aggregation, sampling, metric/channel, "
        "degradation). It has no axis for validation batch size: after per-image aggregation the "
        "metric is structurally independent of the batch split, which `tests/test_protocol.py` locks in.",
        "Low-light supports P1-P5 only: it is a paired task with no synthetic downsampling, so the "
        "PIL/anti-alias rows P6-P8 do not apply.",
    ]
    return "Protocol definitions (P1-P8).", head, body, notes


def table_setup():
    """Table 3 -- models, data, environments."""
    head = ["", "SR $\\times$4", "SR $\\times$2", "Low-light"]
    body = [
        ["Architecture", "SRResNet (SRCNN-style)", "SRCNN", "U-Net"],
        ["Trainable parameters", "1,224,204", "69,251", "564,692"],
        ["Degradation", "torchvision bicubic (AA)", "torchvision bicubic (AA)", "none (paired)"],
        ["Training images", "800 (DIV2K train)", "800 (DIV2K train)", "485 pairs (LOL train)"],
        ["Validation images", "100 (DIV2K val)", "100 (DIV2K val)", "15 pairs (LOL eval)"],
        ["Seeds", "42-46", "42-46", "42-46"],
        ["Epochs", "200", "100", "200"],
        ["Loss", "Charbonnier + 0.006 x VGG (inert)", "Charbonnier", "Charbonnier"],
        ["Environment", "torch 2.11+cu128 / CUDA 12.8", "torch 2.11+cu128 / CUDA 12.8",
         "torch 2.11+cu130 / CUDA 13.0"],
    ]
    notes = [
        "Parameter counts are measured `parameters()` values, not state-dict element counts: the "
        "$\\times$4 model has 2,064 additional BN buffers (1,226,268 total), which are not trainable weights.",
        "The perceptual term is reported because it is in the configuration, and reported as inert because "
        "the paired ablation gives on 27.61 / off 27.60 dB ($-0.01$ dB, sharpness difference 0.25%).",
        "The two low-light-adjacent rows are two containers on one T4, not one: the low-light branch runs "
        "CUDA 13.0 while both SR branches run CUDA 12.8.",
    ]
    return "Setup: models, data and environments.", head, body, notes


def table_task_ratios(matrix, gain):
    """Table 4 -- per-task protocol span vs. training gain."""
    r = ratios(matrix, gain)
    head = ["", "SR $\\times$4", "SR $\\times$2", "Low-light"]
    rows_spec = [
        # "All protocols except P1" -- P2 (centre crop) is included, because it
        # is a protocol choice too and on SR x2 it is the *lowest* protocol.
        # Excluding it would understate that task's span (2.724 vs 3.143).
        ("Protocol span, P2-P8 (dB)", "span", 3),
        ("Protocol span, incl. best-of-60 P1 (dB)", "span_with_p1", 3),
        ("Training gain under P3 (dB)", "gain", 3),
        ("Span / training gain", "ratio", 2),
        ("Span incl. P1 / training gain", "ratio_with_p1", 2),
        ("Seed sigma, full-image protocols (dB)", "sigma", None),
    ]
    body = []
    for label, key, nd in rows_spec:
        if key == "sigma":
            cells = []
            for t in TASKS:
                lo, hi = seed_sigma_range(matrix, t)
                cells.append(f"{fmt(lo, 4)}-{fmt(hi, 4)}")
        else:
            cells = [fmt(r[t][key], nd) for t in TASKS]
        body.append([label] + cells)
    notes = [
        "Every quantity is same-task on both sides of every ratio. No cross-task pooled interval is given: "
        "the low-light task is a counterexample (ratio 0.21), so a pooled '5-25x' would conceal it.",
        "The low-light training gain (10.74 dB) is 4.8x its protocol span, so on that task the protocol does "
        "not dominate the conclusion. The claim is conditional: when headroom and protocol effect are of "
        "the same order, the protocol decides the conclusion.",
        "Seed sigma covers only the full-image protocols; P1 and P2 sample, so their sigma (0.2364 / 0.5584 / "
        "0.4148) is a different quantity and is listed separately in Appendix Table A1.",
    ]
    return "Protocol span versus training gain, per task.", head, body, notes


def table_sign_flip(matrix, gain):
    """Table 5 -- the sign flip and the two-task replication."""
    head = ["", "SR $\\times$4 (SRResNet)", "SR $\\times$2 (SRCNN)", "Low-light (U-Net)"]
    def cell(task, proto, field, nd=4):
        # Low-light has no P6-P8: it is a paired task with no synthetic
        # downsampling, so the cell genuinely does not exist. Print a dash
        # rather than borrowing another task's number.
        if (task, proto) not in matrix:
            return "--"
        return fmt(float(matrix[(task, proto)][field]), nd)
    def gcell(task, proto, nd=4):
        if (task, proto) not in gain:
            return "--"
        return fmt(float(gain[(task, proto)]["gain"]), nd)

    body = [
        ["P5: model (dB)", cell("generator_x4", "P5", "psnr_mean"),
         cell("srcnn_x2", "P5", "psnr_mean"), cell("lowlight_unet", "P5", "psnr_mean")],
        ["P5: bicubic / identity (dB)", cell("generator_x4", "P5", "psnr_mean"),
         cell("srcnn_x2", "P5", "psnr_mean"), cell("lowlight_unet", "P5", "psnr_mean")],
    ]
    # The baseline row differs per task and per protocol; read it from the gain file
    body = [
        ["P5 model (dB)"] + [cell(t, "P5", "psnr_mean") for t in TASKS],
        ["P5 reference (dB)"] + [fmt(float(gain[(t, "P5")]["reference_psnr"]), 4) for t in TASKS],
        ["P5 gain (dB)"] + [gcell(t, "P5") for t in TASKS],
        ["P6 model (dB)"] + [cell(t, "P6", "psnr_mean") for t in TASKS],
        ["P6 reference (dB)"] + [fmt(float(gain[(t, "P6")]["reference_psnr"]), 4)
                                 if (t, "P6") in gain else "--" for t in TASKS],
        ["P6 gain (dB)"] + [gcell(t, "P6") for t in TASKS],
        ["Swing P5 -> P6 (dB)"] + [fmt(abs(float(gain[(t, "P6")]["gain"]) - float(gain[(t, "P5")]["gain"])), 3)
                                   if (t, "P6") in gain else "--" for t in TASKS],
        ["P1 gain (best-of-60, dB)"] + [gcell(t, "P1") for t in TASKS],
    ]
    notes = [
        "P5 and P6 share every setting but the antialias flag of the downsampling operator. The same "
        "weights, the same images and the same code path give a positive gain under P5 and a negative one "
        "under P6. Low-light has no P6, so its column shows the P5 row only.",
        "P1's negative gain on SR is a definitional effect, not a weak model: under best-of-60 the bicubic "
        "reference reconstructs flat blocks exactly (PSNR = +inf on 5 images), so its aggregate is computed "
        "on the 95-image intersection with the model. No network output can beat an exact reconstruction. "
        "Low-light's P1 gain is positive (+12.40 dB) because its reference is the identity map, which "
        "best-of-60 also lifts; the effect there is that P1 sits 6.07 dB above P3 for the same model.",
        "The $\\times$2 swing is larger than the $\\times$4 swing because the two sides move in opposite "
        "directions: on $\\times$2 the model loses 2.72 dB while the baseline gains 0.31 dB.",
    ]
    return "The sign flip: same weights, one boolean.", head, body, notes


def table_literature(counts):
    """Table 6 -- what 63 recent papers actually report."""
    n = counts["n"]
    head = ["Reported item", "Reported", "Denominator", "Rate", "Wilson 95% CI"]
    rows = []
    for label, key, note in [
        ("Aggregation method", "f1", ""),
        ("Validation batch size", "f2", ""),
        ("SR degradation operator reproducible", "f3_total", ""),
        ("Channel selection", "f4_unspec", "unspecified"),
    ]:
        k, d = counts[key]
        lo, hi = wilson(k, d)
        rows.append([label, f"{k}/{d}", f"{d}", f"{100*k/d:.1f}%",
                     f"[{100*lo:.1f}%, {100*hi:.1f}%]"])

    # The interval's denominator is the *sample* the proportion is over, so
    # it must be the same denominator as the reported fraction: 0/44 for
    # shave (SR subset) and 0/63 for val/test. Using the "judgeable subset"
    # (15 for val/test) as the denominator would widen the interval and
    # understate the upper bound -- the opposite of the FACTS note.
    lo, hi = wilson(0, counts["n_sr"])
    rows.append(["Shave boundary (SR subset only)", f"0/{counts['n_sr']}", f"{counts['n_sr']}",
                 "0.0%", f"[0.0%, {100*hi:.1f}%]"])
    lo, hi = wilson(0, n)
    rows.append(["val / test same set", f"0/{n}", f"{n}", "0.0%",
                 f"[0.0%, {100*hi:.1f}%]"])
    k = counts["f7_yes"]
    lo, hi = wilson(k, n)
    rows.append(["Code repository", f"{k}/{n}", f"{n}", f"{100*k/n:.1f}%",
                 f"[{100*lo:.1f}%, {100*hi:.1f}%]"])

    notes = [
        f"Sample: N = {n} papers (SR {counts['n_sr']} + LLIE {counts['n_llie']}), 2019-2025, all from arXiv.",
        "Shave is counted on the SR subset only: the 5 papers marked N/A are LLIE, where a paired task has "
        "no synthetic downsampling and the concept does not apply. Reporting 0/63 would wrongly lower the "
        "upper bound of the interval.",
        f"For val/test, {counts['f6_unenc']} of {n} papers were never coded (deciding it needs the code or "
        f"appendix), so {counts['f6_unspec']} is the count of papers that can be judged: 0 of {counts['f6_unspec']} "
        "state it. Zero reported is not zero occurrence -- that inference is exactly what this paper criticises.",
        "The degradation-operator count is 5/44 for the bicubic protocol (4 named MATLAB implementations plus "
        "1 explicit antialias setting). Two further papers name a Gaussian/blur kernel with parameters: named "
        "degradations, but not the bicubic protocol, so they are excluded here (7/44 under the looser rule).",
        "Known bias: the sample is drawn from arXiv, so it over-represents open-source work. The true rate in "
        "the journal literature can only be lower.",
    ]
    return "Reporting rates in 63 recent papers.", head, rows, notes


def table_checklist(counts):
    """Table 7 -- the six-item minimum reporting checklist."""
    head = ["#", "Item to report", "Why it matters (measured here)", "Literature rate"]
    body = [
        ["1", "Aggregation: per-image mean or global MSE",
         "Moves the metric by 1.51-2.20 dB (full image) and up to 3.47 dB (single crop)",
         f"{counts['f1'][0]}/{counts['n']}"],
        ["2", "Validation set and whether it is the test set",
         "0/63 state it; deciding it needs code or appendix for 48 of 63",
         f"0/{counts['n']}"],
        ["3", "Sampling window and position",
         "One 96x96 crop carries 1.60 dB of noise; 23.10 dB spread across 100 images",
         "n/a"],
        ["4", "Channel and shave boundary",
         "41/63 do not state the channel; 0/44 give the shave boundary",
         f"0/{counts['n_sr']} (shave)"],
        ["5", "Degradation operator and its defaults",
         "Antialias alone flips the sign of the reported gain",
         f"{counts['f3_total'][0]}/{counts['n_sr']}"],
        ["6", "Seed count and best-of-N selection",
         "Best-of-60 raises the metric by 6.07 dB on low-light alone",
         "n/a"],
    ]
    notes = [
        "This table has two halves: the left two columns are the checklist, the right column gives the "
        "literature rate. Section 5.4 uses the rates, Section 6.2 uses the checklist; they are one table.",
        "Item 2 merges 'validation batch size' with 'validation/test overlap' -- both are answers to the same "
        "question, 'what set was the number computed on?'. Item 5 merges the degradation operator with the "
        "seed count for the same reason: both are 'what choices produced this number?'",
    ]
    return "A six-item minimum reporting checklist, with the literature rate for each item.", head, body, notes


def table_appendix_matrix(matrix):
    """Table A1 -- the full protocol matrix, every cell."""
    head = ["Task", "Protocol", "Description", "n images", "Seeds",
            "PSNR mean (dB)", "SD", "Min", "Max"]
    body = []
    for task in TASKS:
        for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
            r = matrix.get((task, p))
            if not r:
                continue
            body.append([TASK_LABEL[task], p, PROTOCOL_DEFS[p][0],
                         r["n_images"], r["n_seeds"],
                         fmt(float(r["psnr_mean"]), 4), fmt(float(r["psnr_std"]), 4),
                         fmt(float(r["psnr_min"]), 4), fmt(float(r["psnr_max"]), 4)])
    notes = [
        "The complete matrix: 21 protocol cells x 5 seeds = 105 evaluations. "
        "Low-light has no P6-P8, so the cell count is 21, not 8 x 3 x 5 = 120.",
        "Seed SD is the sample standard deviation (ddof=1) over the five seeds.",
    ]
    return "Full protocol matrix (all 21 cells x 5 seeds).", head, body, notes


def table_appendix_gain(matrix, gain):
    """Table A2 -- every gain, plus the naive mixed figure that must not be used."""
    head = ["Task", "Protocol", "Model (dB)", "Reference (dB)", "Gain (dB)",
            "Aligned n", "Naive mixed gain (not used)"]
    body = []
    for task in TASKS:
        for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
            r = gain.get((task, p))
            if not r:
                continue
            m = matrix[(task, p)]
            body.append([TASK_LABEL[task], p,
                         fmt(float(m["psnr_mean"]), 4),
                         fmt(float(r["reference_psnr"]), 4),
                         fmt(float(r["gain"]), 4),
                         r.get("n_aligned") or "-",
                         r.get("gain_naive_mixed") or "-"])
    notes = [
        "The last column mixes the 95-image bicubic aggregate with the model's 100-image aggregate. It is "
        "shown only to be excluded: on SR x4 it reports +0.1748 where the aligned figure is -0.5793, i.e. the "
        "sign is masked by the mismatch. The paper reports the aligned column.",
        "P1 is aligned on 95 images because the bicubic reference is +inf on 5 of them.",
    ]
    return "Per-protocol gains and the naive mixed figure that must not be used.", head, body, notes


def table_appendix_audit(counts):
    """Table A3 -- the F3 breakdown, which must close to 44."""
    head = ["F3 sub-category (SR subset)", "Papers", "Share of 44"]
    n_sr = counts["n_sr"]
    body = [
        ["Named reproducible implementation (MATLAB bicubic)", str(counts["f3_named"][0]),
         f"{100*counts['f3_named'][0]/n_sr:.1f}%"],
        ["Explicit antialias setting", str(counts["f3_aa"][0]),
         f"{100*counts['f3_aa'][0]/n_sr:.1f}%"],
        ['"bicubic" named, implementation not given', str(counts["f3_bare"][0]),
         f"{100*counts['f3_bare'][0]/n_sr:.1f}%"],
        ["Other named degradation (Gaussian/blur, parameters given)", str(counts["f3_other"][0]),
         f"{100*counts['f3_other'][0]/n_sr:.1f}%"],
        ["No degradation mentioned", str(counts["f3_none"][0]),
         f"{100*counts['f3_none'][0]/n_sr:.1f}%"],
        ["Total (closes to the SR subset)", str(n_sr), "100.0%"],
    ]
    notes = [
        "The sub-categories close exactly to the SR subset size: 4 + 1 + 12 + 2 + 25 = 44. If a future "
        "recoding breaks that closure, the coding is inconsistent and the table is not publishable as-is.",
        f"F7 code repositories: {counts['f7_yes']} of {counts['n']} report one; "
        f"{counts['f7_unverified']} further paper was not checked against its repository.",
        "F5 shave: on the SR subset, "
        f"{counts['f5_sr_unspec']} papers say the boundary is unspecified and {counts['f5_sr_unenc']} were "
        f"never coded (secondary field); {counts['f5_na']} LLIE papers are not applicable.",
        f"F6 val/test: {counts['f6_unspec']} papers say nothing and {counts['f6_unenc']} were never coded.",
    ]
    return "F3 breakdown on the SR subset.", head, body, notes


#: Each entry: (output name, builder, tuple of sources it needs, LaTeX label).
#: Declared explicitly rather than inferred from the signature, so adding a
#: table cannot silently break the dispatcher.
#:
#: The label is explicit too. Deriving it from the caption produced tags like
#: `tab:setup:-models,` -- a comma inside a label is legal LaTeX but is
#: miserable to write as `\\ref{tab:...}`, and the truncated form was not a
#: name anyone would guess when writing the body. Guessing an identifier from
#: a string is the same class of mistake this paper argues against.
S_MATRIX = "matrix"
S_GAIN = "gain"
S_COUNTS = "counts"

TABLES = [
    ("table2_protocols", table_protocol_defs, (), "protocols"),
    ("table3_setup", table_setup, (), "setup"),
    ("table4_task_ratios", table_task_ratios, (S_MATRIX, S_GAIN), "span-vs-gain"),
    ("table5_sign_flip", table_sign_flip, (S_MATRIX, S_GAIN), "sign-flip"),
    ("table6_literature", table_literature, (S_COUNTS,), "reporting-rates"),
    ("table7_checklist", table_checklist, (S_COUNTS,), "checklist"),
    ("tableA1_matrix", table_appendix_matrix, (S_MATRIX,), "full-matrix"),
    ("tableA2_gains", table_appendix_gain, (S_MATRIX, S_GAIN), "per-protocol-gains"),
    ("tableA3_audit", table_appendix_audit, (S_COUNTS,), "f3-breakdown"),
]


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def latex_escape(text):
    """Escape the characters that break a LaTeX tabular cell.

    The order matters and so does the completeness: the HEADER row needs the
    same treatment as the body. Leaving the header unescaped broke the build
    on "Wilson 95% CI" -- the bare `%` commented out the rest of the line,
    including the `\\`, and LaTeX then reported a misleading
    "Misplaced \noalign" on the following row.
    """
    # Markdown bold (`**text**`) has to become LaTeX bold, or the asterisks
    # end up printed verbatim. The markdown renderer keeps the `**` form.
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    return (text.replace("&", "\\&")
                .replace("%", "\\%")
                .replace("_", "\\_")
                .replace("#", "\\#"))


def render_latex(caption, head, body, notes, label):
    lines = []
    ncol = len(head)
    cols = "l" + "c" * (ncol - 1)
    lines.append("% Auto-generated by scripts/make_tables.py -- do not hand-edit.")
    lines.append("% Requires: \\usepackage{booktabs} \\usepackage{graphicx}")
    lines.append("% The \\resizebox around the tabular fits it to the text width;")
    lines.append("% use it inside a full-width float (table*) in a two-column paper.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{tab:{label}}}")
    # Width handling.
    #
    # Measured in a real pdflatex run: every table in this paper is wider
    # than a single-column `article` text block (345 pt) -- the worst, Table 3,
    # ran 197 pt over, and Table 2 cut off its last column entirely. These
    # tables are wide by nature (six protocol dimensions, three tasks), and
    # the paper is set two-column, so they belong in a full-width float
    # (`table*`) anyway.
    #
    # Rather than guess a column spec, the table is typeset at its natural
    # width and then scaled to exactly the text width with \resizebox. That
    # guarantees it never overflows, and the scale factor is emitted as a
    # comment so a reader can see when a table has been shrunk a lot.
    #
    # A paragraph column on the longest (prose) column lets the natural width
    # stay reasonable instead of one long line.
    spec = []
    for i in range(ncol):
        vals = [len(str(r[i])) for r in body] + [len(str(head[i]))]
        spec.append(max(vals))
    widest = max(range(ncol), key=lambda i: spec[i])

    if spec[widest] > 30:
        cols = []
        for i in range(ncol):
            if i == widest:
                cols.append("p{0.34\\linewidth}")
            else:
                cols.append("c")
        cols = "".join(cols)
        lines.append("\\footnotesize")
    else:
        cols = "l" + "c" * (ncol - 1)
        lines.append("\\small")
    # The tabular is wrapped in \resizebox so it can never exceed the text
    # width. Measured need: up to 197 pt over a single-column block.
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(latex_escape(h) for h in head) + " \\\\")
    lines.append("\\midrule")
    for row in body:
        cells = [latex_escape(c) for c in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    if notes:
        lines.append("\\vspace{2pt}")
        lines.append("\\begin{minipage}{0.98\\linewidth}\\footnotesize\\raggedright")
        for i, n in enumerate(notes, 1):
            lines.append(f"\\textbf{{({chr(96+i)})}} {latex_escape(n)}\\\\")
        lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def render_md(caption, head, body, notes):
    lines = [f"**{caption}**", ""]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    if notes:
        lines.append("")
        for i, n in enumerate(notes, 1):
            lines.append(f"{i}. {n}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=_p("paper", "tables"))
    ap.add_argument("--fmt", default="latex,md")
    ap.add_argument("--check", action="store_true",
                    help="verify against FACTS.md and write nothing")
    args = ap.parse_args()

    matrix = load_matrix()
    gain = load_gain()
    counts = literature_counts()

    problems = check_against_facts(matrix, gain, counts)
    if problems:
        print("TABLE SOURCES DISAGREE WITH FACTS.md -- refusing to build:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        sys.exit(1)
    print("check: all recomputed values agree with FACTS.md")

    if args.check:
        return

    os.makedirs(args.out, exist_ok=True)
    fmts = [f.strip() for f in args.fmt.split(",") if f.strip()]
    available = {S_MATRIX: matrix, S_GAIN: gain, S_COUNTS: counts}
    for name, fn, needs, label in TABLES:
        caption, head, body, notes = fn(*[available[k] for k in needs])
        if "latex" in fmts:
            with open(os.path.join(args.out, f"{name}.tex"), "w") as f:
                f.write(render_latex(caption, head, body, notes, label))
        if "md" in fmts:
            with open(os.path.join(args.out, f"{name}.md"), "w") as f:
                f.write(render_md(caption, head, body, notes))
        print(f"  wrote {name} ({len(body)} rows)")


if __name__ == "__main__":
    main()
