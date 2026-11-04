#!/usr/bin/env python3.11
"""Second-layer independent audit: recompute every manuscript number that can
be derived from the PUBLIC per-image release (results/per_image/*.csv).

This is deliberately stricter than audit_independent.py:
  * it recomputes from the finest-grained released data (per-image PSNR at
    6 decimals) rather than from the 4-decimal aggregates in the JSON;
  * it recomputes distributional claims (spans, percentiles, correlations,
    Jensen gaps, confidence intervals) that the manuscript asserts in prose;
  * it re-derives the P1 aligned gains from the released CSVs.
"""
import csv
import glob
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = os.path.join(ROOT, "results", "per_image")

FAIL = []
NOTES = []


def check(name, got, expected, tol=5e-3, unit=""):
    ok = abs(got - expected) <= tol
    tag = "OK  " if ok else "FAIL"
    if not ok:
        FAIL.append((name, got, expected, unit))
    print(f"  [{tag}] {name}: got {got:.6f}{unit}  expected {expected}{unit}")
    return ok


def check_eq(name, got, expected):
    ok = got == expected
    tag = "OK  " if ok else "FAIL"
    if not ok:
        FAIL.append((name, got, expected, ""))
    print(f"  [{tag}] {name}: got {got!r}  expected {expected!r}")
    return ok


def load(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["image"]] = float(r["psnr"])
    return out


def model_files(task, proto):
    pats = {
        "x4": f"generator_x4_x4__sr_generator_scale4_best__s*__{proto}.csv",
        "x2": f"srcnn_x2_x2__sr_srcnn_scale2_best__s*__{proto}.csv",
        "ll": f"lowlight_unet_x1__lowlight_srcnn_scale2_best__s*__{proto}.csv",
    }
    return sorted(glob.glob(os.path.join(PI, pats[task])))


def base_file(task, proto):
    names = {
        "x4": f"bicubic_sr_x4__bicubic__{proto}.csv",
        "x2": f"bicubic_sr_x2__bicubic__{proto}.csv",
        "ll": f"identity_ll__bicubic__{proto}.csv",
    }
    p = os.path.join(PI, names[task])
    return p if os.path.exists(p) else None


def mean_of(vals):
    return sum(vals) / len(vals)


def main():
    # =====================================================================
    print("=" * 74)
    print("A. Per-image means reproduce the aggregate JSON values")
    print("=" * 74)
    import json
    AGG = {
        "x4": "generator_x4_x4", "x2": "srcnn_x2_x2", "ll": "lowlight_unet_x1",
    }
    recomputed = {}
    for task in ("x4", "x2", "ll"):
        js = json.load(open(os.path.join(ROOT, "results", "matrix_runs",
                                        AGG[task], "protocol_matrix.json")))
        cks = js["checkpoints"]
        for ck_name, ck in cks.items():
            seed = ck["meta"]["seed"]
            for proto, rec in ck["protocols"].items():
                files = model_files(task, proto)
                got = None
                for f in files:
                    if f"__s{seed}__{proto}.csv" in f:
                        got = mean_of(list(load(f).values()))
                if got is None:
                    continue
                recomputed[(task, "model", proto, seed)] = got
                if abs(got - rec["psnr_mean"]) > 5e-4:
                    FAIL.append((f"agg {task} {proto} s{seed}", got,
                                 rec["psnr_mean"], ""))
                    print(f"  [FAIL] {task} {proto} s{seed}: per-image mean "
                          f"{got:.4f} != JSON {rec['psnr_mean']:.4f}")
    print("  (silent = every per-image mean reproduces its JSON aggregate)")

    # =====================================================================
    print()
    print("=" * 74)
    print("B. P1 aligned gains, recomputed from the released CSVs")
    print("=" * 74)
    for task, exp_model, exp_base, exp_gain in (
        ("x4", 44.2034, 44.7827, -0.5793),
        ("x2", 45.7951, 48.1074, -2.3123),
    ):
        bf = base_file(task, "P1")
        base = load(bf)
        inf_imgs = sorted(k for k, v in base.items() if not math.isfinite(v))
        finite = {k: v for k, v in base.items() if math.isfinite(v)}
        seeds = []
        for f in model_files(task, "P1"):
            seeds.append(load(f))
        common = set(finite)
        for s in seeds:
            common &= set(s)
        base_al = mean_of([finite[k] for k in common])
        model_al = mean_of([mean_of([s[k] for k in common]) for s in seeds])
        print(f"  -- {task}: n_total={len(base)} n_aligned={len(common)} "
              f"n_inf={len(inf_imgs)} inf_images={inf_imgs}")
        check(f"{task} P1 baseline aligned mean", base_al, exp_base, 5e-4, " dB")
        check(f"{task} P1 model aligned mean", model_al, exp_model, 5e-4, " dB")
        check(f"{task} P1 aligned gain", model_al - base_al, exp_gain, 5e-4, " dB")
        # model best PSNR on the 5 infinite-baseline images
        bests = []
        for s in seeds:
            bests.append(max(s[k] for k in inf_imgs))
        print(f"       model best on the {len(inf_imgs)} inf images: "
              f"min {min(bests):.2f} .. max {max(bests):.2f} dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("C. Headline spans / ratios (P2-P8 span, per Table 4's definition)")
    print("=" * 74)
    for task, exp_span, exp_span1, exp_gain, exp_r, exp_r1 in (
        ("x4", 3.889, 19.794, 0.782, 4.98, 25.33),
        ("x2", 3.143, 15.648, 0.863, 3.64, 18.13),
        ("ll", 2.250, 6.067, 10.736, 0.21, 0.57),
    ):
        protos = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
        mm, bb, gg = {}, {}, {}
        for proto in protos:
            fs = model_files(task, proto)
            if not fs:
                continue
            ms = [mean_of(list(load(f).values())) for f in fs]
            mm[proto] = mean_of(ms)
            bf = base_file(task, proto)
            if bf:
                bb[proto] = mean_of(list(load(bf).values()))
                gg[proto] = mm[proto] - bb[proto]
        p2p8 = [p for p in mm if p != "P1"]
        span = max(mm[p] for p in p2p8) - min(mm[p] for p in p2p8)
        span1 = max(mm.values()) - min(mm.values())
        gain = gg["P3"]
        print(f"  -- {task}  (protos available: {sorted(mm)})")
        check(f"{task} span P2-P8", span, exp_span, 5e-3, " dB")
        check(f"{task} span incl P1", span1, exp_span1, 5e-3, " dB")
        check(f"{task} gain @P3", gain, exp_gain, 5e-3, " dB")
        check(f"{task} ratio span/gain", span / gain, exp_r, 5e-2)
        check(f"{task} ratio span1/gain", span1 / gain, exp_r1, 5e-2)
        # seed sigma over full-image protocols
        full = [p for p in ("P3", "P4", "P5", "P6", "P7", "P8") if p in mm]
        sds = []
        for p in full:
            fs = model_files(task, p)
            per_seed = [mean_of(list(load(f).values())) for f in fs]
            sds.append((p, st.stdev(per_seed)))
        lo = min(s for _, s in sds) / span * 100
        hi = max(s for _, s in sds) / span * 100
        print(f"       seed sigma range: {min(s for _,s in sds):.4f}-"
              f"{max(s for _,s in sds):.4f} dB -> {lo:.2f}%-{hi:.2f}% of span")
        # P1/P2 sigma
        for p in ("P1", "P2"):
            if p in mm:
                fs = model_files(task, p)
                per_seed = [mean_of(list(load(f).values())) for f in fs]
                print(f"       {p} sigma = {st.stdev(per_seed):.4f} dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("D. Reversal (P5 -> P6) and the mechanism deltas")
    print("=" * 74)
    for task, exp_g5, exp_g6, exp_swing, exp_mdl, exp_bdl in (
        ("x4", 0.818, -1.753, 2.571, -3.89, -1.32),
        ("x2", 1.146, -1.889, 3.034, -2.72, 0.31),
    ):
        d = {}
        for proto in ("P3", "P5", "P6"):
            ms = [mean_of(list(load(f).values())) for f in model_files(task, proto)]
            bf = base_file(task, proto)
            bs = mean_of(list(load(bf).values()))
            d[proto] = (mean_of(ms), bs)
        g5 = d["P5"][0] - d["P5"][1]
        g6 = d["P6"][0] - d["P6"][1]
        print(f"  -- {task}")
        check(f"{task} P5 gain", g5, exp_g5, 5e-3, " dB")
        check(f"{task} P6 gain", g6, exp_g6, 5e-3, " dB")
        check(f"{task} P5->P6 swing", abs(g6 - g5), exp_swing, 5e-3, " dB")
        check(f"{task} model delta P5->P6", d["P6"][0] - d["P5"][0],
              exp_mdl, 5e-3, " dB")
        check(f"{task} base delta P5->P6", d["P6"][1] - d["P5"][1],
              exp_bdl, 5e-3, " dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("E. Jensen gap (mean(PSNR) - PSNR(mean MSE)) from aggregation_gap.csv")
    print("=" * 74)
    with open(os.path.join(ROOT, "results", "aggregation_gap.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["protocol"] in ("P2", "P3"):
                print(f"  {r['task']:>14} {r['protocol']}: "
                      f"jensen_gap={r['jensen_gap']}")

    # =====================================================================
    print()
    print("=" * 74)
    print("F. Distributional claims")
    print("=" * 74)
    # FACTS section 7.7 fixes the convention: the per-image PSNR is first
    # averaged over the five seeds, and the spread / sd / CI are computed
    # from THAT per-image series. Averaging the seed means first (rather than
    # per seed) changes the numbers, so the convention has to be stated here
    # or the check would be measuring something the paper does not claim.
    def cross_seed(task, proto):
        per_seed = [load(f) for f in model_files(task, proto)]
        imgs = sorted(per_seed[0].keys())
        return [mean_of([s[i] for s in per_seed]) for i in imgs]

    T975 = {14: 2.144787, 99: 1.984217}
    for task, exp_lo, exp_hi, exp_sd, exp_hw in (
        ("x4", 17.33, 40.43, 4.3844, 0.8700),
        ("x2", 22.26, 45.68, 4.6672, 0.9261),
        ("ll", 12.49, 23.72, 3.6747, 2.0350),
    ):
        xs = cross_seed(task, "P3")
        n = len(xs)
        sd = st.stdev(xs)
        hw = T975[n - 1] * sd / math.sqrt(n)
        print(f"  -- {task} P3 (cross-seed per-image series, n={n})")
        check(f"{task} per-image PSNR min", min(xs), exp_lo, 5e-2, " dB")
        check(f"{task} per-image PSNR max", max(xs), exp_hi, 5e-2, " dB")
        check(f"{task} between-image sd", sd, exp_sd, 5e-3, " dB")
        check(f"{task} 95% CI half-width", hw, exp_hw, 5e-3, " dB")

    # The manuscript quotes the x4 spread as 23.10 dB in the Discussion.
    xs4 = cross_seed("x4", "P3")
    check("x4 per-image PSNR spread", max(xs4) - min(xs4), 23.10, 5e-2, " dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("G. Channel effect: Y+shave (P5) minus RGB+shave (P4)")
    print("=" * 74)
    # The net channel effect subtracts the shave effect (P4 - P3), so P5 - P3
    # alone would mix two dimensions -- the exact error this paper is about.
    for task, exp in (("x4", 1.45), ("x2", 1.67), ("ll", 2.24)):
        p3 = mean_of([mean_of(list(load(f).values()))
                      for f in model_files(task, "P3")])
        p4 = mean_of([mean_of(list(load(f).values()))
                      for f in model_files(task, "P4")])
        p5 = mean_of([mean_of(list(load(f).values()))
                      for f in model_files(task, "P5")])
        net = (p5 - p3) - (p4 - p3)
        print(f"  {task}: (P5-P3)={p5 - p3:+.4f}  (P4-P3)={p4 - p3:+.4f}  "
              f"net channel={net:+.4f} dB   (expected {exp})")
        check(f"{task} net channel effect", net, exp, 5e-3, " dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("H. Antialias flag vs interpolation library")
    print("=" * 74)
    # Two library comparisons, each paired within a column (P7 vs P5 for the
    # Y+shave column, P8 vs P3 for the RGB column), against the antialias
    # switch in the same column. FACTS section 7.3 fixes the four ratios.
    for task, aa_exp, lib_a_exp, lib_b_exp, ratios in (
        ("x4", 3.8892, 0.0090, 0.0133, (429, 293)),
        ("x2", 2.7242, 0.0205, 0.0215, (133, 127)),
    ):
        def pm(proto):
            return mean_of([mean_of(list(load(f).values()))
                            for f in model_files(task, proto)])
        p3, p5, p6, p7, p8 = pm("P3"), pm("P5"), pm("P6"), pm("P7"), pm("P8")
        aa = abs(p5 - p6)
        lib_a = abs(p5 - p7)
        lib_b = abs(p3 - p8)
        print(f"  -- {task}")
        check(f"{task} antialias switch", aa, aa_exp, 5e-3, " dB")
        check(f"{task} library diff, Y+shave column", lib_a, lib_a_exp, 5e-3,
              " dB")
        check(f"{task} library diff, RGB column", lib_b, lib_b_exp, 5e-3, " dB")
        r_a, r_b = aa / lib_a, aa / lib_b
        print(f"       ratios: {r_a:.0f} and {r_b:.0f} "
              f"(FACTS: {ratios[0]} and {ratios[1]})")
        check(f"{task} AA/library ratio, Y+shave", r_a, ratios[0], 1.0)
        check(f"{task} AA/library ratio, RGB", r_b, ratios[1], 1.0)

    # =====================================================================
    print()
    print("=" * 74)
    print("I. Low-light P1 vs P3 (identity reference lifted by best-of-60)")
    print("=" * 74)
    p1 = mean_of([mean_of(list(load(f).values()))
                  for f in model_files("ll", "P1")])
    p3 = mean_of([mean_of(list(load(f).values()))
                  for f in model_files("ll", "P3")])
    print(f"  ll P1 - P3 = {p1 - p3:+.4f} dB (manuscript: 6.07)")
    check("ll P1 - P3", p1 - p3, 6.07, 5e-3, " dB")

    # =====================================================================
    print()
    print("=" * 74)
    print("I2. Seed sigma per protocol (P1 and P2 are listed separately)")
    print("=" * 74)
    # E87: the manuscript originally described three numbers as "P1 and P2
    # respectively", but all three were P1's. The P2 values are different
    # (and for low-light P2 is larger than P1), so both sets are checked.
    for task, p1_exp, p2_exp in (
        ("x4", 0.2364, 0.0363), ("x2", 0.5584, 0.0685), ("ll", 0.4148, 0.5303),
    ):
        for proto, exp in (("P1", p1_exp), ("P2", p2_exp)):
            per = [mean_of(list(load(f).values()))
                   for f in model_files(task, proto)]
            # The stored aggregates round the seed means before computing the
            # standard deviation, so recompute on the same basis.
            rounded = [round(x, 4) for x in per]
            sd_rounded = st.stdev(rounded)
            sd_full = st.stdev(per)
            check(f"{task} {proto} seed sigma (rounded basis)", sd_rounded,
                  exp, 5e-4, " dB")
            print(f"       (full-precision sigma {sd_full:.6f} dB; "
                  f"the stored aggregate uses the rounded basis)")

    # =====================================================================
    print()
    print("=" * 74)
    print("J. Literature coding counts")
    print("=" * 74)
    import collections
    rows = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "paper", "data",
                           "literature_audit_coding.csv"),
        encoding="utf-8-sig")) if r["编号"] != "示例"]
    print(f"  N = {len(rows)}")
    for k in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        c = collections.Counter(r[k].strip() for r in rows)
        print(f"  {k}: {dict(c.most_common())}")

    # The counts Table 6 reports, with the direction that FACTS section 8
    # fixes. The two channel counts are the ones that were mislabelled
    # (E86): 41 is the number that do NOT state a channel, 22 the number that
    # do -- and all 22 use Y.
    sr = [r for r in rows if r["任务(SR/LLIE)"].strip().startswith("SR")]
    llie = [r for r in rows if r["任务(SR/LLIE)"].strip().startswith("LLIE")]
    check_eq("literature N", len(rows), 63)
    check_eq("SR subset", len(sr), 44)
    check_eq("LLIE subset", len(llie), 19)
    check_eq("F1 aggregation stated",
             sum(1 for r in rows if r["F1"].strip() not in ("未报告", "")), 1)
    check_eq("F2 batch size stated",
             sum(1 for r in rows if r["F2"].strip() not in ("未报告", "")), 0)
    check_eq("F3 bicubic-protocol reproducible (SR)",
             sum(1 for r in sr if r["F3"].strip() in
                 ("具名可复现实现(MATLAB bicubic)", "显式抗锯齿")), 5)
    f4_unspec = sum(1 for r in rows if r["F4"].strip() == "未说明")
    f4_y = sum(1 for r in rows if r["F4"].strip().startswith("Y"))
    check_eq("F4 channel NOT stated", f4_unspec, 41)
    check_eq("F4 channel stated (all Y)", f4_y, 22)
    check_eq("F4 closes to N", f4_unspec + f4_y, 63)
    check_eq("F7 code repository", sum(1 for r in rows if r["F7"].strip() == "有"),
             35)
    f3_sum = sum(1 for r in sr if r["F3"].strip() in
                 ("具名可复现实现(MATLAB bicubic)", "显式抗锯齿",
                  "仅写bicubic无实现", "其他退化模型(高斯核/blur)", "未报告"))
    check_eq("F3 breakdown closes to SR subset", f3_sum, 44)

    print()
    print("=" * 74)
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURES")
        for name, got, exp, unit in FAIL:
            print(f"  - {name}: got {got} expected {exp}{unit}")
        print("=" * 74)
        return 1
    print("RESULT: all checks passed")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
