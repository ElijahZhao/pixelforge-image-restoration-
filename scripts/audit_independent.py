#!/usr/bin/env python3.11
"""Independent re-derivation of every headline number in paper/main.tex.

Deliberately does NOT import anything from scripts/ -- it reads the six raw
protocol_matrix.json files and recomputes means, spreads, gains, spans and
ratios from scratch, so it can disagree with the project's own pipeline.

Usage:
    python scripts/audit_independent.py            # print report
    python scripts/audit_independent.py --json     # machine-readable
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which model directories make up which task.
TASK_DIRS = {
    "x4": ["generator_x4_x4", "bicubic_sr_x4"],
    "x2": ["srcnn_x2_x2", "bicubic_sr_x2"],
    "ll": ["lowlight_unet_x1", "identity_ll"],
}
MODEL_DIR = {"x4": "generator_x4_x4", "x2": "srcnn_x2_x2", "ll": "lowlight_unet_x1"}
BASE_DIR = {"x4": "bicubic_sr_x4", "x2": "bicubic_sr_x2", "ll": "identity_ll"}


def load_cells():
    """Return {task: {protocol: {seed: psnr_mean}}} from the raw JSON."""
    cells = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for task, dirs in TASK_DIRS.items():
        for d in dirs:
            path = os.path.join(ROOT, "results", "matrix_runs", d, "protocol_matrix.json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            meta[task] = data["config"]
            is_model = d == MODEL_DIR[task]
            for ck_name, ck in data["checkpoints"].items():
                # seed id: the checkpoint key carries the seed, e.g.
                # sr_generator_scale4_best__s42 -> 42
                seed = ck.get("meta", {}).get("seed")
                if seed is None:
                    # Baselines are not trained, so their checkpoint key carries
                    # no seed (e.g. "bicubic"). Give them a single pseudo-seed.
                    seed = ck_name.rsplit("__", 1)[-1]
                    try:
                        seed = int(seed.lstrip("s"))
                    except ValueError:
                        seed = 0
                for proto, pv in ck["protocols"].items():
                    slot = "model" if is_model else "base"
                    cells[task][(slot, proto)][seed] = pv["psnr_mean"]
    return cells, meta


def summarise(vals):
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "std": st.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cells, meta = load_cells()
    report = {}
    lines = []
    failures = []

    def out(s=""):
        lines.append(s)

    # (task, key) -> (expected, tolerance). These are the values the
    # manuscript states; this script recomputes them from the raw JSON.
    EXPECT = {
        ("x4", "span_full"): (3.889, 5e-3),
        ("x4", "span_all"): (19.794, 5e-3),
        ("x4", "gain_p3"): (0.782, 5e-3),
        ("x4", "ratio_full"): (4.98, 5e-2),
        ("x4", "ratio_all"): (25.33, 5e-2),
        ("x2", "span_full"): (3.143, 5e-3),
        ("x2", "span_all"): (15.648, 5e-3),
        ("x2", "gain_p3"): (0.863, 5e-3),
        ("x2", "ratio_full"): (3.64, 5e-2),
        ("x2", "ratio_all"): (18.13, 5e-2),
        ("ll", "span_full"): (2.250, 5e-3),
        ("ll", "span_all"): (6.067, 5e-3),
        ("ll", "gain_p3"): (10.736, 5e-3),
        ("ll", "ratio_full"): (0.21, 5e-2),
        ("ll", "ratio_all"): (0.57, 5e-2),
    }
    WORST = {"x4": "P6", "x2": "P2", "ll": "P3"}
    # P5 -> P6 sign reversal: (gain at P5, gain at P6, swing)
    REVERSAL = {
        "x4": (0.818, -1.753, 2.571),
        "x2": (1.146, -1.889, 3.034),
    }

    def check(name, got, exp, tol):
        ok = abs(got - exp) <= tol
        if not ok:
            failures.append((name, got, exp))
        out(f"  [{'OK  ' if ok else 'FAIL'}] {name}: got {got:.6f} "
            f"expected {exp}")
        return ok

    for task in ("x4", "x2", "ll"):
        out(f"########## TASK {task}  cfg={meta[task]}")
        stats = {}
        for proto in sorted({p for (_, p) in cells[task]}):
            row = {}
            for slot in ("model", "base"):
                d = cells[task].get((slot, proto))
                if not d:
                    continue
                row[slot] = summarise(list(d.values()))
            if "model" in row and "base" in row:
                row["gain"] = row["model"]["mean"] - row["base"]["mean"]
            stats[proto] = row
            m = row.get("model", {})
            b = row.get("base", {})
            out(
                f"  {proto}: model {m.get('mean', float('nan')):.4f} "
                f"(sd {m.get('std', float('nan')):.4f})  "
                f"base {b.get('mean', float('nan')):.4f}  "
                f"gain {row.get('gain', float('nan')):+.4f}  "
                f"n={m.get('n')}"
            )
        report[task] = stats

        # --- protocol span
        # Definition taken from FACTS section 6 and Table 4: the span covers
        # ALL protocols except P1, i.e. P2-P8. P2 (centre crop) is a protocol
        # choice too, and on SR x2 it is the lowest protocol, so excluding it
        # would understate that task's span (2.724 vs 3.143). On SR x4 the
        # inclusion makes no difference because P2 sits inside the range.
        mm = {p: stats[p]["model"]["mean"] for p in stats if "model" in stats[p]}
        full = [p for p in ("P3", "P4", "P5", "P6", "P7", "P8") if p in mm]
        span_full = max(mm[p] for p in full) - min(mm[p] for p in full)
        p2p8 = [p for p in mm if p != "P1"]
        span_p2p8 = max(mm[p] for p in p2p8) - min(mm[p] for p in p2p8)
        allp = sorted(mm)
        span_all = max(mm.values()) - min(mm.values())
        gain_p3 = stats["P3"]["gain"]
        out(f"  >> span over P2-P8 = {span_p2p8:.4f} dB")
        out(f"  >> span over {full} = {span_full:.4f} dB")
        out(f"  >> span over {allp} = {span_all:.4f} dB")
        out(f"  >> gain@P3 = {gain_p3:.4f} dB")
        out(f"  >> ratio span/gain = {span_p2p8 / gain_p3:.4f}")
        out(f"  >> ratio span(all)/gain  = {span_all / gain_p3:.4f}")

        check(f"{task} span P2-P8", span_p2p8, EXPECT[(task, "span_full")][0],
              EXPECT[(task, "span_full")][1])
        check(f"{task} span incl P1", span_all,
              EXPECT[(task, "span_all")][0], EXPECT[(task, "span_all")][1])
        check(f"{task} gain @P3", gain_p3, EXPECT[(task, "gain_p3")][0],
              EXPECT[(task, "gain_p3")][1])
        check(f"{task} ratio span/gain", span_p2p8 / gain_p3,
              EXPECT[(task, "ratio_full")][0], EXPECT[(task, "ratio_full")][1])
        check(f"{task} ratio span(incl P1)/gain", span_all / gain_p3,
              EXPECT[(task, "ratio_all")][0], EXPECT[(task, "ratio_all")][1])

        # seed spread (std) over full-image protocols
        sds = [stats[p]["model"]["std"] for p in full if p in stats]
        out(f"  >> seed std over {full}: "
            + ", ".join(f"{p}={stats[p]['model']['std']:.4f}" for p in full))
        if sds:
            lo = min(sds) / span_p2p8 * 100
            hi = max(sds) / span_p2p8 * 100
            out(f"  >> seed std / span: {lo:.4f}% -- {hi:.4f}%")

        # most demanding protocol
        worst = min(mm, key=lambda p: mm[p])
        out(f"  >> lowest model metric protocol = {worst} ({mm[worst]:.4f})")
        if worst != WORST[task]:
            failures.append((f"{task} most demanding protocol", worst,
                             WORST[task]))
            out(f"  [FAIL] {task} most demanding protocol: {worst} "
                f"(expected {WORST[task]})")

        # P5 vs P6 reversal
        if "P5" in stats and "P6" in stats:
            g5, g6 = stats["P5"]["gain"], stats["P6"]["gain"]
            out(f"  >> P5 gain {g5:+.4f} -> P6 gain {g6:+.4f} | "
                f"swing {abs(g6 - g5):.4f} dB")
            m5, m6 = stats["P5"]["model"]["mean"], stats["P6"]["model"]["mean"]
            b5, b6 = stats["P5"]["base"]["mean"], stats["P6"]["base"]["mean"]
            out(f"     model P5->P6 delta {m6 - m5:+.4f}; base delta {b6 - b5:+.4f}")
            out(f"     swing / gain@P3 = {abs(g6 - g5) / gain_p3:.4f}")
            if task in REVERSAL:
                eg5, eg6, esw = REVERSAL[task]
                check(f"{task} P5 gain", g5, eg5, 5e-3)
                check(f"{task} P6 gain", g6, eg6, 5e-3)
                check(f"{task} P5->P6 swing", abs(g6 - g5), esw, 5e-3)
        if "P1" in stats:
            out(f"  >> P1 gain = {stats['P1'].get('gain', float('nan')):+.4f}")
        report[task + "_summary"] = {
            "span_full": span_full,
            "span_all": span_all,
            "gain_p3": gain_p3,
            "ratio_full": span_full / gain_p3,
            "ratio_all": span_all / gain_p3,
            "worst": worst,
        }
        out()

    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print("\n".join(lines))
        print()
        print("=" * 70)
        if failures:
            print(f"RESULT: {len(failures)} FAILURES")
            for name, got, exp in failures:
                print(f"  - {name}: got {got} expected {exp}")
            print("=" * 70)
            return 1
        print("RESULT: all checks passed")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
