#!/usr/bin/env python3
"""Protocol matrix: score the same checkpoint under P1..P8 and show the spread.

Why this exists
---------------
The paper's Figure 1 is "the same model, eight protocols, eight different
numbers". Building it used to require a Kaggle kernel, i.e. the number came
from code that lived outside the repo. This script is the repo-side
implementation, so the figure is reproducible from a clean clone:

    git clone ... && python scripts/protocol_matrix.py --ckpt_dir models

The eight protocols (each one a choice, each one named in the output):

    P1  random crop, best-of-N, RGB, bicubic AA        -- reproduces the OLD behaviour
    P2  centre crop, RGB, bicubic AA                   -- the training-time val metric
    P3  full image, RGB, bicubic AA                    -- this repo's "authoritative" number
    P4  full image, RGB, + border shave                -- literature, colour
    P5  full image, Y channel, + border shave          -- literature, complete
    P6  full image, Y channel, + shave, NO antialias   -- the dimension that flips the sign
    P7  full image, Y + shave, PIL bicubic             -- operator axis, literature metric
    P8  full image, RGB, PIL bicubic                   -- operator axis, repo metric

P1 vs P2 vs P3 is *sampling*; P3 vs P4 vs P5 is *metric channel*; P5 vs P6 vs
P7 is *degradation operator*; P7 vs P8 is metric again, on the PIL operator.
Those are what the matrix decomposes: variance from the protocol versus
distance between models.

A protocol is a point in (sampling x metric x degrade). Two protocols that
share all three except `metric` **see the same network output** -- P3/P4/P5
are one forward pass measured three ways, P7/P8 likewise. The grouped path
below exploits that: one forward, several metrics. It is not an approximation;
measuring the same tensor three ways is what the matrix always meant.

Grouping also removes a hidden cost the old loop carried: it opened every
image once per protocol (100 images x 8 protocols = 800 decodes), because the
file loop lived inside the protocol loop. Hence `--legacy`, which keeps the
old path alive *for equivalence testing only*:

    python scripts/protocol_matrix.py --legacy ... > old.json
    python scripts/protocol_matrix.py ...          > new.json
    # per-image PSNR/SSIM must match to the last bit

Outputs
-------
``<out>/protocol_matrix.json``   nested: ckpt -> protocol -> metrics
                                 (multi-dataset runs add a ``datasets`` level)
``<out>/protocol_matrix.csv``    tidy one row per (dataset, ckpt, protocol)
``<out>/per_image/<ckpt>__<P>.csv``  per-image PSNR/SSIM, so a reviewer can
                                 recompute any aggregate without trusting ours

Usage
-----
    python scripts/protocol_matrix.py --ckpt models/sr_generator_scale4_best.pth --scale 4
    python scripts/protocol_matrix.py --ckpt_dir models/seeds --scale 4
    python scripts/protocol_matrix.py --ckpt_list ckpts.txt --datasets div2k/val,set14
    python scripts/protocol_matrix.py --baseline --datasets div2k/val   # bicubic, no model
    python scripts/protocol_matrix.py --legacy ...                      # equivalence check
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# The repository's `train/` directory holds the shared modules (degradation,
# eval_protocol, metrics, models). The path is derived from THIS FILE, not
# from the current working directory, so the script works when it is invoked
# from elsewhere -- a relative "train" only resolves when the CWD happens to
# be the repository root.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from degradation import build_lr, describe  # noqa: E402
from eval_protocol import psnr_y, ssim_y, shave  # noqa: E402
from metrics import psnr_per_image, ssim_per_image  # noqa: E402
from models import build_model, pad_to_align  # noqa: E402

# `_measure` converts a grad-tracking tensor to float. That is harmless here
# (nothing calls backward) and it emits one UserWarning per image, so the exact
# message is filtered instead of the tensor being detached: on CPU a
# requires_grad tensor takes a different reduction kernel, and detaching moved
# P2/P8 by 1 ulp -- not worth it to quiet a log line.
warnings.filterwarnings(
    "ignore",
    message="Converting a tensor with requires_grad=True to a scalar",
    category=UserWarning,
)

_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Protocol definitions. `sampling` x `metric` x `degrade` is the whole axis
#: system; the eight named protocols are points in that space, not eight code
#: paths.
PROTOCOLS = {
    "P1": dict(sampling="random", metric="rgb", degrade="tv_bicubic_aa",
               agg="best_of_n", label="random crop, best-of-N"),
    "P2": dict(sampling="center", metric="rgb", degrade="tv_bicubic_aa",
               agg="per_image", label="centre crop (training val)"),
    "P3": dict(sampling="full", metric="rgb", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, RGB"),
    "P4": dict(sampling="full", metric="rgb_shave", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, RGB + shave"),
    "P5": dict(sampling="full", metric="y_shave", degrade="tv_bicubic_aa",
               agg="per_image", label="full image, Y + shave"),
    "P6": dict(sampling="full", metric="y_shave", degrade="tv_bicubic_noaa",
               agg="per_image", label="full image, Y + shave, no antialias"),
    # P7/P8 close the degradation axis. Until they existed the matrix only had
    # two of the three operators in `degradation.py`, so "how much does the
    # choice of operator move the score" was answerable for AA on/off but not
    # for PIL. They matter for a concrete reason: every number already
    # published for this repo was produced by `eval_baseline.py`, whose default
    # operator is `pil_bicubic`, while P3/P5 use `tv_bicubic_aa`. P8 is P3 with
    # only the operator changed, so |P8 - P3| is that difference and nothing
    # else; P7 does the same for P5.
    "P7": dict(sampling="full", metric="y_shave", degrade="pil_bicubic",
               agg="per_image", label="full image, Y + shave, PIL bicubic"),
    "P8": dict(sampling="full", metric="rgb", degrade="pil_bicubic",
               agg="per_image", label="full image, RGB, PIL bicubic"),
}


def _load_images(folder, max_images=0):
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(_EXTS))
    if max_images:
        files = files[:max_images]
    return files


def find_checkpoints(ckpt_dir):
    """Every .pth/.pt under a directory, sorted (a seed matrix lives here)."""
    out = []
    for pat in ("**/*.pth", "**/*.pt"):
        out.extend(glob.glob(os.path.join(ckpt_dir, pat), recursive=True))
    return sorted(p for p in out if os.path.isfile(p))


def load_model(path, task, scale):
    """Load either a TorchScript export (``serve/models/*.pt``) or a state dict.

    Both are in the repo for good reason: the deployed weights are TorchScript
    ``.pt`` (they are what the service actually runs), while seed-matrix
    experiments produce ``.pth`` state dicts. A protocol table that can only be
    built from one of them cannot answer "is the *deployed* model's number
    reproducible", which is the question a reviewer asks first.
    """
    try:
        # Load straight onto the target device. The T4 probe did it this way and
        # worked; loading to CPU first and calling .to() afterwards left a
        # TorchScript module whose internals were still CPU-resident.
        model = torch.jit.load(path, map_location=DEVICE)
        model.eval()
        return model.to(DEVICE), {"source": "torchscript"}
    except Exception:  # noqa: BLE001 - not a TorchScript file, fall through
        pass

    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    name = ck.get("model") if isinstance(ck, dict) else None
    sc = ck.get("scale", scale) if isinstance(ck, dict) else scale
    model = build_model(task, scale=sc, advanced=(name == "generator"))
    model.load_state_dict(state)
    model.eval()
    meta = {k: ck.get(k) for k in ("seed", "crop_size", "tag", "epoch", "val_psnr")} \
        if isinstance(ck, dict) else {}
    meta["source"] = "state_dict"
    return model.to(DEVICE), meta


def _crop_boxes(h, w, crop, n, rng):
    if crop is None or crop >= min(h, w):
        return [(0, 0, h, w)]
    if n <= 1 and crop == min(h, w):
        return [(0, 0, crop, crop)]
    # numpy Generator uses `integers` (half-open high); `randint` is the legacy
    # RandomState API and silently fails here.
    return [(int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1)),
             crop, crop) for _ in range(n)]


def _boxes_for(proto, h, w, crop, n_random, rng):
    if proto["sampling"] == "center":
        return [((h - min(crop or h, h)) // 2, (w - min(crop or w, w)) // 2,
                 min(crop or h, h), min(crop or w, w))]
    if proto["sampling"] == "random":
        return _crop_boxes(h, w, crop, n_random, rng)
    return [(0, 0, h, w)]


def _measure(out, hr, proto, border):
    """PSNR/SSIM of one SR output under one protocol's metric.

    Split out so the grouped path can measure a single forward pass several
    ways without duplicating the switch.
    """
    if proto["metric"] == "rgb":
        return (float(psnr_per_image(out, hr).mean()),
                float(ssim_per_image(out, hr).mean()))
    if proto["metric"] == "rgb_shave":
        o, h = shave(out, border), shave(hr, border)
        return (float(psnr_per_image(o, h).mean()),
                float(ssim_per_image(o, h).mean()))
    if proto["metric"] == "y":
        return (float(psnr_y(out, hr, border=0).mean()),
                float(ssim_y(out, hr, border=0).mean()))
    if proto["metric"] == "y_shave":
        return (float(psnr_y(out, hr, border=border).mean()),
                float(ssim_y(out, hr, border=border).mean()))
    raise ValueError(proto["metric"])


def _forward(model, hr_pil, box, scale, degrade, baseline, align=1, lr_pil=None):
    """One crop -> one SR output on the HR grid. No metrics, no policy.

    ``lr_pil`` switches to PAIRED mode. The low-light model consumes a real
    capture (`lol/val/low`), not a synthetic degradation of the target, and its
    scale is 1. v3 got this wrong: it fed `high/` through `build_lr` and also
    passed `--scale 2`, so the U-Net saw a half-resolution image and the score
    came out at 12.5 dB against an 18.0 dB training-time validation. When
    ``lr_pil`` is given, the same crop box is taken from both halves -- they are
    verified same-size upstream -- and `build_lr` is not called at all.

    ``align`` is the network's downsampling stride multiple. U-Net stacks use
    stride-2 convs (floor) followed by nearest x2 upsamples (exact), so an LR
    side that is not a multiple of ``align`` shifts the skip connection by one
    pixel and `torch.cat` raises "Expected size 76 but got size 75". HR is
    therefore reflect-padded up to ``align * scale`` before the forward pass and
    the output is cropped back, so **measurement still happens on the original
    pixels** -- padding is an inference-time accommodation, not a change of
    evaluation region. Plain conv+shuffle SR models need no padding (align=1),
    and 96px crops already satisfy align<=32, so nothing about the SR numbers
    depends on this flag.

    The padding itself comes from ``models.pad_to_align``, the same helper the
    training validation loop uses, so the two cannot drift apart. It falls
    back from reflection to replication for an image smaller than the
    multiple (reflection requires the pad to be narrower than the side it
    pads).
    """
    top, left, ch, cw = box
    patch = hr_pil.crop((left, top, left + cw, top + ch))
    hr = TF.to_tensor(patch).unsqueeze(0).to(DEVICE)
    hr_eval = hr
    if lr_pil is not None:
        # Paired mode. `scale` is 1 by construction, so the pad multiple is
        # `align` alone; padding the INPUT is what keeps the skip connections
        # aligned, and the output is cropped back below.
        lr = TF.to_tensor(lr_pil.crop((left, top, left + cw, top + ch)))
        lr = lr.unsqueeze(0).to(DEVICE)
        if align > 1 and not baseline:
            lr, _ = pad_to_align(lr, align)
    else:
        if align > 1 and not baseline:
            hr, _ = pad_to_align(hr, scale * align)
        # `.to(DEVICE)` is not redundant: the `pil_bicubic` branch of build_lr
        # goes through PIL, which is a CPU library, and re-stacks per-image
        # results back on the CPU. On GPU that produced "Input type (FloatTensor)
        # and weight type (cuda.FloatTensor)" -- but only for P7/P8, i.e. exactly
        # the protocols that the T4 probe never touched (it timed
        # `tv_bicubic_aa`, which stays on the device). The anchor gate caught it
        # in one minute instead of after an eighty-minute matrix.
        lr = build_lr(hr, scale, mode=degrade).to(DEVICE)
    if baseline:
        if lr_pil is not None:
            # Paired mode has no upscaling step, so the honest reference line is
            # "do nothing": return the capture itself. Painting bicubic here
            # would compare an enhancer against an upsampler, which is not a
            # comparison at all. `lr` may have been reflect-padded above, so it
            # is cropped back to the measured region.
            if lr.shape[-2:] != hr_eval.shape[-2:]:
                lr = lr[..., :ch, :cw]
            return lr.clamp(0, 1), hr_eval
        # The classic baseline: LR (already produced by `degrade`) pushed back
        # up with plain bicubic. No network, so every protocol still applies.
        out = torch.nn.functional.interpolate(
            lr, size=list(hr.shape[-2:]), mode="bicubic", align_corners=False)
        return out.clamp(0, 1), hr
    out = model(lr).clamp(0, 1)
    if align > 1 and out.shape[-2:] != hr_eval.shape[-2:]:
        out = out[..., :ch, :cw]
    if out.shape[-2:] != hr_eval.shape[-2:]:
        # The model's true output is lr*scale; when H/W are not divisible by
        # the scale, the grid differs by a pixel. Resize back so PSNR is
        # measured on the HR grid -- an evaluation-time normalisation, not a
        # second upscale (same convention as scripts/eval_baseline.py).
        out = TF.resize(out, list(hr_eval.shape[-2:]), antialias=False)
    return out, hr_eval


def score_one(model, hr_pil, proto, scale, border, crop, n_random, rng,
              baseline=False, align=1, lr_pil=None):
    """Legacy path: ONE protocol, opening the image per protocol.

    Kept verbatim (modulo the baseline switch) so `--legacy` can prove the
    grouped path produces bit-identical numbers.
    """
    w, h = hr_pil.size
    boxes = _boxes_for(proto, h, w, crop, n_random, rng)
    best = None
    vals = []
    for box in boxes:
        out, hr = _forward(model, hr_pil, box, scale, proto["degrade"], baseline,
                           align, lr_pil)
        p, s = _measure(out, hr, proto, border)
        vals.append((p, s))
        best = (p, s) if best is None or p > best[0] else best

    if proto["agg"] == "best_of_n":
        return best
    if proto["agg"] == "mean_of_n":
        return (statistics.mean(v[0] for v in vals), statistics.mean(v[1] for v in vals))
    return vals[0]


def score_image(model, hr_pil, wanted, scale, border, crop, n_random, rng,
                baseline=False, align=1, lr_pil=None):
    """Grouped path: every requested protocol for ONE image.

    Protocols that share (degrade, sampling) share one forward pass -- they
    differ only in how that tensor is measured. P1 is alone in its group
    because `best_of_n` needs N draws, and it keeps a dedicated RNG consumed in
    image order, which is exactly the order the legacy loop used.

    ``lr_pil`` is forwarded unchanged: in paired mode every protocol draws its
    input from `low/` instead of from `build_lr(high)`. Grouping stays valid
    because the pairing is per-image, not per-protocol.
    """
    w, h = hr_pil.size
    groups = {}
    for p in wanted:
        key = (PROTOCOLS[p]["degrade"], PROTOCOLS[p]["sampling"])
        groups.setdefault(key, []).append(p)

    out = {}
    # The group key is (degrade, sampling): protocols that share both forward
    # pass once and are measured several ways. Only `degrade` is needed here,
    # because the crop boxes come from `proto0` (the first protocol in the
    # group, which carries the group's sampling); `sampling` is unpacked to
    # make the key's meaning explicit at the point of use.
    for (degrade, _sampling), plist in groups.items():
        proto0 = PROTOCOLS[plist[0]]
        boxes = _boxes_for(proto0, h, w, crop, n_random, rng)
        got = []
        best = None
        for box in boxes:
            o, hr = _forward(model, hr_pil, box, scale, degrade, baseline, align,
                             lr_pil)
            per = {p: _measure(o, hr, PROTOCOLS[p], border) for p in plist}
            got.append(per)
            if best is None or per[plist[0]][0] > best[plist[0]][0]:
                best = per
        for p in plist:
            agg = PROTOCOLS[p]["agg"]
            if agg == "best_of_n":
                out[p] = best[p]
            elif agg == "mean_of_n":
                out[p] = (statistics.mean(g[p][0] for g in got),
                          statistics.mean(g[p][1] for g in got))
            else:
                out[p] = got[0][p]
    return out


def _resolve_datasets(args):
    """Return [(name, dir), ...]; name is what lands in the output."""
    if args.datasets:
        out = []
        for spec in [s.strip() for s in args.datasets.split(",") if s.strip()]:
            out.append((spec, os.path.join(args.data_root, spec)))
        return out
    if args.task == "lowlight":
        return [("lol/val", os.path.join(args.data_root, "lol", "val", "high"))]
    return [("div2k/val", os.path.join(args.data_root, "div2k", "val"))]


def _collect_ckpts(args):
    ckpts = []
    if args.ckpt:
        ckpts.extend(args.ckpt)
    if args.ckpt_list:
        with open(args.ckpt_list) as f:
            ckpts.extend(ln.strip() for ln in f if ln.strip() and not ln.startswith("#"))
    if args.ckpt_dir:
        ckpts.extend(find_checkpoints(args.ckpt_dir))
    return ckpts


def _ckpt_name(path):
    """Unique label for a checkpoint.

    A seed matrix stores `sr_generator_scale4_best.pth` under `s42/`, `s43/`,
    ... -- identical basenames. Keying results by basename alone silently keeps
    only the last seed (json) or overwrites the per-image CSV, which is exactly
    the "ran 5 seeds, reported 1" failure the ledger warns about. The seed
    directory is therefore part of the name.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if len(parent) > 1 and parent[0] == "s" and parent[1:].isdigit():
        return f"{base}__{parent}"
    return base


def _score_dataset(args, ds_name, ds_dir, ckpts, wanted, border, files, tag_prefix):
    """Run every checkpoint over one dataset. Returns (rec_by_ckpt, rows)."""
    rec_by_ckpt, rows = {}, []
    per_img = os.path.join(args.out, "per_image")
    os.makedirs(per_img, exist_ok=True)

    low_by_stem = {}
    if getattr(args, "paired", False):
        # `files` comes from the target half (`lol/val/high`); the input half is
        # its sibling. Pairs are matched by stem, the same rule
        # `LowLightDataset` uses, and a missing partner is fatal -- silently
        # falling back to a synthetic degradation is precisely the v3 bug.
        low_dir = os.path.join(os.path.dirname(ds_dir.rstrip(os.sep)), "low")
        if not os.path.isdir(low_dir):
            print(f"[x] paired evaluation needs {low_dir}/", flush=True)
            raise SystemExit(2)
        for fn in os.listdir(low_dir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in _EXTS:
                low_by_stem[stem] = os.path.join(low_dir, fn)
        if not low_by_stem:
            print(f"[x] {low_dir}/ has no images", flush=True)
            raise SystemExit(2)

    def _load_pair(hr_path):
        """(hr_pil, lr_pil); lr_pil is None outside paired mode."""
        hr = Image.open(hr_path).convert("RGB")
        if not getattr(args, "paired", False):
            return hr, None
        stem = os.path.splitext(os.path.basename(hr_path))[0]
        lp = low_by_stem.get(stem)
        if lp is None:
            print(f"[x] {os.path.basename(hr_path)} has no low/{stem}.* partner",
                  flush=True)
            raise SystemExit(2)
        lr = Image.open(lp).convert("RGB")
        if lr.size != hr.size:
            # The same crop box is used for both halves, so unequal sizes would
            # silently misalign input and target -- `LowLightDataset` refuses
            # these too.
            print(f"[x] size mismatch for {stem}: low={lr.size} high={hr.size}",
                  flush=True)
            raise SystemExit(2)
        return hr, lr

    models = []
    if args.baseline:
        # Naming the paired reference "bicubic" was wrong twice over: in paired
        # mode no upscaling happens, so the baseline returns the capture
        # unchanged -- it is the do-nothing line, not an interpolator. The
        # archived low-light baseline JSON still carries the old key, so
        # `scripts/aggregate_matrix.py` decides the reference name from
        # `paired`, not from this string.
        models.append(("identity" if args.paired else "bicubic", None,
                       {"source": "interpolation" if not args.paired else "identity"}))
    for ckpt in ckpts:
        try:
            m, meta = load_model(ckpt, args.task, args.scale)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] cannot load {ckpt}: {exc}", flush=True)
            continue
        models.append((_ckpt_name(ckpt), m, meta))

    for name, model, meta in models:
        seed = meta.get("seed")
        tag = f"seed={seed}" if seed else meta.get("source")
        print(f"\n=== {ds_name} | {name} ({tag}) ===", flush=True)
        # The "path" field is part of the JSON schema the archived runs use
        # (14 archived protocol_matrix.json files carry it), so it stays --
        # but it is always null there too: the checkpoint's location is in
        # `meta`, and `path` records nothing. The previous expression was
        # `None if model is None else None`, i.e. a condition with no effect
        # on the result; writing the constant says what it actually means.
        rec = {"path": None, "meta": meta,
               "protocols": {}, "dataset": ds_name}

        # per protocol: list of per-image values, filled by whichever path runs
        ps = {p: [] for p in wanted}
        ss = {p: [] for p in wanted}
        per_sec = {p: 0.0 for p in wanted}
        t_ds = time.time()

        if args.legacy:
            for p in wanted:
                rng = np.random.default_rng(args.seed)
                t0 = time.time()
                for f in files:
                    hr, lr = _load_pair(f)
                    pv, sv = score_one(model, hr, PROTOCOLS[p], args.scale, border,
                                       args.crop, args.n_random, rng, args.baseline,
                                       args.align, lr)
                    ps[p].append(pv)
                    ss[p].append(sv)
                per_sec[p] = time.time() - t0
        else:
            # One RNG for the whole dataset, consumed in image order: P1's
            # window draws then match the legacy per-protocol loop exactly.
            rng = np.random.default_rng(args.seed)
            t0 = time.time()
            for f in files:
                hr, lr = _load_pair(f)
                got = score_image(model, hr, wanted, args.scale, border,
                                  args.crop, args.n_random, rng, args.baseline,
                                  args.align, lr)
                for p in wanted:
                    ps[p].append(got[p][0])
                    ss[p].append(got[p][1])
            # Cost is shared by the group, so it is reported per group and
            # written to every protocol in it; `sec` is diagnostic, never an
            # anchor.
            shared = time.time() - t0
            for p in wanted:
                per_sec[p] = shared / max(1, len(set(
                    (PROTOCOLS[q]["degrade"], PROTOCOLS[q]["sampling"]) for q in wanted)))

        for p in wanted:
            proto = PROTOCOLS[p]
            # A bicubic baseline can reconstruct a flat patch exactly, so MSE==0
            # and `psnr_per_image` returns +inf -- mathematically right, but
            # `statistics.stdev` refuses inf and killed the whole baseline run.
            # Keep +inf in the per-image CSV (it is the true value) and exclude
            # it from the aggregate, recording the count so the exclusion is
            # visible rather than silent.
            finite = [v for v in ps[p] if math.isfinite(v)]
            n_inf = len(ps[p]) - len(finite)
            rec["protocols"][p] = {
                "label": proto["label"], "degrade": proto["degrade"],
                "metric": proto["metric"], "sampling": proto["sampling"],
                "agg": proto["agg"],
                "psnr_mean": round(statistics.mean(finite), 4) if finite else None,
                "psnr_std": round(statistics.stdev(finite), 4) if len(finite) > 1 else None,
                "ssim_mean": round(statistics.mean(ss[p]), 4),
                "n_images": len(ps[p]), "n_inf": n_inf, "sec": round(per_sec[p], 1),
            }
            fn = f"{name}__{p}.csv" if not tag_prefix else f"{tag_prefix}__{name}__{p}.csv"
            with open(os.path.join(per_img, fn), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "psnr", "ssim"])
                for fpath, pv, sv in zip(files, ps[p], ss[p]):
                    w.writerow([os.path.basename(fpath), f"{pv:.6f}", f"{sv:.6f}"])
            r = rec["protocols"][p]
            mark = f" (+{r['n_inf']} exact)" if r["n_inf"] else ""
            pm = (f"{r['psnr_mean']:7.3f}" if r["psnr_mean"] is not None
                  else "   all-inf")
            print(f"  {p} {pm} dB / {r['ssim_mean']:.4f}  "
                  f"[{proto['label']}] ({r['sec']}s){mark}", flush=True)
            rows.append({"dataset": ds_name, "checkpoint": name, "protocol": p,
                         "label": proto["label"], "degrade": proto["degrade"],
                         "metric": proto["metric"], "sampling": proto["sampling"],
                         "agg": proto["agg"], "psnr_mean": r["psnr_mean"],
                         "psnr_std": r["psnr_std"], "ssim_mean": r["ssim_mean"],
                         "n_inf": r["n_inf"],
                         "n_images": r["n_images"]})
        print(f"  [{ds_name} | {name}] {time.time() - t_ds:.1f}s total", flush=True)
        rec_by_ckpt[name] = rec
    return rec_by_ckpt, rows


def run(args):
    # Paired evaluation is a property of the TASK, not a switch. The low-light
    # model consumes a real capture and is scale-1; there is no synthetic
    # degradation to build. Leaving this to a CLI flag is what let v3 ship a
    # wrong number: it degraded `high/` with bicubic *and* passed `--scale 2`,
    # scoring 12.5 dB against an 18.0 dB training-time validation.
    args.paired = (args.task == "lowlight")
    datasets = _resolve_datasets(args)
    usable = []
    for name, d in datasets:
        if not os.path.isdir(d):
            print(f"[!] no images at {d} (dataset {name}) -- skipped", flush=True)
            continue
        files = _load_images(d, args.max_images)
        if not files:
            print(f"[!] {d} is empty -- skipped", flush=True)
            continue
        usable.append((name, d, files))
    if not usable:
        print("[!] no usable dataset")
        return 2

    ckpts = _collect_ckpts(args)
    if not ckpts and not args.baseline:
        print("[!] no checkpoint given (--ckpt / --ckpt_list / --ckpt_dir)")
        return 2

    wanted = [p.strip().upper() for p in args.protocols.split(",") if p.strip()]
    for p in wanted:
        if p not in PROTOCOLS:
            print(f"[!] unknown protocol {p}; known: {sorted(PROTOCOLS)}")
            return 2

    border = args.shave if args.shave is not None else args.scale
    if args.paired and args.scale != 1:
        # LowLightUNet is 1:1 (`build_model("lowlight")` ignores `scale`). A
        # scale > 1 here would downscale the capture first and then resize the
        # output back -- two wrongs that mostly cancel visually and leave a
        # plausible-looking but meaningless PSNR. Refuse instead.
        print(f"[x] --task lowlight is scale-1, got --scale {args.scale}. "
              f"The 'scale2' in the checkpoint filename is a historical name, "
              f"not a property of the network.", flush=True)
        return 2
    print(f"[i] datasets={[u[0] for u in usable]} | device={DEVICE} | "
          f"scale=x{args.scale} | shave={border} | crop={args.crop} | "
          f"n_random={args.n_random} | grouped={not args.legacy} | "
          f"baseline={args.baseline} | paired={args.paired}", flush=True)
    if args.paired:
        print("[i] paired mode: input = lol/val/low, target = lol/val/high; "
              "no synthetic degradation is applied and scale must be 1",
              flush=True)
    print(f"[i] 退化算子会被逐个协议记录; 训练侧用的是 {describe('tv_bicubic_aa')}",
          flush=True)

    os.makedirs(args.out, exist_ok=True)
    multi = len(usable) > 1
    result = {"protocols": {p: PROTOCOLS[p] for p in wanted},
              "config": {"scale": args.scale, "shave": border, "crop": args.crop,
                         "n_random": args.n_random, "device": DEVICE,
                         "grouped": not args.legacy, "baseline": args.baseline,
                         "paired": args.paired},
              "checkpoints": {}, "datasets": {}}
    all_rows = []

    for ds_name, ds_dir, files in usable:
        tag_prefix = ds_name.replace("/", "-") if multi else ""
        recs, rows = _score_dataset(args, ds_name, ds_dir, ckpts, wanted, border,
                                    files, tag_prefix)
        all_rows.extend(rows)
        if multi:
            result["datasets"][ds_name] = {"dir": ds_dir, "n_images": len(files),
                                           "checkpoints": recs}
        else:
            # Single-dataset shape stays exactly as before, so anchor
            # comparisons against the v3 artefact need no adapter.
            result["checkpoints"] = recs
            result["config"]["data"] = ds_dir
            result["config"]["n_images"] = len(files)

    with open(os.path.join(args.out, "protocol_matrix.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if all_rows:
        with open(os.path.join(args.out, "protocol_matrix.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)

    # The spread is the result: if protocols vary more than the models do, the
    # paper's claim is about protocols and must be phrased that way.
    print("\n=== spread across protocols (same checkpoint) ===")
    for ds, blk in (result["datasets"].items() if multi
                    else [(None, {"checkpoints": result["checkpoints"]})]):
        for name, rec in blk["checkpoints"].items():
            vals = [v["psnr_mean"] for v in rec["protocols"].values()
                    if v["psnr_mean"] is not None]
            if len(vals) > 1:
                print(f"  {(ds + ' | ') if ds else ''}{name}: {min(vals):.3f} .. "
                      f"{max(vals):.3f} dB (spread {max(vals) - min(vals):.3f} dB)")
    print(f"\n[i] wrote {args.out}/protocol_matrix.{{json,csv}} and per_image/*.csv")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="checkpoint path (repeatable)")
    ap.add_argument("--ckpt_list", default="",
                    help="file with one checkpoint path per line (# comments ok)")
    ap.add_argument("--ckpt_dir", default="", help="directory of checkpoints (recursive)")
    ap.add_argument("--task", choices=["sr", "lowlight"], default="sr")
    ap.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--datasets", default="",
                    help="comma list of dirs under --data_root, e.g. div2k/val,set14; "
                         "empty = derive from --task")
    ap.add_argument("--protocols", default="",
                    help=f"comma list from {sorted(PROTOCOLS)}; default = all "
                         f"eight for SR, P1-P5 for lowlight (P6-P8 vary the "
                         f"synthetic degradation operator, which does not exist "
                         f"when the input is a real capture)")
    ap.add_argument("--crop", type=int, default=96,
                    help="patch side for P1/P2 (default 96 = training val crop)")
    ap.add_argument("--n_random", type=int, default=60,
                    help="random windows per image for P1 (default 60; the old "
                         "code effectively used --epochs per image)")
    ap.add_argument("--shave", type=int, default=None,
                    help="border pixels for shave protocols (default = scale)")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", action="store_true",
                    help="skip the network, report bicubic-upsampled LR instead; "
                         "gives Figure 1 the reference line it needs")
    ap.add_argument("--align", type=int, default=None,
                        help="network downsampling stride multiple (U-Net=8); HR is "
                         "reflect-padded to align*scale before the forward pass "
                         "and cropped back, so metrics stay on original pixels. "
                         "Defaults to 8 for the low-light U-Net (which halves "
                         "the resolution three times and so needs sides that "
                         "are a multiple of 8) and 1 for SR (conv + pixel "
                         "shuffle, no alignment constraint).")
    ap.add_argument("--legacy", action="store_true",
                    help="old per-protocol loop, for bit-exact equivalence tests only")
    ap.add_argument("--out", default="results/protocol_matrix")
    a = ap.parse_args(argv)
    if a.align is None:
        # The low-light U-Net halves the resolution three times, so its skip
        # connections need both sides to be a multiple of 8. The default used
        # to be 1 for every task, which meant running this script on the
        # low-light model -- its whole purpose -- crashed with "Sizes of
        # tensors must match" unless the caller happened to pass --align 8.
        # LOL-v1 is 400x600, a multiple of 8, so the crash only shows up on
        # any other dataset.
        a.align = 8 if a.task == "lowlight" else 1
    if not a.protocols:
        a.protocols = ("P1,P2,P3,P4,P5" if a.task == "lowlight"
                       else ",".join(PROTOCOLS))
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
