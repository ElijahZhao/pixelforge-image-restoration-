#!/usr/bin/env python3
"""Build the public per-image data release from the private archive.

The archive holds several generations of evaluation output. Only the six
that produced `results/matrix_runs/*/protocol_matrix.json` back the paper,
so only those are published. The mapping is not taken on trust: it is
re-derived here by matching each archived file's name to a matrix cell and
confirming that re-averaging the file reproduces the cell's `psnr_mean`.

Published layout (flat, one file per protocol cell):

    results/per_image/<job>__<checkpoint>__<protocol>.csv

    e.g. results/per_image/generator_x4_x4__sr_generator_scale4_best__s42__P3.csv

Flat naming keeps a cell's provenance readable from the file name alone,
which matters because a reader is meant to be able to re-aggregate these
without reading the pipeline.

Usage:
    python3.11 scripts/build_public_release.py          # build
    python3.11 scripts/build_public_release.py --check  # verify only
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_per_image_mapping import (  # noqa: E402
    ARCHIVE_DIR_TO_JOB,
    classify,
    csv_mean_psnr,
    load_matrix_index,
    parse_cell_from_name,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(REPO, "archive_private")
RELEASE = os.path.join(REPO, "results", "per_image")

#: job -> human-readable task label, for the README.
JOB_LABELS = {
    "generator_x4_x4": "SR x4 (SRGenerator, 5 seeds x 8 protocols = 40 cells)",
    "srcnn_x2_x2": "SR x2 (SRCNN, 5 seeds x 8 protocols = 40 cells)",
    "lowlight_unet_x1": "Low-light (LowLightUNet, 5 seeds x 5 protocols = 25 cells)",
    "bicubic_sr_x4": "SR x4 bicubic baseline (8 protocols = 8 cells)",
    "bicubic_sr_x2": "SR x2 bicubic baseline (8 protocols = 8 cells)",
    "identity_ll": "Low-light identity baseline (5 protocols = 5 cells)",
}


def job_of(path: str) -> str | None:
    normalised = path.replace(os.sep, "/")
    for archive_dir, job in ARCHIVE_DIR_TO_JOB.items():
        if f"/{archive_dir}/" in normalised:
            return job
    return None


def verify_release(index: dict[tuple[str, str, str], float]) -> list[str]:
    """Check the published release without needing the private archive.

    This is the check that must work from a clone: the archive is not
    tracked, so a CI job or a reader cannot run the builder. Instead the
    published files are re-averaged and compared against the matrix JSONs,
    which are tracked.
    """
    problems: list[str] = []
    release = os.path.join(REPO, "results", "per_image")
    if not os.path.isdir(release):
        return [f"release directory missing: {os.path.relpath(release, REPO)}"]

    published = {n for n in os.listdir(release) if n.endswith(".csv")}
    expected = {f"{job}__{ckpt}__{proto}.csv" for (job, ckpt, proto) in index}
    for missing in sorted(expected - published):
        problems.append(f"missing release file: {missing}")
    for extra in sorted(published - expected):
        problems.append(f"release file with no matrix cell: {extra}")

    for name in sorted(published & expected):
        mean = csv_mean_psnr(os.path.join(release, name))
        # Split from the right: the checkpoint key itself contains "__"
        # (for example "sr_generator_scale4_best__s42"), so the protocol is
        # the last field and the job is the first.
        job, _, rest = name[:-4].partition("__")
        checkpoint, _, protocol = rest.rpartition("__")
        expected_mean = index.get((job, checkpoint, protocol))
        if expected_mean is None:
            problems.append(f"{name}: does not parse to a known matrix cell")
            continue
        if mean is None:
            problems.append(f"{name}: no finite PSNR values")
        elif abs(mean - expected_mean) > 1e-3:
            problems.append(
                f"{name}: recomputed {mean:.4f} != matrix {expected_mean:.4f}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the published release (works from a clone)")
    args = parser.parse_args()

    index = load_matrix_index()

    if args.check:
        problems = verify_release(index)
        if problems:
            print(f"[fail] {len(problems)} problem(s):", file=sys.stderr)
            for line in problems[:20]:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"[ok] published release: {len(index)} files, each reproducing its matrix cell")
        return 0

    candidates = sorted(
        glob.glob(os.path.join(ARCHIVE, "**", "per_image", "*.csv"), recursive=True)
    )

    # Collect the release set: name-derived cell plus value confirmation.
    release: dict[str, str] = {}
    for path in candidates:
        status, detail = classify(path, index)
        if status != "match":
            continue
        job, checkpoint, protocol = detail.split("/")
        name = f"{job}__{checkpoint}__{protocol}.csv"
        release[name] = path

    got_cells = {f"{job}__{ckpt}__{proto}.csv" for (job, ckpt, proto) in index}
    missing = got_cells - set(release)
    if missing:
        print(f"[fail] {len(missing)} matrix cell(s) have no matching archived CSV:",
              file=sys.stderr)
        for name in sorted(missing)[:10]:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"release set: {len(release)} files, covering {len(got_cells)} matrix cells")

    # ---- write the release ------------------------------------------------
    if os.path.isdir(RELEASE):
        shutil.rmtree(RELEASE)
    os.makedirs(RELEASE, exist_ok=True)
    for name, src in release.items():
        shutil.copyfile(src, os.path.join(RELEASE, name))

    total = 0
    for name in sorted(release):
        total += os.path.getsize(os.path.join(RELEASE, name))

    readme = os.path.join(RELEASE, "README.md")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write("# Per-image evaluation values\n")
        handle.write(
            "\nEvery protocol cell behind the paper's tables, released as the\n"
            "per-image PSNR/SSIM values that were averaged to produce it.\n"
        )
        handle.write(
            "\n## What is here\n"
            f"\n{len(release)} CSV files, one per (task, seed, protocol) cell -- "
            f"{len(got_cells)} distinct cells in total.\n"
        )
        for job, label in JOB_LABELS.items():
            count = sum(1 for n in release if n.startswith(job + "__"))
            handle.write(f"\n- `{job}/` -- {label}\n")
        handle.write(
            "\n## File naming\n"
            "\n    <job>__<checkpoint>__<protocol>.csv\n"
            "\nfor example\n"
            "\n    generator_x4_x4__sr_generator_scale4_best__s42__P3.csv\n"
            "\nmeans: the x4 super-resolution generator, seed 42, protocol P3\n"
            "(full image, RGB channel, bicubic downsampling with antialiasing).\n"
        )
        handle.write(
            "\n## File format\n"
            "\n    image,psnr,ssim\n"
            "\nOne row per test image. `psnr` is in dB, `ssim` is the standard\n"
            "per-image SSIM.\n"
            "\n**Non-finite PSNR values.** A bicubic upsampler can reconstruct a\n"
            "flat patch exactly, so MSE is 0 and PSNR is `+inf`. Such rows are\n"
            "kept in the file (so the file is a faithful record) but must be\n"
            "dropped before averaging: the reported means for the affected\n"
            "protocols are taken over the finite subset only. This is the\n"
            "reason P1's bicubic baseline is an average over 95 images while\n"
            "the model side is an average over 100 -- see `paper/FACTS.md`\n"
            "section 10, entry E21.\n"
        )
        handle.write(
            "\n## How to reproduce the paper's tables\n"
            "\nRe-average each file to get the cell value, then average the\n"
            "five seeds per (task, protocol) to get the row in the results\n"
            "table, and take the sample standard deviation across those five\n"
            "seed means for the spread column.\n"
            "\nTwo scripts in the repository do this and check the result\n"
            "against the released aggregates:\n"
            "\n    python3.11 scripts/verify_public_release.py   # re-derive every cell\n"
            "    python3.11 scripts/make_tables.py --check      # re-derive every table\n"
        )
        handle.write(
            "\n## Provenance\n"
            "\nEach file was copied from the private evaluation archive after\n"
            "verifying that re-averaging it reproduces the corresponding\n"
            "`psnr_mean` in `results/matrix_runs/<job>/protocol_matrix.json`.\n"
            "Files from superseded or failed runs were excluded; the check is\n"
            "reproducible with `scripts/verify_per_image_mapping.py`.\n"
            "\nThe aggregate matrix itself (`results/matrix_runs/`) and the\n"
            "derived tables (`results/*.csv`) are in the same repository.\n"
        )

    print(f"wrote {len(release)} files + README to results/per_image/ "
          f"({total / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
