#!/usr/bin/env python3
"""权威核对：把每个内核每个版本的产物**完整下载**一份，逐文件比 md5。

为什么不用 `kaggle kernels files` 的清单：
  - 那个接口的 `size` 字段不是字节数（4.9 MB 的权重报 940，2.3 MB 的报 948），
    拿它判大小会得到满屏假警报；
  - 它只给文件名列表，不给校验和。

所以这里的做法是：真下载。下载过程中 Kaggle 会写一个 `<file>.kaggle-partial`，
里面记着 `{"validator": ..., "size": 真实字节数}` —— 那才是权威大小。

比对策略：
  1. 远端 md5 == 本地某文件的 md5            → 已落盘（可能改名归档）
  2. 远端 md5 == 本地同名文件的 md5            → 已落盘（原名）
  3. 远端 md5 在本地找不到，且远端是"重复项"    → 同一份文件被列出多次，只查一次
  4. 远端 md5 在本地找不到                      → 真丢了，必须报出来

用法:
    python scripts/audit_kaggle_assets.py --out-dir /tmp/audit --dl-root /tmp/audit/_dl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

OWNER = "yvlinzhao"

KERNELS = {
    "pixelforge-smoke": range(1, 3),
    "pixelforge-smoke2": range(1, 3),
    "pixelforge-io-probe": range(1, 2),
    "pixelforge-eval-protocol": range(1, 5),
    "pixelforge-protocol-y": range(1, 2),
    "pixelforge-perceptual-ablation": range(1, 3),
    "pixelforge-protocol-matrix": range(1, 6),
    "pixelforge-protocol-repro-v6": range(1, 2),
    "pixelforge-protocol-repo": range(1, 5),
    "pixelforge-seed-matrix": range(1, 7),
    "pixelforge-seed-matrix-d": range(1, 2),
}

SKIP_DIRS = {".git", "backup", ".cache", "node_modules", "__pycache__",
             ".pytest_cache", ".playwright-cli"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_local(root: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """返回 (basename -> [路径], md5 -> [路径])。"""
    by_name: dict[str, list[str]] = {}
    by_md5: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            by_name.setdefault(fn, []).append(p)
            try:
                by_md5.setdefault(md5(p), []).append(p)
            except OSError:
                pass
    return by_name, by_md5


def download(kernel: str, version: int, dest_root: str) -> tuple[str, str | None]:
    """把一个版本完整下载到 dest_root/<kernel>/v<version>。

    版本不存在时 Kaggle 返回 404（"The resource was not found"）——那是
    "这个内核没有这么多版本"，不是错误，记为 None 让它从统计里排除。
    """
    dest = os.path.join(dest_root, kernel, f"v{version}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    ref = f"{OWNER}/{kernel}/{version}"
    p = subprocess.run(
        ["kaggle", "kernels", "output", ref, "-p", dest, "-o", "-q"],
        capture_output=True, text=True, timeout=3600)
    blob = f"{p.stdout} {p.stderr}"
    if "was not found" in blob or "404" in blob:
        shutil.rmtree(dest, ignore_errors=True)
        return ref, None            # 版本不存在，不算错误
    if p.returncode != 0:
        return ref, f"下载失败: {p.stdout[-200:]} {p.stderr[-200:]}"
    return ref, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--out-dir", default="/tmp/audit")
    ap.add_argument("--dl-root", default="/tmp/audit/_downloads")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [(k, v) for k, vs in KERNELS.items() for v in vs]
    results: dict[str, str | None] = {}
    if not args.skip_download:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download, k, v, args.dl_root): (k, v) for k, v in jobs}
            for fut in futs:
                pass
            for fut, (k, v) in futs.items():
                ref, err = fut.result()
                key = f"{k}/v{v}"
                results[key] = err
                print(f"  {key}: {'✗ ' + err[:80] if err else '✓ 下载完成'}", flush=True)

    # 收集远端文件（去重：同一路径被列出多次只算一次）
    remote: dict[str, str] = {}   # key -> 本地下载副本路径
    for k, v in jobs:
        base = os.path.join(args.dl_root, k, f"v{v}")
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith(".kaggle-partial"):
                    continue
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, base)
                remote[f"{k}/v{v}::{rel}"] = p

    print(f"\n远端去重后 {len(remote)} 个文件")

    by_name, by_md5 = index_local(args.root)
    print(f"本地索引：{len(by_name)} 个文件名，{len(by_md5)} 个 md5\n")

    matched = []
    unmatched = []
    for key, rp in sorted(remote.items()):
        rmd5 = md5(rp)
        rsize = os.path.getsize(rp)
        hits = by_md5.get(rmd5, [])
        if hits:
            matched.append({
                "key": key, "remote_md5": rmd5, "remote_size": rsize,
                "local": hits[0], "local_size": os.path.getsize(hits[0]),
                "local_all": hits,
            })
        else:
            unmatched.append({
                "key": key, "remote_md5": rmd5, "remote_size": rsize,
                "same_name_local": [
                    {"path": p, "size": os.path.getsize(p)}
                    for p in by_name.get(os.path.basename(key.split("::")[1]), [])
                ],
            })

    report = {
        "remote_unique_files": len(remote),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "matched_detail": matched,
        "unmatched_detail": unmatched,
        "download_errors": {k: v for k, v in results.items() if v},
    }
    json.dump(report, open(f"{args.out_dir}/audit.json", "w"),
              indent=1, ensure_ascii=False)

    print("=" * 72)
    print(f"远端文件（去重）：{len(remote)}")
    print(f"  md5 命中本地：{len(matched)}")
    print(f"  未命中：      {len(unmatched)}")
    if report["download_errors"]:
        print(f"\n下载失败的版本：{len(report['download_errors'])}")
        for k, v in report["download_errors"].items():
            print(f"  {k}: {v[:120]}")
    if unmatched:
        print("\n未命中明细：")
        for u in unmatched[:60]:
            print(f"  {u['key']}  md5={u['remote_md5']} size={u['remote_size']}")
            for s in u["same_name_local"]:
                print(f"      本地同名 {s['path']} size={s['size']}"
                      f"{'  ← 同名但内容不同' if s['size'] != u['remote_size'] else ''}")
    print(f"\n报告：{args.out_dir}/audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
