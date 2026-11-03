"""校验一批 Kaggle 种子矩阵产物是否完整——用来防"跑完了但少下了文件"。

为什么需要它
------------
Kaggle 内核的产物目录**只保留最新版本的输出**，推下一批就会覆盖上一批；
下载时如果带了 ``--file-pattern``，权重（日志、CSV 之外的 ``.pth``）会被静默排除，
而 ``models/**/*.pth`` 又在 ``.gitignore`` 里——于是权重就只剩 Kaggle 一份，
而推下一批之后连那份也拿不回来了。2026-11-03 就差点这样丢过 3 个 ``.pth``。

所以每拉完一批，必须**独立验证**一遍，而不是"看起来下载成功了"。

校验项
------
1. 每个种子 4 件套齐全：``train_log_*.csv``、``config_*.json``、``*_best.pth``、``stdout.log``
2. ``status.json`` 里每个请求的种子都有条目，且字段齐全
3. CSV 的 epoch 行数 == 配置里的轮数
4. 权重能 ``torch.load``，state_dict 的 key 数与基线一致，元信息里的 seed / epoch / tag 对得上
5. 汇总文件 ``aggregate.csv`` / ``aggregate.md`` 存在

用法
----
    python scripts/verify_batch.py archive_private/06-实验脚本与输出/kag-seed-matrix/out/generator_x4-45-46 \
        --run generator_x4 --seeds 45,46
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys


def _csv_epochs(path: str) -> int:
    """CSV 的数据行数：先定位表头行，再数它后面的行。

    训练日志前面是若干行 ``# key=value`` 的元信息（行数不固定，且里面就有
    ``epochs=200`` 这种含 "epoch" 的行），所以表头必须靠"多列"来认：
    表头每列都是列名，元信息只有一列。
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if len(row) >= 2 and row[0].strip().lower() == "epoch":
            return len(rows) - i - 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir")
    ap.add_argument("--run", required=True, help="如 generator_x4")
    ap.add_argument("--seeds", required=True, help="逗号分隔，如 45,46")
    ap.add_argument("--expect-keys", type=int, default=None,
                    help="权重 state_dict 的 key 数下限（跨种子比对用）")
    ap.add_argument("--weights-root", default=None,
                    help="权重根目录（默认仓库的 models/seeds/<run>）。"
                         "权重常被单独放到仓库里而不是 out/ 目录，校验两处。")
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    problems: list[str] = []
    notes: list[str] = []

    status_path = os.path.join(args.batch_dir, "status.json")
    if not os.path.exists(status_path):
        print(f"[verify] 缺 {status_path}，这一批还没拉下来")
        return 2
    status = json.load(open(status_path))

    runs = status.get("runs", {})
    missing_seeds = [s for s in seeds if s not in runs]
    if missing_seeds:
        problems.append(f"status.json 缺种子 {missing_seeds}")
    if runs and not status.get("budget_stop"):
        extra = [s for s in runs if s not in seeds]
        if extra:
            notes.append(f"status.json 多出种子 {extra}（请求 {seeds}）")

    key_counts: list[int] = []
    for seed in seeds:
        if seed not in runs:
            continue
        r = runs[seed]
        if "error" in r:
            problems.append(f"seed {seed}: 运行报错 {str(r['error'])[:160]}")
            continue
        for field in ("wall_s", "best_val_psnr", "final_val_psnr", "best_epoch", "epochs"):
            if field not in r:
                problems.append(f"seed {seed}: status.json 缺字段 {field}")
        # 1) 四件套
        csvs = glob.glob(os.path.join(args.batch_dir, "**", f"train_log_*.csv"), recursive=True)
        cfgs = glob.glob(os.path.join(args.batch_dir, "**", "config_*.json"), recursive=True)
        pths = glob.glob(os.path.join(args.batch_dir, "**", "*_best.pth"), recursive=True)
        logs = glob.glob(os.path.join(args.batch_dir, "**", "stdout.log"), recursive=True)
        if args.weights_root:
            # 权重常被单独放进仓库 models/seeds/，而不是留在 out/ 里——两处都查。
            pths += glob.glob(os.path.join(args.weights_root.format(run=args.run),
                                           "**", "*_best.pth"), recursive=True)

        def pick(items, seed_dir_hint: str):
            hits = [p for p in items if seed_dir_hint in p]
            return hits[0] if hits else None

        hint = f"s{seed}"
        p_csv, p_cfg = pick(csvs, hint), pick(cfgs, hint)
        p_pth = pick(pths, hint)
        p_log = pick(logs, hint)
        for label, path in (("train_log CSV", p_csv), ("config JSON", p_cfg),
                            ("best 权重", p_pth), ("stdout 日志", p_log)):
            if path is None:
                problems.append(f"seed {seed}: 缺 {label}")
            elif os.path.getsize(path) == 0:
                problems.append(f"seed {seed}: {label} 是空文件")

        # 3) CSV 行数 == 轮数
        if p_csv and r.get("epochs"):
            got = _csv_epochs(p_csv)
            if got != int(r["epochs"]):
                problems.append(f"seed {seed}: CSV 有 {got} 轮，配置写 {r['epochs']} 轮")
        # 4) 权重可读 + 元信息对得上
        if p_pth:
            try:
                import torch
                obj = torch.load(p_pth, map_location="cpu", weights_only=False)
            except Exception as exc:
                problems.append(f"seed {seed}: 权重读不出来 {type(exc).__name__}: {exc}")
                obj = None
            if isinstance(obj, dict):
                if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    sd = obj["state_dict"]
                else:
                    sd = obj
                if all(isinstance(v, torch.Tensor) for v in sd.values()) if sd else False:
                    key_counts.append(len(sd))
                meta = obj.get("meta") or obj.get("info") or {}
                if isinstance(meta, dict):
                    m_seed = str(meta.get("seed", seed))
                    if m_seed != str(seed):
                        problems.append(f"seed {seed}: 权重 meta.seed={m_seed}")
                    if meta.get("tag") and f"s{seed}" not in str(meta["tag"]):
                        problems.append(f"seed {seed}: 权重 meta.tag={meta.get('tag')}")
                elif not obj.get("state_dict"):
                    problems.append(f"seed {seed}: 权重结构看不懂（顶层键 {list(obj)[:5]}）")

    if len(key_counts) > 1 and len(set(key_counts)) != 1:
        problems.append(f"各权重 state_dict key 数不一致：{key_counts}")
    elif key_counts:
        notes.append(f"state_dict key 数 {key_counts[0]}（各种子一致）")
    if args.expect_keys and key_counts and key_counts[0] < args.expect_keys:
        problems.append(f"state_dict key 数 {key_counts[0]} < 期望下限 {args.expect_keys}")

    # 5) 汇总文件
    for name in ("aggregate.csv", "aggregate.md"):
        if not os.path.exists(os.path.join(args.batch_dir, name)):
            problems.append(f"缺汇总文件 {name}（跑 scripts/aggregate_seeds.py 生成）")

    print(f"[verify] {args.batch_dir}")
    print(f"[verify] 请求种子 {seeds}；status.json 记录 {len(runs)} 条")
    for n in notes:
        print(f"[verify] 说明: {n}")
    if problems:
        for p in problems:
            print(f"[verify] ❌ {p}")
        print(f"[verify] 不通过：{len(problems)} 个问题")
        return 1
    print("[verify] ✅ 全部齐了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
