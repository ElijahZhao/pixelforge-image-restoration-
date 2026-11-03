"""把 Kaggle 内核的执行记录整理成一份算力台账。

为什么需要这个脚本
------------------
Kaggle 的内核页面只告诉你"这次跑完了"，不告诉你"一共花了多久、失败过几次、
返工烧掉多少卡时"。论文里要写清实验投入，附录里要能复算，就必须有一份
逐次 run 的台账，而不是凭印象写"跑了几个小时"。

数据来源（全部可重新拉取，不需要手工记录）
------------------------------------------
1. ``kaggle kernels logs <owner>/<slug>/<version>``
   返回该版本的执行日志。每条事件带 ``time`` 字段（相对内核启动的秒数），
   取最大值即该版本的墙钟时长（含环境安装、数据下载）。
2. ``kaggle kernels files <owner>/<slug>/<version> --format csv``
   返回该版本产物清单，其中 ``creationDate`` 是版本创建（push）时刻，
   即内核开始运行的近似时间。
3. 仓库内 ``results/seeds/*/`` 的训练日志（脚本自报的 ``wall_s``）
   用于区分"内核墙钟"与"训练净时长"两个口径——前者含 3.5 GB DIV2K 下载
   与 pip 安装，后者才是模型真正在算的时间。

口径说明
--------
- ``duration_s``：内核墙钟，从 env 探测到最后一个事件。
- ``device``：``T4`` 计 GPU 配额，``CPU`` 不计。对账内核刻意跑在 CPU 上，
  因为它的目的是"任何人 clone 下来都能复现"，不能依赖 GPU。
- ``outcome``：人工判读。脚本无法从日志可靠地判断"这次结果是否被采纳"，
  所以维护一张 ``VERDICTS`` 表，缺项会以 ``?`` 标出而不是猜。

用法
----
    python scripts/kernel_ledger.py --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

OWNER = "yvlinzhao"

# 版本号 -> (硬件, 这次在干什么, 结局)。结局是人工判读，不自动推断。
VERDICTS: dict[tuple[str, int], tuple[str, str, str]] = {
    ("pixelforge-smoke", 1): ("T4", "环境自检：torch / CUDA / 磁盘 / 内存", "✅ 通过"),
    ("pixelforge-smoke", 2): ("T4", "环境自检（补 GPU 型号与显存）", "✅ 通过"),
    ("pixelforge-smoke2", 1): ("T4", "挂载探测：在 /kaggle/input 找固件", "❌ 输入目录未挂载"),
    ("pixelforge-smoke2", 2): ("T4", "挂载探测：确认 working 目录可写", "✅ 通过"),
    ("pixelforge-io-probe", 1): ("T4", "IO 探测：写盘位置 / 软链 / 缓存复用", "✅ 通过"),
    ("pixelforge-eval-protocol", 1): ("T4", "首次对外评估：SR×2 / SR×4，100 张 DIV2K val", "⚠️ 拿到 27.47 / 32.35，随后低光分支崩溃"),
    ("pixelforge-eval-protocol", 2): ("T4", "重跑评估，补 cross-check", "❌ LOL val 路径不存在"),
    ("pixelforge-eval-protocol", 3): ("T4", "重跑评估：改从 HF parquet 取 LOL", "❌ train/datasets.py 遮蔽了 HF datasets"),
    ("pixelforge-eval-protocol", 4): ("T4", "最终评估：三任务 + 逐图明细", "✅ 5 个产物，此后所有表格引用它"),
    ("pixelforge-protocol-y", 1): ("T4", "Y 通道 + shave 口径复评，6 个口径一次跑完", "✅ P5 锚点来源"),
    ("pixelforge-perceptual-ablation", 1): ("T4", "感知消融：软链已存在，启动即挂", "❌ FileExistsError"),
    ("pixelforge-perceptual-ablation", 2): ("T4", "感知消融：VGG 感知开 / 关各训一次 + 评估", "✅ 权重与日志齐全"),
    ("pixelforge-protocol-matrix", 1): ("T4", "协议矩阵首跑：两版权重 × 多协议", "⚠️ 第二版权重找不到，只跑了一组"),
    ("pixelforge-protocol-matrix", 2): ("T4", "协议矩阵：改权重查找路径", "❌ 仍找不到第二版权重"),
    ("pixelforge-protocol-matrix", 3): ("T4", "协议矩阵：改用 base64 内嵌权重", "❌ 内嵌模块没生成"),
    ("pixelforge-protocol-matrix", 4): ("T4", "协议矩阵：核对内嵌权重体积", "❌ 仍失败"),
    ("pixelforge-protocol-matrix", 5): ("T4", "协议矩阵：两版权重 × 8 协议全跑", "✅ P1–P8 首份完整矩阵"),
    ("pixelforge-protocol-repro-v6", 1): ("T4", "独立复现：绕开主仓脚本，直接读权重算协议", "✅ 与主仓脚本逐位一致"),
    ("pixelforge-protocol-repo", 1): ("CPU", "对账 v1：P3 / P5 / P6，x4 三协议", "❌ 锚点配错，判 FAIL"),
    ("pixelforge-protocol-repo", 2): ("CPU", "对账 v2：commit pin 改成短哈希", "❌ 断言拦截：HEAD 全长哈希 != 短哈希"),
    ("pixelforge-protocol-repo", 3): ("CPU", "对账 v3：P8 / P7 双协议 × 两个尺度", "❌ x2 单协议超 7200 s 被 kill"),
    ("pixelforge-protocol-repo", 4): ("CPU", "对账 v4：x2 只跑 P7，timeout 收到 6600 s", "✅ 四锚点全过"),
    ("pixelforge-seed-matrix", 1): ("T4", "种子矩阵冒烟：SRCNN ×1，少轮", "✅ 测速样本"),
    ("pixelforge-seed-matrix", 2): ("T4", "种子矩阵冒烟：重跑核对耗时", "✅ 测速样本"),
    ("pixelforge-seed-matrix", 3): ("T4", "低光分支冒烟：2 轮，lr=0", "✅ 测速样本"),
    ("pixelforge-seed-matrix", 4): ("T4", "A 批：SRCNN ×2，种子 42–46，100 轮", "✅ n=5，30.686 ± 0.069"),
    ("pixelforge-seed-matrix", 5): ("T4", "B 批：SRResNet ×4，种子 42–44，200 轮", "✅ n=3，26.430 ± 0.027"),
    ("pixelforge-seed-matrix", 6): ("T4", "C 批：SRResNet ×4，种子 45–46，200 轮", "✅ n=5 补齐"),
}

# 已知内核；其余内核不会被扫。
KNOWN = [
    "pixelforge-smoke",
    "pixelforge-smoke2",
    "pixelforge-io-probe",
    "pixelforge-eval-protocol",
    "pixelforge-protocol-y",
    "pixelforge-perceptual-ablation",
    "pixelforge-protocol-matrix",
    "pixelforge-protocol-repro-v6",
    "pixelforge-protocol-repo",
    "pixelforge-seed-matrix",
]


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout


def fetch_logs(kernel: str, version: int, cache_dir: str | None = None) -> tuple[float, int] | None:
    # 内核正在运行时 API 偶发返回空，重试；真没有就是没跑过这个版本。
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{kernel}__v{version}.json")
        if os.path.exists(cache):
            try:
                events = json.load(open(cache))
                if events:
                    return max(e["time"] for e in events), len(events)
            except (json.JSONDecodeError, KeyError, OSError):
                pass
    for attempt in range(3):
        rc, out = _run(["kaggle", "kernels", "logs", f"{OWNER}/{kernel}/{version}"])
        if rc == 0 and out.strip() not in ("", "[]"):
            try:
                events = json.loads(out)
            except json.JSONDecodeError:
                events = []
            if events:
                if cache:
                    with open(cache, "w") as f:
                        f.write(out)
                return max(e["time"] for e in events), len(events)
        time.sleep(1.5 * (attempt + 1))
    return None


def fetch_files(kernel: str, version: int) -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = []
    token = None
    for _ in range(10):
        cmd = ["kaggle", "kernels", "files", f"{OWNER}/{kernel}/{version}", "--format", "csv", "--page-size", "200"]
        if token:
            cmd += ["--page-token", token]
        rc, out = _run(cmd)
        if rc != 0:
            break
        token = None
        kept = []
        for line in out.splitlines():
            if line.startswith("Next Page Token = "):
                token = line[len("Next Page Token = "):].strip()
            elif line.strip():
                kept.append(line)
        for row in csv.DictReader(io.StringIO("\n".join(kept))):
            if row.get("name"):
                names.append((row["name"], row.get("creationDate", "")))
        if not token:
            break
    return names


def last_runs() -> dict[str, datetime]:
    """slug -> 最近一次运行结束时刻。用于没有产物的内核（产物为空就取不到 creationDate）。"""
    rc, out = _run(["kaggle", "kernels", "list", "-m", "--format", "csv", "--page-size", "100"])
    if rc != 0:
        return {}
    res: dict[str, datetime] = {}
    for row in csv.DictReader(io.StringIO(out)):
        ref, t = row.get("ref", ""), row.get("lastRunTime", "")
        if not ref or not t:
            continue
        try:
            res[ref.split("/")[-1]] = datetime.strptime(t[:19].replace(" ", "T"), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return res


def version_start(files: list[tuple[str, str]]) -> datetime | None:
    stamps = {t for _, t in files if t}
    if not stamps:
        return None
    parsed = []
    for t in stamps:
        try:
            parsed.append(datetime.strptime(t, "%I:%M %p, %A %d %B %Y UTC"))
        except ValueError:
            continue
    return min(parsed) if parsed else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-versions", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cache-dir", default=".cache/kaggle-logs",
                    help="执行日志本地缓存目录。Kaggle 对运行中内核的历史日志查询不稳，缓存可让重跑不再依赖 API。")
    args = ap.parse_args()

    rows, evidence = [], []
    fallback = last_runs()
    jobs = [(k, v) for k in KNOWN for v in range(1, args.max_versions + 1)]
    results: dict[tuple[str, int], tuple[float, int, list]] = {}

    def work(job):
        k, v = job
        logs = fetch_logs(k, v, args.cache_dir)
        if logs is None:
            return job, None
        files = fetch_files(k, v)
        return job, (logs[0], logs[1], files)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for job, res in ex.map(work, jobs):
            if res:
                results[job] = res

    for (kernel, version), (dur, n_events, files) in results.items():
        device, task, outcome = VERDICTS.get((kernel, version), ("?", "?", "?"))
        start = version_start(files)
        if start is None:
            # 没有产物的内核（比如只打印探测信息的 io-probe）拿不到 creationDate。
            # 用"最近一次运行结束时刻 - 时长"反推；只对最新版本成立，旧版本留空。
            end = fallback.get(kernel)
            if end is not None:
                start = end - timedelta(seconds=dur)
        rows.append({
            "kernel": kernel,
            "version": version,
            "start_utc": start.strftime("%Y-%m-%d %H:%M") if start else "",
            "end_utc": (start + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M") if start else "",
            "duration_s": round(dur, 1),
            "duration_min": round(dur / 60, 1),
            "duration_h": round(dur / 3600, 2),
            "device": device,
            "task": task,
            "outcome": outcome,
            "n_files": len(files),
            "n_log_events": n_events,
        })
        evidence.append({
            "kernel": kernel,
            "version": version,
            "duration_s": round(dur, 1),
            "n_log_events": n_events,
            "files": sorted({n for n, _ in files}),
        })

    rows.sort(key=lambda r: (r["start_utc"], r["kernel"], r["version"]))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "compute_ledger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "kernel_runs.json"), "w") as f:
        json.dump(evidence, f, indent=1, ensure_ascii=False)

    gpu = sum(r["duration_h"] for r in rows if r["device"] == "T4")
    cpu = sum(r["duration_h"] for r in rows if r["device"] == "CPU")
    bad = sum(r["duration_h"] for r in rows if r["outcome"].startswith("❌"))
    print(f"runs={len(rows)} total={gpu + cpu:.2f}h gpu={gpu:.2f}h cpu={cpu:.2f}h failed={bad:.2f}h")
    unknown = [r for r in rows if r["device"] == "?"]
    if unknown:
        print("未标注结局的 run：", [(r["kernel"], r["version"]) for r in unknown])
    return 0


if __name__ == "__main__":
    sys.exit(main())
