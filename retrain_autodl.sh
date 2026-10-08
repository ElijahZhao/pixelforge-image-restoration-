#!/usr/bin/env bash
# Pixelforge 重训脚本（在 AutoDL 实例的 tmux 会话内运行）
#
# 前置：
#   1. 项目已在 /root/autodl-tmp/pixelforge
#   2. data/ 下已放好 DIV2K 与 LOL-v1（数据准备说明见私有归档 archive_private/docs/RETRAIN_DATA_GUIDE.md）
#      自检：bash scripts/download_data.sh --check   （会核对文件数量）
#
# 跑法：
#   tmux new -s train
#   bash retrain_autodl.sh
#   Ctrl-B 然后 D 挂起（断线不掉）
#
# 中断后直接重跑本脚本即可：训练带 --auto-resume，有断点就续、没有就开始。
# 若某个阶段已完成、不想重跑，用开关跳过，例如：
#   SKIP_SMOKE=1 SKIP_SR4=1 bash retrain_autodl.sh
set -euo pipefail

cd /root/autodl-tmp/pixelforge || { echo "请先 cd 到项目目录"; exit 1; }
PY="${PYTHON:-python3}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-16}"

SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_SR4="${SKIP_SR4:-0}"
SKIP_SR2="${SKIP_SR2:-0}"
SKIP_LOWLIGHT="${SKIP_LOWLIGHT:-0}"

# ===== Step 0 · 数据自检 + 跑测试 =====
# `download_data.sh --check` exits 1 when data is missing, which under `set -e`
# would kill this script mid-way through a wall of download instructions. Catch
# it explicitly so the failure is one clear sentence pointing at the guide.
if ! bash scripts/download_data.sh --check; then
  echo
  echo "================================================================"
  echo "数据集未就绪，训练无法开始。"
  echo "准备步骤见私有归档 archive_private/docs/RETRAIN_DATA_GUIDE.md，或重跑："
  echo "    bash scripts/download_data.sh               # 打印下载指引"
  echo "    bash scripts/download_data.sh --check       # 校验结构与文件数量"
  echo "如果你有意只用数据子集，加 --allow-partial 显式放行数量不足。"
  echo "================================================================"
  exit 1
fi

"$PY" -m train.tests.run_tests

# ===== Step 1 · 备份旧产物 =====
mkdir -p /root/autodl-tmp/backup_old
cp -r results models serve/models /root/autodl-tmp/backup_old/ 2>/dev/null || true
echo "[备份] 旧结果在 /root/autodl-tmp/backup_old/"

# ===== Step 2 · 冒烟测试（2 轮，确认 GPU 上跑通）=====
if [[ "$SKIP_SMOKE" != "1" ]]; then
  "$PY" train/train.py --task sr --model generator --scale 4 \
    --data_root data --epochs 2 --batch_size "$BATCH" --lr 1e-4 --perceptual
fi

# ===== Step 3 · SR x4 重训 =====
# 旧 x4 权重的 best 落在第 101 轮（27.39 dB），而它前后各轮都在 25.3-26.6 之间：
# 那个峰值由验证裁剪的运气决定，不是模型质量。验证已改为中心裁剪，
# 现在选出的 checkpoint 才真正是"最好的那个"。
if [[ "$SKIP_SR4" != "1" ]]; then
  "$PY" train/train.py --task sr --model generator --scale 4 \
    --data_root data --epochs "$EPOCHS" --batch_size "$BATCH" --lr 1e-4 --perceptual \
    --save_every 10 --auto-resume \
    2>&1 | tee train_sr_scale4.log
  echo "[SR x4 完成] models/sr_generator_scale4_best.pth"
fi

# ===== Step 4 · SR x2（全新训练）=====
if [[ "$SKIP_SR2" != "1" ]]; then
  "$PY" train/train.py --task sr --model generator --scale 2 \
    --data_root data --epochs "$EPOCHS" --batch_size "$BATCH" --lr 1e-4 --perceptual \
    --save_every 10 --auto-resume \
    2>&1 | tee train_sr_scale2.log
  echo "[SR x2 完成] models/sr_generator_scale2_best.pth"
fi

# ===== Step 5 · 低光重训 =====
# 旧权重的验证管线把 400x600 压成 128x128，后 100 轮指标近乎常数（标准差 0.0457）
# ——那是管线失效，不是收敛。尺子坏了，选出来的 checkpoint 就不可信。
#
# 不传 --model：低光一律构建 LowLightUNet（见 train/models.py 的 build_model），
# 该参数对结构没有影响，只决定产物文件名里的那个 token。省略它＝沿用默认
# `srcnn`，与已提交的权重名和日志名保持一致，避免同一份模型出现两种叫法。
if [[ "$SKIP_LOWLIGHT" != "1" ]]; then
  "$PY" train/train.py --task lowlight \
    --data_root data --epochs "$EPOCHS" --batch_size "$BATCH" --lr 2e-4 \
    --save_every 10 --auto-resume \
    2>&1 | tee train_lowlight.log
  echo "[低光完成] models/lowlight_srcnn_scale2_best.pth"
fi

# ===== Step 6 · 导出 TorchScript =====
# 文件名必须匹配 serve/model_loader.py 的 glob：sr_*_scale{scale}.pt / lowlight.pt
"$PY" train/export.py --checkpoint models/sr_generator_scale4_best.pth \
  --out serve/models/sr_generator_scale4.pt --task sr --scale 4
"$PY" train/export.py --checkpoint models/sr_generator_scale2_best.pth \
  --out serve/models/sr_generator_scale2.pt --task sr --scale 2
"$PY" train/export.py --checkpoint models/lowlight_srcnn_scale2_best.pth \
  --out serve/models/lowlight.pt --task lowlight --scale 2

# ===== Step 7 · 导出后自检 =====
"$PY" - <<'PYEOF'
from pathlib import Path
import torch
d = Path("serve/models")
for name in ["sr_generator_scale4.pt", "sr_generator_scale2.pt", "lowlight.pt"]:
    p = d / name
    if not p.exists():
        print(f"  [缺] {name}")
        continue
    m = torch.jit.load(str(p), map_location="cpu").eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = m(x)
    print(f"  [有] {name}  {p.stat().st_size/1e6:.1f} MB  前向 {tuple(x.shape)} -> {tuple(y.shape)}")
PYEOF

echo "===== DONE ====="
echo "发回以下文件以便复算指标："
echo "  results/train_log_sr_generator_scale4.csv"
echo "  results/train_log_sr_generator_scale2.csv"
echo "  results/train_log_lowlight_srcnn_scale2.csv"
echo "  train_sr_scale4.log  train_sr_scale2.log  train_lowlight.log"
echo "  serve/models/ 下三个 .pt"
