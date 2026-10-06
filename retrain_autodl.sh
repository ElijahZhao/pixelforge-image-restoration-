#!/usr/bin/env bash
# Pixelforge 重训脚本（在 AutoDL 实例的 tmux 会话内运行）
#
# 前置：
#   1. 项目已在 /root/autodl-tmp/pixelforge
#   2. data/ 下已放好 DIV2K 与 LOL-v1（见 data/README.md；不随仓库分发）
#
# 跑法：tmux new -s train  →  bash retrain_autodl.sh  →  Ctrl-B D 挂起
set -euo pipefail

cd /root/autodl-tmp/pixelforge || { echo "请先 cd 到项目目录"; exit 1; }
PY="${PYTHON:-python3}"

# ===== Step 0 · 跑测试 =====
"$PY" -m train.tests.run_tests

# ===== Step 1 · 备份旧产物 =====
mkdir -p /root/autodl-tmp/backup_old
cp -r results models serve/models /root/autodl-tmp/backup_old/ 2>/dev/null || true
echo "[备份] 旧结果在 /root/autodl-tmp/backup_old/"

# ===== Step 2 · 冒烟测试（2 epoch）=====
"$PY" train/train.py --task sr --model generator --scale 4 \
  --data_root data --epochs 2 --batch_size 8 --lr 1e-4 --perceptual

# ===== Step 3 · SR 重训 =====
# 两件事：
#   1. 旧 ×4 权重的 best checkpoint 是在随机裁剪的验证噪声上选出来的
#      （后 50 轮波动 2.15 dB，而真实增益仅 +0.46 dB），需要重训。
#   2. ×2 权重从未训练过，一并补上。
# 验证现已改为中心裁剪，指标可跨轮比较，best checkpoint 的选择才有意义。
"$PY" train/train.py --task sr --model generator --scale 4 \
  --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual \
  2>&1 | tee train_sr_scale4.log
echo "[SR x4 完成] models/sr_generator_scale4_best.pth"

"$PY" train/train.py --task sr --model generator --scale 2 \
  --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual \
  2>&1 | tee train_sr_scale2.log
echo "[SR x2 完成] models/sr_generator_scale2_best.pth"

# ===== Step 4 · 低光训练 =====
# 旧权重的验证管线把 400x600 压成了 128x128，验证指标近乎常数（后 50 轮
# 极差 0.22 dB）——那是管线失效的症状，不是收敛。此处按修复后的管线重训。
"$PY" train/train.py --task lowlight --model generator \
  --data_root data --epochs 200 --batch_size 8 --lr 2e-4 \
  2>&1 | tee train_lowlight.log
echo "[低光完成] models/lowlight_generator_scale2_best.pth"

# ===== Step 5 · 导出 TorchScript =====
# 文件名必须与 serve/model_loader.py 的 glob 一致：
#   sr_*_scale{scale}.pt  /  lowlight.pt
"$PY" train/export.py --checkpoint models/sr_generator_scale4_best.pth \
  --out serve/models/sr_generator_scale4.pt --task sr --scale 4
"$PY" train/export.py --checkpoint models/sr_generator_scale2_best.pth \
  --out serve/models/sr_generator_scale2.pt --task sr --scale 2
"$PY" train/export.py --checkpoint models/lowlight_generator_scale2_best.pth \
  --out serve/models/lowlight.pt --task lowlight --scale 2

echo "===== DONE ====="
echo "发回以下文件以便复算指标："
echo "  results/train_log_sr_generator.csv"
echo "  results/train_log_lowlight_generator.csv"
echo "  train_sr_scale4.log  train_sr_scale2.log  train_lowlight.log"
echo "  serve/models/ 下的三个 .pt（用于校验 sha256）"
