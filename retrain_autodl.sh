#!/usr/bin/env bash
# PixelForge 修复后重训脚本（在 AutoDL 克隆实例的 tmux 会话内运行）
# 前置：已把 pixelforge_fixed_train.zip 上传到 /root/autodl-tmp/
set -euo pipefail

cd /root/autodl-tmp/pixelforge || { echo "请先 cd 到你的项目目录"; exit 1; }
PY="${PYTHON:-python3}"   # AutoDL 多数镜像 python 就是 python3；若 python 可用可改 PY=python

# ===== Step 0 · 覆盖修复代码 + 跑测试 =====
unzip -o /root/autodl-tmp/pixelforge_fixed_train.zip -d /root/autodl-tmp/pixelforge
"$PY" -m train.tests.run_tests     # 期望输出 28 passed

# ===== Step 1 · 备份旧产物（防覆盖）=====
mkdir -p /root/autodl-tmp/backup_old
cp -r results models serve/models /root/autodl-tmp/backup_old/ 2>/dev/null || true
echo "[备份完成] 旧结果在 /root/autodl-tmp/backup_old/"

# ===== Step 2 · 冒烟测试（2 epoch，确认 GPU 上跑通）=====
"$PY" train/train.py --task sr --model generator --scale 4 \
  --data_root data --epochs 2 --batch_size 8 --lr 1e-4 --perceptual

# ===== Step 3 · 正式 SR×4 重训（前台，tmux 内跑即可，断线不掉）=====
"$PY" train/train.py --task sr --model generator --scale 4 \
  --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual \
  2>&1 | tee train_sr_percep_fixed.log
echo "[SR 重训完成] best checkpoint: models/sr_generator_scale4_best.pth"

# ===== Step 4 · 低光重训（接在 SR 后面，单卡顺序跑）=====
"$PY" train/train.py --task lowlight \
  --epochs 200 --batch_size 8 --lr 2e-4 \
  2>&1 | tee train_lowlight_fixed.log
echo "[低光重训完成] best checkpoint: models/lowlight_srcnn_scale2_best.pth"

# ===== Step 5 · 导出 TorchScript + 收尾 =====
"$PY" train/export.py --checkpoint models/sr_generator_scale4_best.pth \
  --out serve/models/sr_generator_scale4.pt --task sr --scale 4
"$PY" train/export.py --checkpoint models/lowlight_srcnn_scale2_best.pth \
  --out serve/models/lowlight.pt --task lowlight

echo "===== DONE ====="
echo "请把以下文件发回给我，以便复算指标："
echo "  results/train_log_sr_generator.csv"
echo "  results/train_log_lowlight_srcnn.csv"
echo "  train_sr_percep_fixed.log  train_lowlight_fixed.log"
