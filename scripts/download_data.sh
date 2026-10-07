#!/usr/bin/env bash
# 准备训练/评测所需的数据集目录结构。
#
# 说明：这个脚本**不会自动下载** DIV2K / LOL。两个数据集加起来好几个 GB，
# 而且各有自己的许可条款，自动拉取既慢又替使用者做了授权决定。
# 它做的是：建好目录、打印下载地址、最后校验结构是否就位。
#
# 用法：
#   bash scripts/download_data.sh                        # 检查与提示
#   bash scripts/download_data.sh --check                # 只校验，实测数量是否达标
#   bash scripts/download_data.sh --check --allow-partial  # 有意只用子集时放行数量不足
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DATA="$ROOT/data"

DIV2K_DIR="$DATA/div2k"
LOL_DIR="$DATA/lol"

echo "=== PixelForge 数据集准备 ==="
echo "数据根目录: $DATA"
echo

mkdir -p "$DIV2K_DIR/train" "$DIV2K_DIR/val"
mkdir -p "$LOL_DIR/train/low" "$LOL_DIR/train/high"
mkdir -p "$LOL_DIR/val/low"   "$LOL_DIR/val/high"

if [[ "${1:-}" != "--check" ]]; then
  cat <<EOF
目录已建好。请自行下载并解压到对应位置：

[DIV2K 超分]
  下载页: https://data.vision.ee.ethz.ch/cvl/DIV2K/
  需要:   DIV2K_train_HR.zip, DIV2K_valid_HR.zip
  解压后:
    data/div2k/train/   <- DIV2K_train_HR 里的 HR png
    data/div2k/val/     <- DIV2K_valid_HR 里的 HR png
  低清图不用准备，训练时用 bicubic 在线下采样（见 train/datasets.py）。

[LOL-v1 低光]
  来源:   https://github.com/weichen582/RetinexNet  (LOLdataset_v1)
  解压后:
    data/lol/train/low/    data/lol/train/high/     (low/high 文件名要一一对应)
    data/lol/val/low/      data/lol/val/high/

EOF
fi

echo "=== 结构校验 ==="
# 期望数量（官方划分）。少于这个数说明解压/拷贝中途漏了文件——
# 只看"目录非空"是不够的：漏掉 400 张图不会报错，只会让结果悄悄变差。
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
if [[ "${2:-}" == "--allow-partial" || "${1:-}" == "--allow-partial" ]]; then
  ALLOW_PARTIAL=1
fi

missing=0
partial=0
count() { find "$1" -type f \( -iname "*.png" -o -iname "*.jpg" \) 2>/dev/null | wc -l | tr -d ' '; }

check_dir() {  # $1=路径 $2=说明 $3=期望数量
  local n; n="$(count "$1")"
  if [[ "$n" -eq 0 ]]; then
    echo "  [缺] $2 ($1)  期望 $3"
    missing=1
  elif [[ "$n" -lt "$3" ]]; then
    if [[ "$ALLOW_PARTIAL" == "1" ]]; then
      echo "  [少] $2: $n 个文件（期望 $3，已用 --allow-partial 放行）"
    else
      echo "  [少] $2: $n 个文件，期望 $3"
      partial=1
    fi
  elif [[ "$n" -gt "$3" ]]; then
    echo "  [多] $2: $n 个文件（期望 $3，多余文件会被忽略，通常无害）"
  else
    echo "  [有] $2: $n 个文件"
  fi
}

check_dir "$DIV2K_DIR/train"    "DIV2K 训练图"    800
check_dir "$DIV2K_DIR/val"      "DIV2K 验证图"    100
check_dir "$LOL_DIR/train/low"  "LOL 训练 low"    485
check_dir "$LOL_DIR/train/high" "LOL 训练 high"   485
check_dir "$LOL_DIR/val/low"    "LOL 验证 low"    15
check_dir "$LOL_DIR/val/high"   "LOL 验证 high"   15

echo
if [[ "$missing" -eq 0 && "$partial" -eq 0 ]]; then
  echo "数据就绪。可以跑评测："
  echo "  python scripts/eval_baseline.py --task sr --scale 4"
  echo "  python scripts/eval_baseline.py --task lowlight"
elif [[ "$missing" -eq 0 && "$partial" -eq 1 ]]; then
  echo "数据不完整：有些目录的文件数少于官方划分。"
  echo "少一半的图不会在训练时报错，只会让结果悄悄变差。"
  echo "请确认解压/拷贝是否完整，补齐后重跑 --check。"
  echo "若你确实有意只用子集，可显式放行："
  echo "  bash scripts/download_data.sh --check --allow-partial"
  exit 1
else
  echo "数据未就绪。按上面的说明补齐后重跑本脚本的 --check 即可。"
  echo "注意：只想跑推理服务的话不需要这些数据，serve/ 自带权重，开箱即用。"
  exit 1
fi
