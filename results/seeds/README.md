# 种子矩阵日志（阶段 4）

误差棒的原始出处。每个 subdirectory 是**一个种子的一次完整训练**：CSV 是逐轮日志，
JSON 是该次运行的参数 + 协议项 + 数据量 + 环境快照。权重（`.pth`）不入库，只在
`models/seeds/` 本地保留——但只靠这些 CSV 就能验证报告里的每一个数，因为
`val_psnr` / `val_ssim` 是逐轮的，不是只留最后一轮。

## 为什么这些 CSV 必须进仓库

论文里"5 个种子 × 3 个模型"的误差棒如果只有一张表，读者无法判断它是 5 次独立训练
还是 3 次里挑了 2 次。逐轮日志是唯一能自证"确实跑了 5 次、每次都跑满"的东西。

## 现有批次

| 目录 | 模型 | 种子 | epochs | final PSNR | 状态 |
|---|---|---|---|---|---|
| `srcnn_x2/` | SRCNN ×2 | 42–46 | 100 | **30.686 ± 0.069** [30.601, 30.771] | 完成 2026-11-03 |

`aggregate.csv` / `aggregate.md` 由 `scripts/aggregate_seeds.py` 生成，可随时重算：

```bash
python scripts/aggregate_seeds.py --runs-root results --expect-seeds 5
```

## 三个读数陷阱

1. **30.686 不能和文献比，也不能和本仓对外表格的 32.35 相减。** 这是训练内验证：
   中心裁剪（96×96）、per-image mean、训练侧退化算子 `tv_bicubic_aa`。对外表格是
   全图 + `pil_bicubic`。两者差 1.66 dB 是口径差，不是进步，也不是退步。主表由阶段 5
   用 P1–P8 重评同一批 checkpoint 产出。
2. **`best == final`**：100 轮里最后一轮就是 val 最优。这批权重不适合演示
   "best-of-N 虚高"——那个现象来自随机裁剪口径（P1），不来自训练轮次。
3. **std 0.069 dB 是结论，不是缺陷。** SRCNN 在这个配方下种子不敏感且已收敛。
   要论证"种子敏感"，得看两个 generator（B/C/D 批），不能用 SRCNN 举例。

## 代码版本

全部 pin `30e50503805fc0654bb735a2e83391f313471e66`（v1.4.0）。后续批次**不跟主仓
前进**：15 个 checkpoint 必须来自同一份代码。主仓后来把 LR 构造收进
`train/degradation.py`，逐元素等价由 `train/tests/test_degradation.py` 钉住，但那是
测试的结论，不是这批权重的结论。

## 环境

Kaggle kernel `yvlinzhao/pixelforge-seed-matrix`（Tesla T4，私有，0 元）：
torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / CUDA 12.8。DIV2K 800 训练 + 100 验证，
kernel 内自取。单 run 3192–3251 s。
