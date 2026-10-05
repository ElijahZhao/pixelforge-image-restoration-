# 项目说明（边界与分工）

> 本文档记录开发分工、已验证结果与当前边界。

---

## 一、这个项目是怎么做出来的

PixelForge 是一个在 AI 辅助下完成的计算机视觉项目。分工如下：

| 环节 | 谁主导 |
|---|---|
| 需求与目标设定（要做什么、达到什么标准） | 作者 |
| 深度诊断（找真实缺陷） | 作者与 AI 协作 |
| 代码修复（修复方案与实现） | 作者与 AI 协作 |
| 训练环境搭建（AutoDL 租用、数据准备、实例克隆） | 作者 |
| 实际重训执行（SR×4 与低光各 200 epoch） | 作者 |
| 结果验证与迭代决策（发现口径问题、推动同口径复测） | 作者 |
| 文档与证据归档 | 作者与 AI 协作 |

**关于分工的说明**：训练、调试、验证的每个关键步骤，作者都是实际的执行者与决策者：租了 GPU、跑了训练、拿到了平台账单、发现了"低光看似退化其实口径不同"、推动了同口径复测。这些工作由作者完成，每一步都有第三方凭证（见下）。

---

## 二、能拿出什么证据

评审者可以自行核验以下材料：

| 证据 | 位置 | 说明 |
|---|---|---|
| 平台实例列表 | `docs/retrain_journey/09_*.png` | AutoDL RTX 3080 Ti 实例，含实例 ID 与运行状态 |
| 平台计费明细 | `docs/retrain_journey/10_*.png` | 带时间戳、金额、实例 ID 的扣费记录 |
| GPU 监控曲线 | `docs/retrain_journey/11_*.png` | 显存阶梯爬升 → 稳定 → 归零，对应真实训练 |
| 完整训练日志 | `docs/retrain_journey/train_*.log` | 200 epoch 逐轮输出 |
| 逐 epoch 指标 CSV | `results/train_log_*.csv` | 200 行原始数据，可复算 |
| 评测脚本 | `scripts/eval_baseline.py` | 同口径对比模型与基线，任何人可重跑 |
| 过程截图 | `docs/retrain_journey/01–08` | 从选机到完成的完整链路 |

上述材料的时间戳互相衔接：实例创建 → 计费扣款 → GPU 曲线 → 训练日志 → 评测结果。

---

## 三、边界

本项目未声称以下事情：

1. "代码 100% 由作者独立手写"：不实。代码与文档有 AI 参与的成分，已在第一节说明。
2. "模型达到 SOTA"：不实。SR×4 全图 PSNR 27.47，高于 bicubic 基线（+0.77 dB）但低于 DIV2K 大数据集上的学术 SOTA（30+），原因是训练集规模与网络容量有限。
3. "低光指标优于某某方法"：谨慎。低光的 +10.41 dB 是相对"不处理"基线，不是相对文献中其他低光方法。
4. "SRCNN ×2 已训练"：未训练，表中为 `TBD`。

本项目声称以下事情（均有证据）：

1. 端到端的完整流水线（数据→训练→评测→导出→服务→前端）可运行、可复现；
2. 修复后的模型在同口径下确实高于基线（SR +0.77 dB、低光 +10.41 dB）；
3. 训练在真实 GPU 上完成，有平台凭证；
4. 项目发现并修复了一个真实的 ML 缺陷（感知损失未归一化 + 权重被 `0.01` 抹除），并用重训验证了修复有效。

---

## 四、项目的技术要点

- **问题定位**：找出感知损失缺陷、隐藏的 4 倍假放大、验收判据错误等；这些不是看代码就能一眼看出的。
- **自我纠错**：不仅修复，还用变异测试验证"测试真的能抓到这类错误"；低光口径问题也是主动发现并澄清的。
- **工程闭环**：从数据到部署的每一步都能跑、能复现、有记录。
- **边界管理**：明确区分"已证明"与"未证明"，并说明 AI 辅助的参与范围。

> 项目难点通常不在"跑通"，而在能否判断自己哪里可能错、并且有办法验证。PixelForge 在这一方向上有完整的记录。

---

## 五、复现方式

```bash
git clone https://github.com/ElijahZhao/pixelforge-image-restoration-.git
cd pixelforge-image-restoration-

# 测试套件（pytest 28/28：20 训练单测 + 8 API 冒烟；亦可用 python -m train.tests.run_tests）
python -m pytest

# 基线评测（需数据在 data/div2k 与 data/lol）
python scripts/eval_baseline.py --task sr --scale 4
python scripts/eval_baseline.py --task lowlight

# 重训命令见 retrain_autodl.sh
```

详细成果见 [`PIXELFORGE_RETRAIN_RESULTS.md`](docs/history/PIXELFORGE_RETRAIN_RESULTS.md)，
开发期诊断记录见 `docs/history/`。
