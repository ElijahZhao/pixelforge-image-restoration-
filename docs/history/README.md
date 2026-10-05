# 诊断与重训记录（Diagnosis & Retrain Trail）

本目录按时间顺序保留项目开发期的诊断、修复与重训记录，供追溯。

> 归类说明：2026-10-04 项目整理时，将根目录散落的 26 份诊断/报告文档统一归入本目录，根目录仅保留 `README` / `PROGRESS` / `DEPLOY` / `PROJECT_NOTES` / `LICENSE`。

## 编号时间线

| 文档 | 主题 |
|---|---|
| `DIAGNOSIS_AND_PLAN.md` | 第 1 轮：感知损失权重失衡的根因诊断 + 修复计划 |
| `DIAGNOSIS_ROUND2.md` ~ `DIAGNOSIS_ROUND20.md` | 第 2–20 轮诊断（低光失败、VGG 归一化、serve 语义、运行时 DoS、指标不可复现、数据管线鲁棒性、AMP 误解、部署契约、架构保真度、叙述矛盾、demo 复现等） |
| `DIAGNOSIS_ROUND22.md` | 全项目审查修缮（安全 / 并发 / 复现性 / 边界缺陷，11 项） |
| `DIAGNOSIS_ROUND23.md` | 全项目审查修缮（部署依赖契约 / 测试有效性 / 死代码清理，7 项） |

> 编号说明：`ROUND1` 即本目录的 `DIAGNOSIS_AND_PLAN.md`（第 2/3 轮文中称其为「第一轮」）。

## 修复方案与成果报告

| 文档 | 内容 |
|---|---|
| `PIXELFORGE_FIX_PLAN.md` | 修复清单（含验证方法） |
| `PIXELFORGE_EXECUTION_REPORT.md` | 修复方案的实际执行结果记录 |
| `PIXELFORGE_RETRAIN_RESULTS.md` | 修复后重训成果（SR ×4 +0.77 dB / 低光 +10.41 dB，含口径说明） |
| `PIXELFORGE_RETROSPECTIVE.md` | 开发期诊断与修复的回顾 |

## 关联证据

- 训练过程与平台凭证（实例 / 计费 / GPU 显存曲线 / 逐 epoch 日志）：[`../retrain_journey/`](../retrain_journey/)
- 逐 epoch 指标：`../../results/train_log_*.csv`
- 同口径基线评测脚本：`../../scripts/eval_baseline.py`
