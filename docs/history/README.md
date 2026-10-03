# 诊断与审计归档（Diagnosis & Audit Trail）

本目录收录 PixelForge 从「能跑的玩具」到「有证据、可复现的真实 ML 项目」的完整迭代记录。
这些文档**刻意保留**，因为它们是项目最宝贵的资产之一：**真实踩过的坑、如何定位、如何修复、以及用重训证明修复有效**——这正是招生评审/技术面试官最想看到的「工程严谨性」证据，而非一次跑通的漂亮结果。

> 归类说明：2026-10-04 项目整理时，将根目录散落的 26 份诊断/报告文档统一归入本目录，根目录仅保留 `README` / `PROGRESS` / `DEPLOY` / `PROJECT_NOTES` / `LICENSE`，避免克隆仓库第一眼看到「诊断文档墙」。文档内容一字未改。

## 编号时间线

| 文档 | 主题 |
|---|---|
| `DIAGNOSIS_AND_PLAN.md` | **第一轮（ROUND1）**：感知损失权重失衡的根因诊断 + 修复计划（审计链起点） |
| `DIAGNOSIS_ROUND2.md` ~ `DIAGNOSIS_ROUND20.md` | 第 2–20 轮深度审计（低光失败、VGG 归一化、serve 语义、运行时 DoS、指标不可复现、数据管线鲁棒性、AMP 误解、部署契约、架构保真度、叙述矛盾、demo 复现等） |
| `DIAGNOSIS_ROUND22.md` | 全项目审查修缮（安全 / 并发 / 复现性 / 边界缺陷，11 项） |
| `DIAGNOSIS_ROUND23.md` | 全项目审查修缮（部署依赖契约 / 测试有效性「假通过」/ 死代码清理，7 项） |

> ⚠️ **编号说明**：`ROUND21` 在原始序列中**被跳过**（无对应文档）；`ROUND1` 即本目录的 `DIAGNOSIS_AND_PLAN.md`（第 2/3 轮文中称其为「第一轮」）。`ROUND22/23` 是项目后期追加的两轮全量审计，与早期 ROUND2–20 同属一套审计方法论。

## 修复方案与成果报告

| 文档 | 内容 |
|---|---|
| `PIXELFORGE_FIX_PLAN.md` | 可执行修复清单（F1–F14，纯方案，含验证方法） |
| `PIXELFORGE_EXECUTION_REPORT.md` | 修复方案的实际执行结果记录 |
| `PIXELFORGE_RETRAIN_RESULTS.md` | 修复后重训成果（SR ×4 +0.77 dB / 低光 +10.41 dB，含口径说明） |
| `PIXELFORGE_RETROSPECTIVE.md` | 回顾叙事：19 轮诊断脉络与「叙述层自洽性」复盘 |

## 关联证据

- 训练过程与平台凭证（实例 / 计费 / GPU 显存曲线 / 逐 epoch 日志）：[`../retrain_journey/`](../retrain_journey/)
- 逐 epoch 指标：`../../results/train_log_*.csv`
- 同口径基线评测脚本：`../../scripts/eval_baseline.py`

> 这些文档中的交叉引用（如 `ROUND2` 链 `ROUND3`）在移入本目录后依然成立（同目录相对路径）。
