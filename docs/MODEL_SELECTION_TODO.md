# ⚠️ 重训收尾待办清单（按槽位择优）

> 2026-10-07 AutoDL 新实例（RTX 4090 24GB / torch 2.3.0+cu121）三阶段重训。
> 策略：**按槽位择优**——每个模型槽位各自挑历史/新版中更好的那个，互不绑定。
> **本文件是收尾整合时的硬性检查单，逐项打勾前不得关闭。**

---

## 🔴 最高优先级：恢复旧 SR×4 模型（千万别忘！）

- [ ] **最终整合时，SR×4 槽位必须用旧模型，禁止使用本次新训的 SR×4 权重。**

| 项 | 说明 |
|---|---|
| 为什么 | 本次 SR×4（BATCH=16）200 epoch 跑完 Best val PSNR 仅 **23.34 / SSIM 0.727**，远低于历史同口径 **27.39 / 0.819**，甚至低于 bicubic 基线 26.69 —— 欠训练，不能要 |
| 旧模型在哪 | AutoDL 实例 `/root/autodl-tmp/backup_old/`（开训前脚本自动备份，含 `results/ models/ serve/models/` 三份） |
| 新的坏权重在哪 | `/root/autodl-tmp/pixelforge/models/sr_generator_scale4_best.pth`（**不要下载、不要导出、不要覆盖旧模型**） |
| 恢复动作 | 从 `backup_old/serve/models/sr_generator_scale4.pt` 恢复部署权重；本地仓库同样保留旧 `sr_generator_scale4.pt`，不被新产物覆盖 |
| 佐证截图 | `docs/autodl_screenshots/18_SRx4完成但仅23.34_发现batch16问题_SRx2接续中.png` |

## 🟡 文档说明：仓库更新时必须写明

- [ ] 在 `PROGRESS.md` / `REMEDIATION_PLAN.md`（或训练总结文档）中写明本次结论：
  1. **SR×4**：本次重训因 `BATCH=16`（每 epoch 优化步数 50，历史为 batch 8 的 100 步）欠训练，成绩 23.34 不可用；**沿用历史旧模型 27.47**。
  2. **SR×2**：历史上从未训过（一直回退 bicubic），本次为首个 ×2 权重；是否采纳视最终成绩与 bicubic ×2 基线对比而定。
  3. 训训脚本 `retrain_autodl.sh` 默认 `BATCH=16` 与历史最佳配置（batch 8）不一致的问题，要么改默认值、要么在脚本注释/手册中标注"复现历史成绩须 `BATCH=8`"。
- [ ] 若 SR×2 最终达标，补充其训练日志 CSV 与导出说明；不达标则注明"×2 维持 bicubic 回退"。

## 🟢 各槽位决策记录（跑完后填写）

> **硬约束（2026-10-07 用户拍板）：不再重训任何模型。** 本次是最后一次训练，
> 所有槽位只能在"已有产物"（新训 / `backup_old/` 历史 / bicubic 回退）里择优，
> 三次训练全程留档形成对照系列（见下方"三次训练对照"）。

| 槽位 | 新模型成绩 | 对照基准 | 决策 |
|---|---|---|---|
| SR×4 | 23.34 / 0.727（ep200） | 历史 27.39 / 0.819（同口径）；bicubic 26.69 | ❌ 弃用，**恢复旧模型** |
| SR×2 | **27.72 / 0.8994（ep195，已训完）** | bicubic ×2 基线（跑 `python scripts/eval_baseline.py --task sr --scale 2` 获取） | ⏳ ≥bicubic → 用新；否则维持 bicubic 回退（不重训） |
| 低光 | 训练中（ep12 时 15.95 / 0.7159；开局优于历史：ep1 14.35 vs 历史 11.87） | 历史最佳 18.59 / 0.7963（ep100） | ⏳ 跑完对比历史，取更优 |

## 🟢 三次训练对照系列（证明材料，写文档时用）

| 轮次 | 环境 | 内容 | 结果 | 留档 |
|---|---|---|---|---|
| 第 1 次 | 历史本地训练 | SR×4 batch8 + 低光 batch8 | SR×4 27.39~27.47 / 低光 18.59 | 仓库 `results/train_log_*.csv`（历史 CSV） |
| 第 2 次 | AutoDL 4090（本批） | 三阶段 batch16 | SR×4 23.34 ❌ 欠训练；SR×2 27.72（首训新槽位）；低光 进行中 | `docs/autodl_screenshots/16~21` |
| 第 3 次 | 同上 | （若低光单独计）低光 batch16 | 待补 | 待补 |

> 文档结论角度：同一代码、同一数据、三种配置（历史 batch8 / 本批 batch16 / 按槽位择优），
> 恰好构成**超参敏感性对照实验**——batch16 导致 SR×4 欠训练（23.34 vs 27.39），
> 反证历史配置的合理性；SR×2 首训 27.72 补齐了从未有过的 ×2 槽位。这是加分项，不是黑历史。

## 🟢 收尾步骤提醒

- [ ] 三阶段全部跑完后，AutoDL 上执行同口径评估：
  ```bash
  python scripts/eval_baseline.py --task sr --scale 2   # 看 bicubic ×2 基线与新 SR×2 对比
  python scripts/eval_baseline.py --task lowlight        # 低光对比
  ```
- [ ] 按上表决策导出最终三个 `.pt`（SR×4 一定来自 `backup_old/`）。
- [ ] 打包下载 `outputs/ checkpoints/ models/ results/ serve/models/` 后再关机省钱。
- [ ] 本文件勾完、决策写入正式文档后，本清单方可标记完成。
