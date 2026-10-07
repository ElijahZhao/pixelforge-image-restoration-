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

| 槽位 | 新模型成绩 | 对照基准 | 决策 |
|---|---|---|---|
| SR×4 | 23.34 / 0.727（ep200） | 历史 27.39 / 0.819（同口径）；bicubic 26.69 | ❌ 弃用，**恢复旧模型** |
| SR×2 | 训练中（ep45 时 27.07 / 0.8866，持续刷 best） | bicubic ×2 基线（跑 `python scripts/eval_baseline.py --task sr --scale 2` 获取） | ⏳ 跑完再定：≥bicubic → 用新；否则维持 bicubic 回退 |
| 低光 | 未开跑 | 历史低光成绩 | ⏳ |

## 🟢 收尾步骤提醒

- [ ] 三阶段全部跑完后，AutoDL 上执行同口径评估：
  ```bash
  python scripts/eval_baseline.py --task sr --scale 2   # 看 bicubic ×2 基线与新 SR×2 对比
  python scripts/eval_baseline.py --task lowlight        # 低光对比
  ```
- [ ] 按上表决策导出最终三个 `.pt`（SR×4 一定来自 `backup_old/`）。
- [ ] 打包下载 `outputs/ checkpoints/ models/ results/ serve/models/` 后再关机省钱。
- [ ] 本文件勾完、决策写入正式文档后，本清单方可标记完成。
