# 本次提交清单（feat: 修复 + 重训闭环）

> 目的：把"诊断 → 修复 → 重训 → 验证 → 证据归档"整条链路一次性提交进仓库。
> **提交前已完成的安全检查**：
> - 全仓库扫描无明文 token（`DIAGNOSIS_ROUND7.md` 里的 token 已脱敏为 `***REDACTED***`）；
> - `.git/config` 无明文凭证；
> - `.gitignore` 增加白名单，放行 4 个核心证据文件（见下）。

---

## 一、提交分组与命令

建议**分 4 个语义化提交**，而非一次性全提——这样 git 历史本身就能讲清"做了什么"。

### Commit 1 · 代码修复（F2/F3/F5/F7/F8/F9/F13 + 设备对齐）

```bash
git add train/train.py train/datasets.py serve/app.py serve/model_loader.py \
        scripts/make_demo.py deploy/streamlit/streamlit_app.py deploy/hf_space/app.py
git commit -m "fix(train,serve): 感知损失归一化/权重显式化/训练-评估对齐/消除假放大/输入限制

- F2: VGGPerceptualLoss 提取特征前做 ImageNet 归一化；_feat 增加设备对齐
- F3: loss 改为 w_pixel*charbonnier + w_percep*vgg_perceptual（可 CLI 配置）
- F5: 训练目标与 best 判据写入同一 CSV 日志头
- F7: predict_sr 返回模型真实分辨率 (lr*scale)，移除二次 PIL 假放大
- F8: ×2 无权重时显式回退并标注 note/徽章
- F9: LowLightDataset 按文件名配对 + 尺寸校验 + 损坏跳过 + 验证确定性
- F13: 推理服务增加 10MB/4MP 输入限制，超限返回 413"
```

### Commit 2 · 正确性测试（F14）

```bash
git add train/tests/test_correctness.py train/tests/run_tests.py
git commit -m "test(train): 新增 8 项正确性测试，形状测试升级为正确性测试

- 覆盖 VGG 归一化敏感性、损失权重合理性、predict_sr 尺寸契约、数据配对
- 变异测试验证：注入 4 类真实缺陷 → 新套件 4/4 拦截（旧形状测试 0 拦截）
- 全套 20/20 通过"
```

### Commit 3 · 重新训练的真实权重

```bash
git add serve/models/*.pt deploy/streamlit/models/*.pt deploy/hf_space/models/*.pt \
        assets/demo_*.jpg assets/sample_*.png
git commit -m "feat(models): 用修复后代码重训的 SR×4 与低光权重（含 demo 图重生成）

- SR×4: val PSNR 17.20 -> 27.47（相对 bicubic +0.77 dB），SSIM 0.217 -> 0.780
- 低光: 相对不处理基线 +10.41 dB / SSIM +0.55
- demo 图由 scripts/make_demo.py 依据新权重重生成
- 详见 PIXELFORGE_RETRAIN_RESULTS.md"
```

### Commit 4 · 诊断/成果文档 + 证据链归档 + 可复现指标

```bash
git add DIAGNOSIS_*.md PIXELFORGE_*.md docs/ retrain_autodl.sh scripts/eval_baseline.py \
        results/train_log_*.csv results/README.md README.md PROGRESS.md DEPLOY.md .gitignore
git commit -m "docs: 19 轮诊断/修复方案/重训成果报告 + 重训证据链归档

- DIAGNOSIS_ROUND1-19: 逐轮深度审计
- PIXELFORGE_FIX_PLAN/RETROSPECTIVE/EXECUTION_REPORT/RETRAIN_RESULTS
- docs/retrain_journey/: 11 张过程+平台凭证截图 + 2 份训练日志
  （含 GPU 显存阶梯曲线等第三方凭证）
- results/train_log_*.csv: 200 epoch 逐轮指标，使报告数字可复现（F6）
- scripts/eval_baseline.py: 与 bicubic/不处理基线的同口径评测脚本
- .gitignore: 白名单放行上述证据文件"
```

---

## 二、逐文件分类

### A. 代码修复（7 个）
| 文件 | 修复项 |
|---|---|
| `train/train.py` | F2/F3/F5 + 设备对齐 |
| `train/datasets.py` | F9 |
| `serve/app.py` | F8/F13 |
| `serve/model_loader.py` | F7 |
| `scripts/make_demo.py` | F12（.jpg 一致性） |
| `deploy/streamlit/streamlit_app.py` | F7/F8 |
| `deploy/hf_space/app.py` | F7/F8 |

### B. 测试（2 个）
- `train/tests/test_correctness.py`（新）
- `train/tests/run_tests.py`（注册新模块）

### C. 模型权重（6 个）
- `serve/models/{sr_generator_scale4.pt, lowlight.pt}`
- `deploy/streamlit/models/{...}`
- `deploy/hf_space/models/{...}`

### D. Demo 图（9 个）
- `assets/demo_*.jpg`（6）、`assets/sample_*.png`（2）

### E. 文档与证据（新增）
- 诊断：`DIAGNOSIS_AND_PLAN.md`、`DIAGNOSIS_ROUND2-19.md`
- 方案/成果：`PIXELFORGE_FIX_PLAN.md`、`PIXELFORGE_RETROSPECTIVE.md`、
  `PIXELFORGE_EXECUTION_REPORT.md`、`PIXELFORGE_RETRAIN_RESULTS.md`
- 证据链：`docs/retrain_journey/`（11 图 + 2 日志 + README）
- 工具：`retrain_autodl.sh`、`scripts/eval_baseline.py`
- 指标：`results/train_log_sr_generator.csv`、`results/train_log_lowlight_srcnn.csv`

### F. 叙述对齐（修改）
- `README.md`、`PROGRESS.md`、`DEPLOY.md`、`results/README.md`、`.gitignore`

---

## 三、提交前自检清单

- [x] 无明文 token/密钥（已扫描 + 脱敏）
- [x] `.git/config` 无凭证
- [x] 证据文件已放行（`.gitignore` 白名单）
- [x] 权重 `.pt` 与 `.gitignore` 中"plan B 特意跟踪"一致
- [x] 测试 20/20（本地 CPU）；GPU 上亦 20/20（AutoDL 实测）
- [ ] 提交后推送（需你确认 remote 与权限）

## 四、注意

- **推送前**：确认 GitHub 上的**旧 token 已撤销**（F1 的收尾动作，仅你能操作）。
- 本项目 `.gitignore` 明确跟踪 `serve/models/*.pt`（6MB 级），提交后仓库会涨约 20MB（3 处副本）；
  这是**有意为之**（`git clone` 即可运行），如需瘦身可改用 Git LFS，但会牺牲开箱即用性。
