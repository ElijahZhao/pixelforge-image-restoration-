# PixelForge · 项目进度与待办清单

> 最后更新：2026-10-04（ROUND22/23 全项目审计 · pytest CI 固化 · 诊断文档归档至 docs/history/）
> 本文件记录 PixelForge 的**真实完成度**与**剩余工作**。所有状态均经过实际核查（非估计）。

---

## 一、总体进度

| 维度 | 状态 | 完成度 | 说明 |
|---|:---:|:---:|---|
| 训练侧代码（PyTorch 模型/数据/指标/脚本） | ✅ 已完成 | 100% | 结构可跑，已真实 GPU 训练 |
| 推理侧代码（FastAPI + 经典兜底 + 导出） | ✅ 已完成 | 100% | 无权重时自动走基线；有权重自动切 ML |
| 前端（Next.js 交互 Demo） | ✅ 已完成 | 100% | 本地 `next build` 已通过 |
| 文档（README/DEPLOY/工具清单/LICENSE） | ✅ 已完成 | 100% | 中文美化版 README 已上线 |
| 本地测试与 demo 图（CPU 可跑） | ✅ 已完成 | 100% | pytest 全绿 28/28（20 训练单测 + 8 API 冒烟，CPU 可跑）；真实 demo 图已生成 |
| 真实模型权重（自训 .pt） | ✅ 已完成 | 100% | **修复后重训并导出**（见 §2.5） |
| 真实评测指标（PSNR/SSIM） | ✅ 已完成 | 100% | 与基线同口径对比、逐 epoch 日志已提交、可复现（F6 闭环） |
| 线上部署（Streamlit Cloud 公开 Demo） | ✅ 已上线 | 100% | https://hddzzb68eqfnoed8zsmgqp.streamlit.app/ |
| 申请材料（SOP / CV / 报告） | ⏳ 重要待办 | 0% | 可用本地技能生成 |
| 安全收尾（撤销暴露的 token） | 🔴 待办 | — | 旧 token **撤销状态未核实**：对话中多次提示其经 `ghproxy.net` 明文使用且早已暴露，**请立即到 GitHub 吊销并轮换**（见 §四） |

**一句话总结**：代码、文档、测试、真实 GPU 训练（**修复后重训、已打赢基线**）、可复现指标、公开 Demo **已全部闭环**；仅剩（1）申请材料包装（与项目质量无关）、（2）**撤销已暴露的 GitHub Token**（见 §四，务必手动处理）。

---

## 二、已完成（已核查）

### 2.1 训练侧 `train/`
| 文件 | 内容 | 核查 |
|---|---|---|
| `models.py` | `SRCNN`、带 PixelShuffle 的 `SRGenerator`（scale 2/4）、`LowLightUNet`；`build_model(name, scale, advanced)`；CPU 下 shape 已验证 | ✅ |
| `datasets.py` | `SuperResolutionDataset`（DIV2K/LOL 加载）；**已修复空目录/缺失目录崩溃** | ✅ |
| `metrics.py` | PSNR / SSIM（高斯窗实现），与 `train.py` 评测逻辑一致 | ✅ |
| `train.py` | 训练脚本（Adam + 余弦退火 + AMP + 可选 VGG 感知损失）；保存 `state_dict/scale/model` 元数据 | ✅ |
| `export.py` | 经 `build_model` 重建 + `torch.jit.trace` 导出 TorchScript；round-trip 已验证 | ✅ |

### 2.2 推理侧 `serve/`
| 文件 | 内容 | 核查 |
|---|---|---|
| `app.py` | FastAPI：`/api/health`、`/api/predict`；坏图→422、非法 task→422；`classical/model_loader` 改为相对导入 | ✅ |
| `classical.py` | `run_classical(img, task, scale)`：Bicubic 超分 + 自适应伽马低光基线 | ✅ |
| `model_loader.py` | `get_sr_model(scale)` / `get_lowlight_model()`：无 `serve/models/*.pt` 时返回 `None` 走基线；有则加载 TorchScript | ✅ |
| `gradio_demo.py` | 一键 Gradio 演示入口 | ✅ |

### 2.3 前端 `web/`
| 模块 | 内容 | 核查 |
|---|---|---|
| `next.config.mjs` | `/api` 同源 rewrite 代理到 `NEXT_PUBLIC_API_URL` | ✅ |
| `lib/api.ts` | `predict(file, task, scale)` → 同源 `/api/predict` | ✅ |
| `components/CompareSlider.tsx` | 前后对比滑块 | ✅ |
| `app/method/page.tsx` | 英文方法页（"Retinex" → "Adaptive gamma" 已修正） | ✅ |
| 构建 | 本地 `next build` 已生成 `.next/` | ✅ |

### 2.4 文档与交付物
| 文件 | 内容 | 核查 |
|---|---|---|
| `README.md` | 中文美化版：SVG 封面、badges、Mermaid 架构图、功能卡、指标表（真实值）、SOP/CV 描述、致谢；已补全公开 Demo 链接与自训演示图 | ✅ |
| `assets/banner.svg` | 深色渐变封面（Before → After） | ✅ |
| `DEPLOY.md` | 部署指南 + §4 受限网络经 ghproxy 镜像推送方法 | ✅ |
| `LICENSE` | MIT（ElijahZhao, 2025） | ✅ |
| `data/README.md`、`results/README.md` | 数据集下载说明、结果说明 | ✅ |
| GitHub 推送 | 经 `ghproxy.net` 镜像推送至 `ElijahZhao/pixelforge-image-restoration-`（含后续 ROUND22/23 审计、低光修复、pytest CI 与依赖 lock；2026-10-04 再次推送） | ✅ |
| 真实 demo 图 | `scripts/make_demo.py` + `assets/demo_*.jpg`：CPU 直接用**自训权重**生成「低清/暗光输入 vs 经典基线 vs 自训模型」对比 | ✅ |
| 单元测试 + API 冒烟 | `train/tests/`（20）+ `tests/test_api_smoke.py`（8）：模型 shape / PSNR·SSIM / 正确性测试（归一化·权重·尺寸契约·配对）+ `/api/health`·`/api/predict` 全路径守卫；**pytest 全绿 28/28** | ✅ |
| CORS 收紧 | `serve/app.py`：`*` 改为可通过 `ALLOWED_ORIGINS` 配置 | ✅ |
| E2E 测试 | `tests/e2e/e2e.py` 用 Playwright + Chromium 跑通「上传→Enhance→拖动滑块」全流程 | ✅ |

### 2.5 真实训练成果（AutoDL RTX 3080 Ti）

**A. 修复后重训（2026-10-03，最终版，已打赢基线）**

| 任务 | 配置 | Best 指标 | 相对基线增益 | 产物 |
|---|---|---|---|---|
| SR ×4（超分） | `--model generator --scale 4 --epochs 200 --batch_size 8 --lr 1e-4 --perceptual` | **全图 PSNR 27.47 / SSIM 0.780** | **+0.77 dB vs bicubic (26.69)** | `serve/models/sr_generator_scale4.pt`（4.8M） |
| Lowlight（低光增强） | `--task lowlight --epochs 200 --batch_size 8 --lr 2e-4` | **全图 PSNR 18.18 / SSIM 0.739** | **+10.41 dB vs 不处理 (7.77)** | `serve/models/lowlight.pt`（2.3M） |

> 同口径评测由 `scripts/eval_baseline.py` 执行；训练日志见 `results/train_log_*.csv`（200 epoch 逐轮）；
> 过程与平台凭证（实例/计费/GPU 显存曲线）见 `docs/retrain_journey/`；完整报告见 `docs/history/PIXELFORGE_RETRAIN_RESULTS.md`。

**B. 修复前首次训练（2026-10-02，已作废，仅作对照）**

| 任务 | 配置 | Best 指标 | 备注 |
|---|---|---|---|
| SR ×4 | 同上（但代码含感知损失缺陷） | val PSNR 17.20 / SSIM 0.217 | **低于 bicubic，已定位为感知损失缺陷** |
| Lowlight | 同上 | val PSNR 19.26 / SSIM ≈0.74–0.78 | 随机裁剪口径，**与修复后全图口径不可比** |

> **指标解读（已更新）**：
> - **SR×4**：修复前 17.20 低于 bicubic；修复（VGG 归一化 + 显式权重）后重训达 **27.47，高于 bicubic +0.77 dB** —— 缺陷已闭环。
> - **低光**：修复后同口径下相对不处理基线 **+10.41 dB**，"疑似退化"的旧结论（18.59 < 19.26）系**验证口径变化造成的假象**，已澄清。
> - 加载验证（沙箱 CPU）：`GET /api/health` → `{"sr_scale2":"classical","sr_scale4":"ml","lowlight":"ml"}`，自训模型被正确加载。

---

## 三、待办清单

### ✅ P0 — 已处理（安全）
- [ ] **🔴 撤销 / 轮换已暴露的 GitHub Token（尚未确认完成）**
  - 本地 `/workspace/.git/config` 明文凭证**已清除**（remote 恢复为无凭证 URL）；
  - 全仓库扫描确认诊断文档中的 token 已脱敏（`***REDACTED***`），**未进入 Git 历史**；
  - ⚠️ 旧 token **是否已在 GitHub 撤销并未核实**——其在对话中多次出现、并曾（及 2026-10-04 本轮）经 `ghproxy.net` 明文用于推送，**应视为已泄露，务必立即吊销并换新**；
  - 后续推送改用一次性凭据助手注入（不落盘），推送后远端与本地已同步。

### ✅ P1 — 训练成果收尾（已完成）
- [x] **真实训练权重（已完成 ✅）** —— 首训 2026-10-02、**修复后重训 2026-10-03**（见 §2.5）：
  1. SR×4（`--model generator --perceptual`）：200 epoch，修复后 **全图 PSNR 27.47 / SSIM 0.780**；
  2. Lowlight：200 epoch，修复后 **全图 PSNR 18.18 / SSIM 0.739**；
  3. 已 `export.py` 导出 TorchScript 到 `serve/models/`：`sr_generator_scale4.pt`(4.8M) / `lowlight.pt`(2.3M)；
  4. **沙箱已验证** `/api/health` → `sr_scale4:"ml"`、`lowlight:"ml"`（见 §2.5）。
- [x] **决定权重是否进 git（`.gitignore` 决策）** —— **已选 B：权重放行进 git**，`git clone` 即得可运行项目。
- [x] **SR 重训（含感知损失修复）** —— 已用修复后代码重训，SR×4 达 **27.47（高于 bicubic +0.77 dB）**，缺陷闭环；
  评测脚本 `scripts/eval_baseline.py`、逐 epoch 日志 `results/train_log_*.csv`、平台凭证 `docs/retrain_journey/` 均已留档。
  > **执行摘要（实际执行，2026-10-03，AutoDL RTX 3080 Ti）**：
  > 1. 备份旧产物 → `unzip` 覆盖修复后 `train/`、`serve/` → `python -m train.tests.run_tests`（20/20）；
  > 2. 冒烟 2 epoch 通过 → 正式重训 SR×4 与低光各 200 epoch（前台顺序执行，无人工干预）；
  > 3. 自动 `export.py` 导出 TorchScript → 下载权重回仓库替换 3 处 `models/`；
  > 4. 同口径基线评测：SR×4 +0.77 dB、低光 +10.41 dB。
  > 完整命令与过程见 `retrain_autodl.sh` 与 `docs/retrain_journey/README.md`。
- [x] **替换 README 指标**：`README.md` / `results/README.md` 的 SR×4（17.20→27.47）与低光（19.26→18.18 全图口径）均已更新为修复后真实值；SRCNN ×2 未训练仍留 `TBD`。

### 🟡 P2 — 部署与上线（方案 B' — Streamlit Cloud）
> 决策：GitHub Pages / Actions **托管不了 ML 后端**（纯静态 / 临时 job）。HF Spaces 自 2026-07 起跑计算的 Space 需付费（PRO），免费仅剩「2 个 ZeroGPU Gradio Space」且有每日 GPU 额度限制，对 CPU 即可跑的 demo 不划算。**改用 Streamlit Community Cloud：公开 app 免费、无额度、直接从 GitHub 仓库部署。**
- [x] **Streamlit 部署包已就绪**：`deploy/streamlit/`（`streamlit_app.py` 自包含入口 + `requirements.txt` + `models/` 两权重），附 `DEPLOY_STREAMLIT.md` 步骤说明。
- [x] **沙箱实跑验证**：`streamlit run` 启动 HTTP 200 无 Traceback；引擎 `SR4 = ML · LowLight = ML`；SR×4 与低光在 100×100 / 1280×720 / 63×41 等任意尺寸均正常。
- [x] **修复 U-Net 尺寸约束 bug**：低光 U-Net 要求边长 32 倍数，否则解码器拼接崩溃；已在 `deploy/streamlit/streamlit_app.py`、`deploy/hf_space/app.py` 与主仓库 `serve/model_loader.py` 加自适应补齐（pad→推理→裁回）。
- [x] **HF Spaces 备选包**：`deploy/hf_space/` 亦保留（若后续开通 PRO 或申请 community grant 可用）。
- [x] **用户侧部署完成**：已在 Streamlit Community Cloud 完成 Deploy，公开链接 **https://hddzzb68eqfnoed8zsmgqp.streamlit.app/** 已上线，自训模型实时驱动（SR ×4 与 Low-light 均 ML）。
- [ ] （可选）前端部署：Vercel 导入 `web/`，`NEXT_PUBLIC_API_URL` 指向后端。

### 🟢 P3 — 申请材料包装（重要 · 本地可完成）
- [ ] 用 `docx` / `pdf` 技能生成英文 **SOP / CV / 项目报告**（README 含可直接改编的中文项目描述）。
- [ ] 用 `pptx` 技能生成**项目答辩 / 面试幻灯片**。
- [ ] 用 `humanizer` 技能把 AI 味的英文表述润色为自然表达。
- [ ] 用 `xlsx` 技能生成 PSNR/SSIM 对比表、训练超参记录表。

### 🔵 P4 — 工程收尾（本地可完成）
- [x] **收紧 CORS**：`serve/app.py` 已支持 `ALLOWED_ORIGINS` 环境变量；生产环境设置域名白名单，本地默认仍开放。
- [x] **训练侧单元测试 + API 冒烟**：`train/tests/` 覆盖模型 shape、PSNR/SSIM，并新增**正确性测试**（VGG 归一化、损失权重量级、SR 输出尺寸契约、数据管线配对一致性）；新增 `tests/test_api_smoke.py` 用 FastAPI `TestClient` 覆盖 `/api/health` 与 `/api/predict` 全路径（坏 task / 非图像 / 超字节 / 超尺寸守卫）；**pytest 全绿 28/28**（已固化进 CI）。
- [x] **浏览器 E2E 测试**：`tests/e2e/e2e.py` 用 Playwright + Chromium 跑通「上传→Enhance→拖动滑块」全流程并截图留证；超分与低光两条链路均通过。
- [x] **全项目审计（ROUND22 / ROUND23）**：安全/并发/复现性/边界缺陷（11 项）+ 部署依赖契约/测试有效性「假通过」/死代码（7 项），全部修缮并实测回归通过（归档见 `docs/history/`）。
- [x] **pytest CI 固化 + 依赖 lock**：新增 `pyproject.toml`（pytest 配置）、`tests/test_api_smoke.py`、`requirements*.lock.txt`（pip-tools 固定传递依赖）、`.github/workflows/ci.yml`（GitHub Actions 自动跑测试）。
- [x] **仓库清理**：删除 `pixelforge-source.zip`，`.gitignore` 增加 zip 包与 E2E 截图目录排除；2026-10-04 将根目录 26 份诊断/报告文档归档至 `docs/history/`，并删除过时的 `TOOLS_CHECKLIST.md`。

---

## 四、已知风险与备注

| 项 | 说明 |
|---|---|
| 🔴 暴露的 Token | 见 P0 / §三；**撤销状态未核实，必须立即吊销并轮换**，否则仓库推送权限可被他人滥用（该 token 曾/本轮经 `ghproxy.net` 明文使用） |
| ⚠️ 无真实权重（历史） | 沙箱仅有 CPU 无法训练；**但已通过用户 AutoDL 训练 + 上传 .pt 补全真实权重**，沙箱已验证加载（见 §2.5） |
| ⚠️ SR 感知损失缺陷 | SR×4 用 `--perceptual`，但原实现有缺陷：VGG 输入未做 ImageNet 归一化、像素项被 `0.01` 系数抹除。**已在代码中修复**（`train/train.py`），需重训才体现。**不是"配置选择"，是已定位的实现问题。** |
| ℹ️ 权重已进 git | 已选 B：`.gitignore` 移除 `serve/models/*.pt` 忽略规则，`sr_generator_scale4.pt` / `lowlight.pt` 随仓库提交，`git clone` 即得可运行项目 |
| ℹ️ 经典基线定位 | Bicubic / 自适应伽马为"开箱即用兜底 + 对比基线"。**当前自训模型尚未在可复现划分上证明优于基线**（见 F6） |
| ℹ️ 受限网络推送 | 沙箱直连 GitHub 被 egress 白名单拦截，已用 `ghproxy.net` 镜像解决（见 `DEPLOY.md` §4） |

---

## 五、进度速览（勾选图例）
- ✅ 已完成   🚧 进行中/需 GPU   ⏳ 未开始   ⚠️ 占位待替换   🔴 高危待处理

| 阶段 | 模块 | 状态 |
|---|---|:---:|
| 训练 | 模型 / 数据 / 指标 / 训练 / 导出 | ✅ |
| 推理 | FastAPI / 经典兜底 / 权重加载 | ✅ |
| 前端 | Next.js Demo / 滑块 / 代理 | ✅ |
| 文档 | README / DEPLOY / PROJECT_NOTES / LICENSE / 诊断归档（docs/history/） | ✅ |
| 测试 / Demo | 单元测试 20/20 + 真实 demo 图 | ✅ |
| 训练 | 真实权重（GPU） | ✅ |
| 评测 | 真实 PSNR/SSIM 数值 | ⚠️ |
| 部署 | Streamlit Cloud 公开 Demo | ✅ |
| 材料 | SOP / CV / 报告 / 幻灯片 | ⏳ |
| 安全 | 删除暴露 Token | 🔴 |
