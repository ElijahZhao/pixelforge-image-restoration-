# PixelForge · 项目进度与待办清单

> 最后更新：2025-09-14
> 本文件记录 PixelForge 的**真实完成度**与**剩余工作**。所有状态均经过实际核查（非估计）。

---

## 一、总体进度

| 维度 | 状态 | 完成度 | 说明 |
|---|:---:|:---:|---|
| 训练侧代码（PyTorch 模型/数据/指标/脚本） | ✅ 已完成 | 100% | 结构可跑，待真实 GPU 训练 |
| 推理侧代码（FastAPI + 经典兜底 + 导出） | ✅ 已完成 | 100% | 无权重时自动走基线 |
| 前端（Next.js 交互 Demo） | ✅ 已完成 | 100% | 本地 `next build` 已通过 |
| 文档（README/DEPLOY/工具清单/LICENSE） | ✅ 已完成 | 100% | 中文美化版 README 已上线 |
| 本地测试与 demo 图（CPU 可跑） | ✅ 已完成 | 100% | train/ 单元测试 12/12 通过；真实 demo 图已生成 |
| 真实模型权重（自训 .pt） | 🚧 待办 | 0% | **需用户在 GPU 环境训练** |
| 真实评测指标（PSNR/SSIM） | ⚠️ 占位 | 0% | 当前为占位示例值 |
| 线上部署（Vercel + HF Spaces/VPS） | ⏳ 待办 | 0% | 尚未部署 |
| 申请材料（SOP / CV / 报告） | ⏳ 待办 | 0% | 可用本地技能生成 |
| 安全收尾（删除暴露的 token） | 🔴 必须 | — | **高危，需立即处理** |

**一句话总结**：代码、文档、本地测试与真实 demo 图已闭环完成并本地跑通；**唯一硬性缺口是真实训练权重（必须靠 GPU）**，其余为部署、申请材料包装与安全收尾（token 轮换）。

---

## 二、已完成（已核查）

### 2.1 训练侧 `train/`
| 文件 | 内容 | 核查 |
|---|---|---|
| `models.py` | `SRCNN`、带 PixelShuffle 的 `SRGenerator`（scale 2/4）、`LowLightUNet`；`build_model(name, scale, advanced)`；CPU 下 shape 已验证 | ✅ |
| `datasets.py` | `SuperResolutionDataset`（DIV2K/LOL 加载）；**已修复空目录/缺失目录崩溃**（原 `TypeError: 'PosixPath' object is not iterable`） | ✅ |
| `metrics.py` | PSNR / SSIM（高斯窗实现），与 `train.py` 评测逻辑一致 | ✅ |
| `train.py` | 训练脚本（Adam + 余弦退火 + AMP + 可选 VGG 感知损失）；保存 `state_dict/scale/model` 元数据 | ✅ |
| `export.py` | 经 `build_model` 重建 + `torch.jit.trace` 导出 TorchScript；round-trip 已验证 | ✅ |

### 2.2 推理侧 `serve/`
| 文件 | 内容 | 核查 |
|---|---|---|
| `app.py` | FastAPI：`/api/health`、`/api/predict`；**已修复**坏图→422、非法 task→422（原 500/200 错配）；`JSONResponse` 导入；`classical/model_loader` 改为相对导入以匹配 `uvicorn serve.app:app` 启动方式 | ✅ |
| `classical.py` | `run_classical(img, task, scale)`：Bicubic 超分 + **自适应伽马**低光基线（docstring 已从 "Retinex" 修正） | ✅ |
| `model_loader.py` | `get_sr_model(scale)` / `get_lowlight_model()`：无 `serve/models/` 时返回 `None`，服务自动走基线 | ✅ |
| `gradio_demo.py` | 一键 Gradio 演示入口 | ✅ |

### 2.3 前端 `web/`
| 模块 | 内容 | 核查 |
|---|---|---|
| `next.config.mjs` | `/api` 同源 rewrite 代理到 `NEXT_PUBLIC_API_URL`（未设置则同源） | ✅ |
| `lib/api.ts` | `predict(file, task, scale)` → 同源 `/api/predict`，返回纯 base64 `before/after` | ✅ |
| `components/CompareSlider.tsx` | 前后对比滑块（`data:image/png;base64,${after}`，无前缀重复） | ✅ |
| `app/method/page.tsx` | 英文方法页（**已修正** "Retinex" → "Adaptive gamma"） | ✅ |
| 构建 | 本地 `next build` 已生成 `.next/` | ✅ |

### 2.4 文档与交付物
| 文件 | 内容 | 核查 |
|---|---|---|
| `README.md` | 中文美化版：SVG 封面、badges、Mermaid 架构图、功能卡、指标表（⚠️占位）、SOP/CV 描述、致谢 | ✅ commit `df57135` |
| `assets/banner.svg` | 深色渐变封面（Before → After） | ✅ |
| `DEPLOY.md` | 部署指南 + §4 **受限网络经 ghproxy 镜像推送**方法 | ✅ |
| `TOOLS_CHECKLIST.md` | 内置工具/技能/连接器使用清单 | ✅ |
| `LICENSE` | MIT（ElijahZhao, 2025） | ✅ |
| `data/README.md`、`results/README.md` | 数据集下载说明、结果说明（"Retinex"→"自适应伽马" 已修正） | ✅ |
| GitHub 推送 | 31 个文件经 `ghproxy.net` 镜像推送至 `ElijahZhao/pixelforge-image-restoration-` | ✅ |
| 真实 demo 图 | `scripts/make_demo.py` + `assets/demo_*.png`：CPU 直接跑经典方法生成前后对比 | ✅ |
| 单元测试 | `train/tests/`：模型 shape、PSNR/SSIM 正确性，**12/12 通过** | ✅ |
| CORS 收紧 | `serve/app.py`：`*` 改为可通过 `ALLOWED_ORIGINS` 配置，默认本地开发仍开放 | ✅ |

---

## 三、待办清单

### 🔴 P0 — 必须立即处理（安全）
- [ ] **删除 / 轮换已暴露的 GitHub Token**（形如 `ghp_********************`，下文称「旧 Token」）
  - 当前明文存在于 `/workspace/.git/config`（remote URL）；
  - 已通过第三方镜像 `ghproxy.net` 转发，存在泄露面；
  - 处理步骤：① GitHub → Settings → Developer settings → 撤销该 token；② 重新生成新 token；③ 在本地把 remote 改为新 token（不提交到仓库）。

### 🚧 P1 — 决定项目"成色"的核心待办（需 GPU，用户侧）
- [ ] **真实训练权重（最关键缺口）**
  1. 在 Kaggle / Colab 免费 GPU 上 `pip install -r requirements.txt`；
  2. `python train/train.py --task sr --model generator --scale 4 --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual`；
  3. `python train/train.py --task lowlight --data_root data --epochs 200 --batch_size 8 --lr 2e-4`；
  4. `python train/export.py ...` 导出 TorchScript 到 `serve/models/`（文件名：`sr_generator_scale4.pt` / `lowlight.pt`）；
  5. 服务检测到权重后自动从基线切换到**自训模型**（访问 `/api/health` 可见引擎状态）。
- [ ] **下载真实数据集到 `data/`**：DIV2K（超分）、LOL（低光）；当前 `data/` 仅含下载说明。
- [ ] **替换占位指标**：在测试集（Set5 / Set14 / LOL-test）跑出**真实 PSNR/SSIM**，替换 `README.md` 指标表与 `results/README.md` 中的 `_待填真实值_`。

### 🟡 P2 — 部署与上线
- [ ] **前端部署**：Vercel 导入 `web/`，环境变量 `NEXT_PUBLIC_API_URL` 指向后端。
- [ ] **后端部署**：Hugging Face Spaces（`gradio_demo.py` 或 `app.py` 部署到小 VPS）。
- [ ] （可选）用「发布为应用」技能生成公开分享链接，方便导师直接打开 demo。

### 🟢 P3 — 申请材料包装（本地可完成）
- [ ] 用 `docx` / `pdf` 技能生成英文 **SOP / CV / 项目报告**（README 含可直接改编的中文项目描述）。
- [ ] 用 `pptx` 技能生成**项目答辩 / 面试幻灯片**。
- [ ] 用 `humanizer` 技能把 AI 味的英文表述润色为自然表达。
- [ ] 用 `xlsx` 技能生成 PSNR/SSIM 对比表、训练超参记录表。

### 🔵 P4 — 工程收尾（本地可完成）
- [x] **收紧 CORS**：`serve/app.py` 已支持 `ALLOWED_ORIGINS` 环境变量；生产环境设置域名白名单，本地默认仍开放。
- [x] **训练侧单元测试**：`train/tests/` 已覆盖模型 shape、PSNR/SSIM 正确性，12/12 通过。
- [x] **浏览器 E2E 测试**：`tests/e2e/e2e.py` 用 Playwright + Chromium 跑通「上传→Enhance→拖动滑块」全流程并截图留证；超分与低光两条链路均通过。
- [x] **仓库清理**：删除 `pixelforge-source.zip`，`.gitignore` 增加 zip 包与 E2E 截图目录排除。

---

## 四、已知风险与备注

| 项 | 说明 |
|---|---|
| 🔴 暴露的 Token | 见 P0；必须轮换，否则仓库推送权限可被他人滥用 |
| ⚠️ 占位指标 | README / results 中的 PSNR/SSIM 为格式示例，**非真实训练结果**，提交申请前务必替换 |
| ⚠️ 无真实权重 | 沙箱仅有 CPU，无法训练；当前线上 demo 跑的是**经典基线**，并非自训模型 |
| ℹ️ 经典基线定位 | Bicubic / 自适应伽马仅为"开箱即用兜底 + 对比基线"，招生委员会看重的是自训模型对比基线后的指标提升 |
| ℹ️ 受限网络推送 | 沙箱直连 GitHub 被 egress 白名单拦截，已用 `ghproxy.net` 镜像解决（见 `DEPLOY.md` §4） |

---

## 五、进度速览（勾选图例）
- ✅ 已完成   🚧 进行中/需 GPU   ⏳ 未开始   ⚠️ 占位待替换   🔴 高危待处理

| 阶段 | 模块 | 状态 |
|---|---|:---:|
| 训练 | 模型 / 数据 / 指标 / 训练 / 导出 | ✅ |
| 推理 | FastAPI / 经典兜底 / 权重加载 | ✅ |
| 前端 | Next.js Demo / 滑块 / 代理 | ✅ |
| 文档 | README / DEPLOY / 工具清单 / LICENSE | ✅ |
| 测试 / Demo | 单元测试 12/12 + 真实 demo 图 | ✅ |
| 训练 | 真实权重（GPU） | 🚧 |
| 评测 | 真实 PSNR/SSIM 数值 | ⚠️ |
| 部署 | Vercel / HF Spaces | ⏳ |
| 材料 | SOP / CV / 报告 / 幻灯片 | ⏳ |
| 安全 | 删除暴露 Token | 🔴 |
