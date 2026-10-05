# PixelForge · 项目进度

> 最后更新：2026-10-05
> 本文件记录 PixelForge 的完成度与剩余工作。

---

## 一、总体进度

| 维度 | 状态 | 说明 |
|---|---|---|
| 训练侧代码（PyTorch 模型 / 数据 / 指标 / 脚本） | 已完成 | 结构可跑，已在 GPU 上真实训练 |
| 推理侧代码（FastAPI + 经典兜底 + 导出） | 已完成 | 无权重时自动走基线，有权重自动切 ML |
| 本地测试与 demo 图（CPU 可跑） | 已完成 | pytest 29 passed / 1 skipped；覆盖率 79.7%（门槛 75%）；前端 vitest 3 项 |
| 前端（Next.js 交互 Demo） | 已完成 | `next build` + `tsc` + vitest 已进 CI |
| 依赖与接口安全 | 已完成 | CI 跑 pip-audit；`/api/predict` 带按 IP 限流 |
| 文档（README / API / 运维 / LICENSE） | 已完成 | 中文主文档 + `README.en.md` + `docs/API.md` + `docs/OPERATIONS.md` |
| 容器化（后端） | 已完成 | `Dockerfile`：Python 3.11-slim、非 root、带健康检查、内置权重 |
| 真实模型权重（自训 `.pt`） | 已完成 | 修复后重训并导出（见 §2.4） |
| 真实评测指标（PSNR / SSIM） | 已完成 | 与基线同口径对比，逐 epoch 日志已提交，可复现 |
| 线上部署（Streamlit Cloud 公开 Demo） | 已上线 | https://pixelforge-image-restoration.streamlit.app/ （2026-10-05 更换域名，旧 app 已停用） |
| 延伸材料（报告 / 幻灯片） | 未开始 | 与项目本身质量无关 |
| 安全收尾（吊销开发期 PAT） | 待办 | 见 §四 |

代码、文档、测试、真实 GPU 训练、可复现指标与公开 Demo 均已完成；仅剩延伸材料与 PAT 吊销两项。

---

## 二、已完成

### 2.1 训练侧 `train/`

| 文件 | 内容 |
|---|---|
| `models.py` | `SRCNN`、带 PixelShuffle 的 `SRGenerator`（scale 2/4）、`LowLightUNet`；`build_model(name, scale, advanced)` |
| `datasets.py` | `SuperResolutionDataset`、`LowLightDataset`（DIV2K / LOL 加载） |
| `metrics.py` | PSNR / SSIM（高斯窗实现），与 `train.py` 评测逻辑一致 |
| `train.py` | 训练脚本（Adam + 余弦退火 + AMP + 可选 VGG 感知损失），保存 `state_dict/scale/model` 元数据 |
| `export.py` | 经 `build_model` 重建 + `torch.jit.trace` 导出 TorchScript |

### 2.2 推理侧 `serve/`

| 文件 | 内容 |
|---|---|
| `app.py` | FastAPI：`/api/health`、`/api/predict`；坏图 / 非法 task 返回 422 |
| `classical.py` | `run_classical(img, task, scale)`：Bicubic 超分 + 自适应伽马低光基线 |
| `model_loader.py` | `get_sr_model(scale)` / `get_lowlight_model()`：无权重时返回 `None` 走基线 |
| `gradio_demo.py` | 一键 Gradio 演示入口 |

### 2.3 前端与部署

| 模块 | 内容 |
|---|---|
| `web/` | Next.js 14 + Tailwind + TypeScript；`/api` 同源 rewrite，前后对比滑块，英文方法页 |
| `deploy/streamlit/` | 自包含部署包（已上线 Streamlit Community Cloud） |
| `deploy/hf_space/` | Hugging Face Spaces 备选包 |

### 2.4 真实训练成果（AutoDL RTX 3080 Ti）

**A. 修复后重训（2026-10-03）**

| 任务 | 配置 | 指标 | 相对基线 | 产物 |
|---|---|---|---|---|
| SR ×4 | `--model generator --scale 4 --epochs 200 --batch_size 8 --lr 1e-4 --perceptual` | 全图 PSNR 27.47 / SSIM 0.780 | +0.77 dB vs bicubic (26.69) | `serve/models/sr_generator_scale4.pt`（4.8M） |
| 低光 | `--task lowlight --epochs 200 --batch_size 8 --lr 2e-4` | 全图 PSNR 18.18 / SSIM 0.739 | +10.41 dB vs 不处理 (7.77) | `serve/models/lowlight.pt`（2.3M） |

> 同口径评测由 `scripts/eval_baseline.py` 执行；训练日志见 `results/train_log_*.csv`（200 epoch 逐轮）；
> 过程与平台凭证（实例 / 计费 / GPU 显存曲线）见 `docs/retrain_journey/`；完整报告见 `docs/history/PIXELFORGE_RETRAIN_RESULTS.md`。

**B. 修复前首次训练（2026-10-02，仅供对照）**

| 任务 | 指标 | 备注 |
|---|---|---|
| SR ×4 | val PSNR 17.20 / SSIM 0.217 | 低于 bicubic，成因是感知损失实现缺陷 |
| 低光 | val PSNR 19.26 / SSIM ≈0.74–0.78 | 随机裁剪口径，与修复后全图口径不可比 |

> SR×4 修复前为 17.20；补上 VGG 归一化与显式权重后重训达 27.47，高于 bicubic +0.77 dB。
> 低光修复后同口径下相对不处理基线 +10.41 dB；此前"疑似退化"的结论（18.59 < 19.26）来自验证口径变化，已澄清。
> 加载验证（CPU）：`GET /api/health` → `{"sr_scale2":"classical","sr_scale4":"ml","lowlight":"ml"}`。

### 2.5 工程收尾

- 收紧 CORS：`serve/app.py` 支持 `ALLOWED_ORIGINS` 环境变量。
- 接口限流：`/api/predict` 按 IP 令牌桶限流，超限 429（默认 30 突发 / 0.5 补充每秒）。
- 测试：`train/tests/`（20）+ `tests/test_api_smoke.py`（9），覆盖模型 shape、PSNR/SSIM、正确性测试（VGG 归一化、损失权重量级、SR 输出尺寸契约、数据管线配对一致性）与 API 全路径守卫；pytest 29 passed。
- 覆盖率：`pytest-cov` 门槛 75%，当前 79.7%（配置见 `pyproject.toml`）。
- 前端测试：`web/lib/api.test.ts`（vitest）覆盖请求构造与错误映射。
- 浏览器 E2E：`tests/e2e/test_e2e.py`（Playwright + Chromium）跑通「上传 → Enhance → 拖动滑块」全流程；不在 CI 内运行。
- 依赖漏洞扫描：CI `audit` job 跑 `pip-audit -r requirements.lock.txt`。
- CI 与依赖锁定：`pyproject.toml`、`requirements*.lock.txt`（pip-tools）、`.github/workflows/ci.yml`（三个 job：pytest+覆盖率 / pip-audit / 前端 tsc+vitest+build）。
- 仓库整理：删除 `pixelforge-source.zip`；26 份诊断/报告文档归档至 `docs/history/`；删除过时的 `TOOLS_CHECKLIST.md`。
- 修复 U-Net 尺寸约束：低光 U-Net 要求边长 32 倍数，已在 `serve/model_loader.py`、`deploy/streamlit/streamlit_app.py`、`deploy/hf_space/app.py` 加自适应补齐（pad → 推理 → 裁回）。

---

## 三、待办

### P0 — 安全收尾

- [ ] 吊销并轮换开发期使用的 GitHub PAT
  - 本地 `.git/config` 明文凭证已清除（remote 恢复为无凭证 URL）；
  - 全仓库扫描确认历史文档中的 token 已脱敏，未进入 Git 历史；
  - 该 PAT 曾在开发过程中经 `ghproxy.net` 明文用于推送，应视为已泄露，需到 GitHub → Settings → Developer settings 中吊销并轮换；
  - 后续推送改用一次性凭据助手注入（不落盘）。

### P1 — 延伸材料

- [ ] 生成英文项目报告（可用本地 `docx` / `pdf` 工具）。
- [ ] 生成项目演示幻灯片（可用本地 `pptx` 工具）。
- [ ] 生成 PSNR/SSIM 对比表与训练超参记录表（可用本地 `xlsx` 工具）。

### P2 — 可选

- [ ] `SRCNN ×2` 训练：当前为 `TBD`，可用 `train/train.py --model srcnn --scale 2` 补训。
- [ ] 前端部署：Vercel 导入 `web/`，`NEXT_PUBLIC_API_URL` 指向后端。

---

## 四、已知风险与备注

| 项 | 说明 |
|---|---|
| 开发期 PAT | 见 §三 P0；应视为已泄露，需吊销并轮换 |
| 无真实权重（历史） | 沙箱仅有 CPU 无法训练；已通过 AutoDL 训练 + 上传 `.pt` 补全权重 |
| SR 感知损失缺陷 | 原实现中 VGG 输入未做 ImageNet 归一化、像素项被 `0.01` 系数抹除；已在 `train/train.py` 修复，并已重训验证 |
| 权重已进 git | `.gitignore` 未忽略 `serve/models/*.pt`，`sr_generator_scale4.pt` / `lowlight.pt` 随仓库分发，`git clone` 即得可运行项目 |
| 受限网络推送 | 沙箱直连 GitHub 被 egress 白名单拦截，已用 `ghproxy.net` 镜像解决（见 `DEPLOY.md` §4） |
