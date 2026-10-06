# Changelog

本项目的所有重要变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-10-05

补齐工程完备性：测试与 CI 覆盖到前端，加上依赖漏洞扫描、接口限流，并补了 API 与运维文档。核心指标与训练权重未改动。

### Added

- `docs/API.md`：手写接口参考（端点、字段、错误码、环境变量），不再只依赖 FastAPI 自动生成的 `/docs`。
- `docs/OPERATIONS.md`：部署形态、健康检查、日志、回滚方式与已知缺口清单。
- `scripts/download_data.sh`：建数据集目录结构 + 打印下载地址 + 校验就位情况（不自动下大文件）。
- 前端单元测试（`web/lib/api.test.ts`，vitest），覆盖 `predict()` 的端点选择与错误映射。
- CI 新增 `audit`（pip-audit 扫锁定依赖）与 `frontend`（tsc + vitest + next build）两个 job。
- 覆盖率门槛：`pytest-cov`，阈值 75%（配置在 `pyproject.toml`），当前实测 79.7%。
- `/api/predict` 的按 IP 令牌桶限流（进程内内存实现，内置默认值可用环境变量调整），超限返回 429。
- `Dockerfile` + `.dockerignore`：推理服务的容器镜像（Python 3.11-slim，非 root 运行，带 `HEALTHCHECK`，内置权重）。前端不在该镜像内。
- `README.en.md`：英文入口文档。中文 `README.md` 仍是主文档。

### Changed

- README 功能特性按实际能力分述：4× 由自训 SRResNet 驱动，2× 走经典兜底（SRCNN ×2 未训练）。此前笼统写「支持 2×/4×」与实现不符。
- E2E 脚本 `tests/e2e/e2e.py` 改名 `test_e2e.py`，并加模块级 `pytest.skip`，确保它不会被 CI 当成浏览器用例拉起。
- 测试计数随本轮更新：28 → 29 passed（新增限流测试），另有 1 skipped（E2E）。

## [1.0.0] - 2026-10-04

首个正式发布版本：端到端图像复原流水线（超分辨率 + 低光增强）已闭环、可复现、已上线公开 Demo。

### Added

- 训练侧：SRCNN / SRResNet（含 PixelShuffle 与残差块）超分模型、低光 U-Net；DIV2K / LOL 数据加载；PSNR / SSIM 指标（高斯窗实现）；训练脚本（Adam + 余弦退火 + AMP + 可选 VGG 感知损失）与 TorchScript 导出。
- 推理侧：FastAPI 服务（`serve/app.py`），含经典方法兜底（Bicubic / 自适应伽马），无权重时开箱即用、有 `serve/models/*.pt` 时自动切换到 ML 引擎；`GET /api/health` 暴露当前引擎。
- 前端：Next.js 14 + Tailwind + TypeScript 上传与前后对比滑块。
- 部署：Streamlit Community Cloud 公开 Demo（`deploy/streamlit/`）已上线；HF Spaces 备选包。
- 测试：pytest 套件 28 项，20 个训练侧单元测试（含 8 项正确性测试：VGG 归一化、损失权重量级、SR 输出尺寸契约、数据管线配对一致性）+ 8 项 API 冒烟测试（全离线、无浏览器）。
- CI：GitHub Actions 工作流（`.github/workflows/ci.yml`），在 push / PR 时自动运行 pytest。
- 依赖锁定：`requirements.lock.txt` / `requirements-dev.lock.txt` / `deploy/streamlit/requirements.lock.txt`（pip-tools 生成，固定全部传递依赖）；前端 `web/pnpm-lock.yaml`。
- 自训权重：`serve/models/{sr_generator_scale4.pt, lowlight.pt}` 随仓库分发，`git clone` 即得可运行项目。
- 可复现证据：逐 epoch 训练日志（`results/train_log_*.csv`）、同口径评测脚本（`scripts/eval_baseline.py`）。
- 文档：中文 README、`PROGRESS.md`（进度与待办）、`DEPLOY.md`（部署与受限网络推送）。

### Changed

- 修复感知损失缺陷并重训：VGG 特征提取前补 ImageNet 归一化、像素/感知项权重显式化；SR ×4 由低于 bicubic 修复为 +0.77 dB（27.47 vs 26.69），低光相对不处理基线 +10.41 dB（18.18 vs 7.77）。
- 消除假放大：`predict_sr` 返回模型真实分辨率（`lr * scale`），移除二次 PIL 上采样。
- 低光修复：修正「越增强越暗」（训练域不匹配，改为加曝光门控）、padding 改用边缘反射消除接缝、补输出守卫。
- Streamlit 前端：复古像素风精修、双语（英/中）、暗/亮双主题，修复侧栏收起/展开与主题注入等部署期问题。
- 统一对外指标口径为全图验证值（与训练日志的随机裁剪口径区分说明）。

### Fixed

- 安全 / 入库审查：修复路径穿越、并发竞态、复现性、边界条件等 18 项缺陷；部署依赖契约由 `streamlit>=1.30` 上修为 `>=1.49`（`st.image(width="stretch")` 自 1.49 起支持）。
- 测试有效性：将形状测试升级为正确性测试，并用变异测试确认新套件能拦截真实缺陷；`test_loss_weights` 改读真实默认值。
- 推理服务新增 10 MB / 4 MP 输入限制（超限返回 413）。

### Security

- 全仓库扫描：无明文令牌、无敏感文件进入 Git 历史。
- 安全收尾：开发期使用的凭据已轮换，仓库不保留任何长期有效令牌。

### Removed

- 清理根目录噪音：`TOOLS_CHECKLIST.md`、`COMMIT_PLAN.md`（过时的内部过程文档）。

---

## 版本说明

- 1.0.0：项目主体（功能闭环 / 真实权重 / 可复现指标 / 公开 Demo）完成，作为开源的稳定基线。
- 尚未包含：SRCNN ×2 的权重训练（当前为 `TBD`，可用 `train/train.py --model srcnn --scale 2` 补训）。

[1.0.0]: https://github.com/ElijahZhao/pixelforge-image-restoration-/releases/tag/v1.0.0
