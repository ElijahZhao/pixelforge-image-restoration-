# Changelog

本项目的所有重要变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
- 可复现证据：逐 epoch 训练日志（`results/train_log_*.csv`）、训练过程与平台凭证（`docs/retrain_journey/`）、同口径评测脚本（`scripts/eval_baseline.py`）。
- 文档：中文 README、`PROGRESS.md`（进度与待办）、`PROJECT_NOTES.md`（项目说明与边界）、`DEPLOY.md`（部署与受限网络推送）、`docs/history/`（开发期诊断记录）。

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

- 全仓库扫描：无明文令牌、无敏感文件进入 Git 历史；诊断文档中的历史 token 已脱敏为 `***REDACTED***`。
- 遗留运维项：一个曾在开发与推送过程中使用的 GitHub PAT 应视为已泄露，需由仓库所有者到 GitHub 吊销并轮换（详见 `PROGRESS.md` §四）。

### Removed

- 清理根目录噪音：`TOOLS_CHECKLIST.md`、`COMMIT_PLAN.md`（过时的内部过程文档）；26 份诊断/报告 md 归档至 `docs/history/`（原文归档保留）。

---

## 版本说明

- 1.0.0：项目主体（功能闭环 / 真实权重 / 可复现指标 / 公开 Demo）完成，作为开源的稳定基线。
- 尚未包含：SRCNN ×2 的权重训练（当前为 `TBD`，可用 `train/train.py --model srcnn --scale 2` 补训）。

[1.0.0]: https://github.com/ElijahZhao/pixelforge-image-restoration-/releases/tag/v1.0.0
