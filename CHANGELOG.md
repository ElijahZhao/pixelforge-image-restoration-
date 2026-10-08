# Changelog

本项目的所有重要变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.2.2] - 2026-10-08

项目审查（提示词库 5.0 · 任务④ 修缮清单 R1–R6）落地：补全上传守卫时序缺口、清理死依赖、固化 CI 覆盖率门禁、修正文档与过时测试。

### Fixed

- **Streamlit 上传守卫时序缺口（R1，修订 1.2.1 的 OOM 修复）**：1.2.1 的守卫在 `Image.open(uploaded).convert("RGB")` **解码之后**才检查尺寸，超大图在解码阶段（如 8000×8000 ≈ 192MB RGB）已分配全图内存才被拦，仍可能压垮免费档 ~1GB 内存而 "Oh no." 反复重启。改为**先读 header `.size` 再决定**：超限图先等比缩小（LANCZOS）后解码；并新增 `Image.MAX_IMAGE_PIXELS = 50_000_000` 硬上限 + 显式 header 尺寸判断，超过 50 Mpx 在**解码前**直接拒绝（双语 `bomb_error` 提示 + `st.stop()`），绝不分配其位图。本机 playwright 实测：4000×3000 → 缩至 1265×949 正常出图；9000×9000 → 解码前拒绝、零崩溃。
- **`serve/model_loader.py` 注释与过时集成测试（R5）**：注释原称 "x2 has no trained weight / falls back to bicubic"，但 `serve/models/` 确有 `sr_generator_scale2.pt`、`get_sr_model(2)` 加载真模型——已更正为「SR 两档均发真权重，无 bicubic 回退」。随之修正 `tests/test_inference_integration.py` 中已**失败**的 `test_sr_scale2_falls_back_to_classical`（旧假设残留）→ `test_sr_scale2_uses_ml_engine_and_keeps_resolution`，并扩展 `_require_weight` 支持 `sr2`。CI 此前该测试红，现已绿。

### Changed

- **清除死依赖 gradio（R2）**：`requirements.txt` 移除 `gradio>=4.0`（仅服务于已删的 `serve/gradio_demo.py`）；`requirements.lock.txt` 手术式剪除 gradio 独占依赖闭包（gradio / gradio-client / hf-gradio，共享依赖如 fastapi/pydantic/httpx 保留），避免镜像徒增无用包。
- **CI 覆盖率门禁显式化（R3）**：pytest 命令加 `--cov-fail-under=75`，与 `pyproject.toml` 既有 `fail_under` 对齐，使门禁不依赖 pytest-cov 版本行为。
- **CI 纳入训练单元测试（R4）**：test job 新增 `python -m train.tests.run_tests` 显式步骤，训练逻辑回归在独立带标签的 gate 下暴露（离线缺失依赖时 skip 非 fail）。
- **`deploy/streamlit/requirements.txt` 加独立说明（R6）**：注明该文件为何独立于根 `requirements.txt`/`requirements.lock.txt`（Community Cloud 仅 CPU、pin CPU-only torch，避免把 CUDA/nvidia 19 个包拖进免费档）。

## [1.2.1] - 2026-10-08

README 展示页全面换新（低光 ×2 / SR×4 / SR×2 三档实测截图与案例图）；修复公开 Demo 反复崩溃（上传大图 OOM）。

### Added

- `assets/` 新增 7 张真实素材：低光 Demo 截图 ×2（暗色主题）、SR×4 Demo 截图 ×1（亮色主题，LR 111×170）、SR×2 Demo 截图 + ×2 案例三联图（原始上传 446×683 / 模型输入 LR 223×341 / ×2 输出 446×682）。`README.md` 与 `README.zh-CN.md` 的 "Running live" 展示段按任务分档重写，旧 `demo_live_lowlight.png` / `demo_live_sr.png` 移除。

### Fixed

- **公开 Demo 反复崩溃（根因：上传大图 OOM）**：`deploy/streamlit/streamlit_app.py` 此前对上传图片无像素上限，低光 U-Net 以全分辨率跑激活、SR×4 输出张量按 宽×高×12 字节膨胀，一张大图即可把 Community Cloud 免费档（约 1GB 内存）进程 OOM 杀掉，表现为 "Oh no." 空白页并反复重启。现新增 `MAX_INPUT_PIXELS = 1_200_000` 上传守卫：超限图自动等比缩小（LANCZOS）后再推理，并以双语 `st.caption` 明示缩放前后尺寸；只缩不放、绝不拒绝用户图。经本机 playwright 实测：1600×1200 上传 → 自动缩至 1265×949，SR×4 与低光均正常推理无崩溃。

### Changed

- （随 8092fff / f63b9f9 补记）Streamlit demo UI：侧栏训练注释逐行卡片化排列；新增模型输入 PNG 下载按钮；三面板说明改为逐面板图例式排版（中英）；修复 `st.info` 不支持 `unsafe_allow_html` 导致的说明区渲染失败（改用 markdown 硬换行）。

## [1.2.0] - 2026-10-07

三次训练对照完成，按槽位择优更新部署权重；首个 ×2 超分权重上线。

### Added

- `serve/models/sr_generator_scale2.pt`：首个自训 ×2 超分权重（SRCNN，AutoDL RTX 4090 batch16）。全图验证 **32.35 / 0.917**，胜 bicubic 基线 31.04 +1.31 dB；此前 ×2 请求一直回退经典 bicubic（1.1.1 中"×2 无训练权重"的临时限制解除）。
- `deploy/streamlit/models/sr_generator_scale2.pt`：Streamlit demo 同步上线 ×2 真模型档位（加载逻辑按 `sr_*_scale{scale}.pt` 自动识别，零代码改动）。
- `results/train_log_*_batch16_20261007.csv` ×3 与 `results/logs_batch16_20261007/`：第 ②③ 次训练（batch16）完整日志留档，与第 ① 次（batch8）构成超参对照实验证据。
- `docs/MODEL_SELECTION_TODO.md`：三次训练按槽位择优的决策记录（含评估口径说明、备份路径坑位与恢复验证）。

### Changed

- **低光部署权重更新**：`serve/models/lowlight.pt` 与 `deploy/streamlit/models/` 由第 ① 次权重（18.12 / 0.743）换为第 ③ 次权重（**18.32 / 0.746**，PSNR / SSIM 双指标同向胜出）。
- **SR ×4 部署权重保持第 ① 次产物**：27.47 / 0.780（第 ② 次新训 27.38 以 0.09 dB 险负）。三个部署槽位在全图口径下全部打赢各自基线：SR×4 27.47 / SR×2 32.35 / 低光 18.32。
- Streamlit demo 文案更新为三次训练择优结论（×2 / ×4 均标注自训模型与实测 PSNR）。
- 修正评估叙事：训练日志 val 数字跨协议不可比（历史随机裁剪 vs 现行确定性协议，见 1.1.1），槽位决策一律以 `scripts/eval_baseline.py` 全图同口径对比为准；batch16 日志名义上的大幅落后（23.34 / 17.20）主要为协议差异假象，同口径下新模型与旧代持平（×4）或更优（×2、低光）。

## [1.1.1] - 2026-10-07

修正依赖安装内容与训练/验证口径。模型结构与对外指标口径未改动；**已发布权重的重新训练在另行进行中**。

### Fixed

- 依赖锁定为 CPU 构建：根 `requirements.lock.txt` 此前解析出 CUDA 版 torch 并连带 38 个 `nvidia-*` / `cuda-*` / `triton` 包，与 CPU 部署环境不符。CI 安装与 `pip-audit` 审计的也是这一份。
- `deploy/hf_space/requirements.txt` 的 torch 约束补上上限，与另两端一致。
- 训练验证口径改为确定性：SR 验证集此前每轮随机裁剪，实测极差 7.35 dB，而 200 轮真实增益仅 +0.46 dB；低光验证集此前把 400x600 压成 128x128 正方形，后 100 轮指标标准差仅 0.0457，无法区分 checkpoint 优劣。两处均改为不引入随机性的口径。
- 验证 loader 传入固定 seed，消除 worker 随机性。
- 训练脚本支持断点续训（`--resume` / `--auto-resume` / `--save_every`），状态写入改为原子替换，避免中断产生截断文件导致续训崩溃。
- `scripts/download_data.sh --check` 改为核对文件数量（DIV2K 800/100、LOL 485/15），此前只判断目录非空：放 1 张图也会打印"数据就绪"并放行，训练照常开始。有意使用子集时用新增的 `--allow-partial` 显式放行。

### Changed

- 超分倍率默认值统一为 ×4（Web、FastAPI、Gradio、Streamlit、HF Space 五端）：×2 无训练权重，默认落在 bicubic 兜底上容易被误读为 AI 结果。
- 训练日志文件名带上 scale，避免 ×2 与 ×4 互相追加。既有日志同步改名。
- 移除正文中的自指性时间词（`DEPLOY_DIAGNOSIS.md`、`CHANGELOG.md`）。

### Removed

- `deploy/streamlit/requirements.lock.txt`：与实际安装内容不符，且 Streamlit 实际读取的是同目录的 `requirements.txt`。

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
- `README.md` 为英文主文档，中文版见 `README.zh-CN.md`。

### Changed

- README 功能特性按实际能力分述：4× 由自训 SRResNet 驱动，2× 走经典兜底（SRCNN ×2 未训练）。此前笼统写「支持 2×/4×」与实现不符。
- E2E 脚本 `tests/e2e/e2e.py` 改名 `test_e2e.py`，并加模块级 `pytest.skip`，确保它不会被 CI 当成浏览器用例拉起。
- 测试计数更新：28 → 29 passed（新增限流测试），另有 1 skipped（E2E）。

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
