# Changelog

本项目的所有重要变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.3.2] - 2026-11-02

新增文献口径评测协议（Y 通道 + 边缘裁剪），把"本仓数字折算出文献数字"从外推变成实测。**默认行为不变**：`eval_baseline.py` 不带 `--protocol` 时仍走 RGB 全通道，历史数字照旧复现。

### Added

- **`train/eval_protocol.py`**：`rgb_to_y`（BT.601 studio swing，`Y=(16+65.481R+128.553G+24.966B)/255`，等价 MATLAB `rgb2ycbcr`）、`shave`（每边裁 `border` px）、`psnr_y` / `ssim_y` / `evaluate_y`。SSIM 沿用 `metrics.ssim_per_image` 的高斯 11×11 / σ=1.5 窗（MATLAB `ssim` 默认），只在 Y 上算。
- **`scripts/eval_baseline.py --protocol {rgb,y,y_shave}`**：默认 `rgb`（行为与 1.3.1 完全一致）；`y_shave` 采用 `border=scale`（EDSR 惯例，×4 时等于 NTIRE 的固定 4px）；低光无倍率、误传 `y_shave` 直接报错而不是静默按 0 处理。
- **`train/tests/test_eval_protocol.py`**：10 条 golden test。解析型断言包括三原色 Y 值 81.481 / 144.553 / 40.966、灰度仿射权重和 219（这正是"Y 域误差 = 219/255 × RGB 常数偏移"的来源）、PSNR 解析式 `10log10(1/(c·d)²)`；结构性断言包括 shave 精确裁剪、只削边界损伤、越界裁剪显式报错、`psnr_y` 与产出 28.91/33.82 的内核实现**逐位相等**、批次划分不改变单图得分。

### Verified

- **同一批权重、同一 100 张 DIV2K valid、同一个 `eval_baseline.py`，只换 `--protocol`**（Kaggle T4，HEAD `be4aee8`，六个配置全跑，全部 delta = 0.00）：

  | 槽位 | RGB（本仓） | Y + shave（文献） | Δ |
  |---|---|---|---|
  | SR ×4 模型 | 27.47 / 0.7798 | **28.91 / 0.8026** | +1.44 |
  | SR ×4 bicubic | 26.69 / 0.7541 | 28.10 / 0.7762 | +1.41 |
  | SR ×2 模型 | 32.35 / 0.9165 | **33.82 / 0.9275** | +1.47 |
  | SR ×2 bicubic | 31.04 / 0.8935 | 32.43 / 0.9046 | +1.39 |
  | 低光模型 | 18.32 / 0.7457 | **20.59 / 0.8444** | +2.27 |
  | 低光不处理 | 7.77 / 0.1924 | 9.28 / 0.3928 | +1.51 |

- **第八轮的外推偏大 1 dB**：当时的估计是"RGB → Y+shave 约 2.4–2.8 dB"，同权重实测量为 1.44（×4）/ 1.47（×2）。**以实测列为准，外推值作废。**
- **增益几乎不随口径变化**（×4：≈+0.78 → +0.81 dB；×2：+1.31 → +1.39 dB），但**绝对值不能跨表搬运**——这正是需要两列并存、而不是换掉旧列的理由。
- 低光 Y 通道位移 2.27 dB 明显大于 SR 侧（1.44），与文献在低光基准上报告的 luma-vs-RGB 差值同向。

## [1.3.1] - 2026-11-02

文档事实修正：×2 权重的模型身份、方法页过时表格、评测口径表述。**不改任何代码行为**（前端文案除外）。

### Fixed

- **×2 模型身份全仓修正（SRCNN → SRResNet 生成器，17 处）**：`sr_generator_scale2.pt` 从来不是 SRCNN。铁证三条互相独立：① 权重结构实测含 `entry → res_blocks.0..7 → upsample`（PixelShuffle），参数 781,068，SRCNN 是三层 conv（约 5.7k 参数）；② 训练日志名由 `train.py` 的 `--model` 参数生成，留档日志为 `train_log_sr_generator_scale2_batch16_20261007.csv`；③ `build_model` 仅在 `--model srcnn` 时构建 SRCNN。修正覆盖 `README.md` / `README.zh-CN.md` / `results/README.md` / `REPORT.md` / `docs/API.md` / `deploy/streamlit/DEPLOY_STREAMLIT.md` / 交付材料与 CHANGELOG 1.2.0 历史条目（原文误记，加更正注）。**最重的一处不是改名**：README 的 ×2 复现命令原写 `--model srcnn`，照它跑不出 32.35（会训出一个真 SRCNN），已改 `--model generator`。代码级的 SRCNN 基线描述（`models.py`、架构清单、参考文献）不受影响、保留。
- **方法页评测表双重过时（`web/app/method/page.tsx`）**：×2 行仍显示 "SRCNN / TBD"（权重 1.2.0 已训出部署）；低光 ours 行仍是旧数字 18.18 / 0.739（部署为 18.32 / 0.746）；×2 的 bicubic 行是 Set5 参考值 33.66*，与本仓 DIV2K 口径不可比。现全表统一为本仓实测值：×2 bicubic 31.04 / 0.894、SRResNet 32.35 / 0.917、低光 U-Net 18.32 / 0.746，脚注同步改为口径说明。
- **评测口径表述自相矛盾（`results/README.md`）**：原文 "(full-image, one fixed crop)"——"full-image" 与 "one fixed crop" 互斥。厘清：表格数字由 `eval_baseline.py` 在**整张图**上测得（"full-image" 对这三个数字是准确的）；训练日志才是 96×96 中心块口径。并补实测差值：同一部署权重 × DIV2K val-100，中心块口径比全图**低 1.17 dB（×4）/ 1.42 dB（×2）**。

### Verified

- **三个报告数字在独立环境逐位复现**（Kaggle T4 + torch 2.11.0+cu128，本仓 HEAD `f76ca96`，官方 DIV2K valid-100 + LOL eval15，`eval_baseline.py` 未改一字）：×4 **27.47 / 0.7798**、×2 **32.35 / 0.9165**、低光 **18.32 / 0.7457**，与 `results/README.md` 两/四位小数一致；同 kernel 重跑逐位相同。顺带量化：低光逐图平均（1.3.0 口径）18.3205 vs 全局 MSE（旧路径）17.1645，旧口径低估 **1.156 dB**，与 1.3.0 条目在 8×256² 块上实测的 1.16 dB 方向、量级一致。

## [1.3.0] - 2026-11-01

指标改为**逐图计算再平均**，验证批大小与训练解耦，退化算子的抗锯齿行为显式写死。**报告口径已变，新旧训练日志的 `val_psnr` 不可直接比较**（见下）。

### Changed

- **指标聚合改为逐图（`train/metrics.py`）**：原实现用一个 batch 级全局 MSE 得到一个标量（`F.mse_loss(pred, target, reduction="mean")` 覆盖 B×C×H×W），`train.py` 再对这些 batch 标量求算术平均。这样报告值会随**验证批大小**变化——同一个模型、同一批图，仅改分组就差 1.16 dB（8 个 256² 块实测：逐图平均 26.172，batch=8 全局 MSE 25.009，拆成两组 4 张则 25.051）。现改为每张图各自算 PSNR/SSIM 再等权平均，聚合结果在结构上不再能"看见" batch。
- **验证批大小与训练解耦（`train/train.py`）**：新增 `--val_batch_size`（默认 1），不再复用 `--batch_size`。此前换个训练 batch 会顺带重排验证集，让指标变成训练超参的函数。
- **CSV 记录口径（`results/*.csv`）**：新增 `# val_aggregation=per_image_mean` 与 `# val_batch_size=<n>` 两行表头。不写清聚合方式的日志无法与另一份日志比较。
- **退化算子写死抗锯齿（`train/datasets.py`）**：`TF.resize(..., BICUBIC)` 显式补 `antialias=True`。该参数默认值在 torchvision 0.17 由 False 翻为 True，而本项目训练环境不锁 torchvision 版本，训练机与评测机可能不同版本。实测影响：同一权重同一张图，关掉抗锯齿后模型相对 bicubic 的增益从 **+0.57 dB 翻成 −1.19 dB**——胜负取决于这个开关，不能留给库默认值。

### Added

- `train/metrics.py`：`psnr_per_image` / `ssim_per_image`（返回 `(B,)`）、`aggregate()`（等权平均的唯一实现）、`psnr`/`ssim` 新增 `reduction="none"`、`evaluate_batch(..., per_image=True)`。
- `train/tests/test_protocol.py`：7 条测试锁定协议——逐图指标与 batch 划分无关（PSNR/SSIM 各一条）、聚合不等价于全局 MSE、`evaluate_batch` 自洽、等权平均、完美图不污染均值（inf 语义）、`datasets` 产出的 LR 必须等于显式 `antialias=True` 的结果且不等于 `False`。
- `train/tests/run_tests.py` 纳入 `test_protocol`（34 项全绿）。

### 影响范围（重要）

- **`scripts/eval_baseline.py` 的数字不变**：它本来就是逐图（batch=1）评测，旧口径在 batch=1 时与新口径等价。`results/README.md` 对比表里的 32.35 / 27.47 / 18.32 **不受本次改动影响**。
- **训练日志的 `val_psnr` 会系统性升高**：旧口径被 batch 内最难的那张图拉低。8 个 256² 块实测低估 1.16 dB；batch=16 且图像难度分散时更大。因此 `results/train_log_*_20261007.csv` 与今后新日志**不可直接相减**，跨协议比较一律以 `eval_baseline.py` 为准。
- **不影响已发布权重**：本次未改动任何模型结构或权重文件。

## [1.2.2] - 2026-10-08

补全上传守卫时序缺口、清理死依赖、固化 CI 覆盖率门禁、修正文档与过时测试。

### Fixed

- **Streamlit 上传守卫时序缺口（修订 1.2.1 的 OOM 修复）**：1.2.1 的守卫在 `Image.open(uploaded).convert("RGB")` **解码之后**才检查尺寸，超大图在解码阶段（如 8000×8000 ≈ 192MB RGB）已分配全图内存才被拦，仍可能压垮免费档 ~1GB 内存而 "Oh no." 反复重启。改为**先读 header `.size` 再决定**：超限图先等比缩小（LANCZOS）后解码；并新增 `Image.MAX_IMAGE_PIXELS = 50_000_000` 硬上限 + 显式 header 尺寸判断，超过 50 Mpx 在**解码前**直接拒绝（双语 `bomb_error` 提示 + `st.stop()`），绝不分配其位图。本机 playwright 实测：4000×3000 → 缩至 1265×949 正常出图；9000×9000 → 解码前拒绝、零崩溃。
- **`serve/model_loader.py` 注释与过时集成测试**：注释原称 "x2 has no trained weight / falls back to bicubic"，但 `serve/models/` 确有 `sr_generator_scale2.pt`、`get_sr_model(2)` 加载真模型——已更正为「SR 两档均发真权重，无 bicubic 回退」。随之修正 `tests/test_inference_integration.py` 中已**失败**的 `test_sr_scale2_falls_back_to_classical`（旧假设残留）→ `test_sr_scale2_uses_ml_engine_and_keeps_resolution`，并扩展 `_require_weight` 支持 `sr2`。CI 此前该测试红，现已绿。

### Changed

- **清除死依赖 gradio**：`requirements.txt` 移除 `gradio>=4.0`（仅服务于已删的 `serve/gradio_demo.py`）；`requirements.lock.txt` 手术式剪除 gradio 独占依赖闭包（gradio / gradio-client / hf-gradio，共享依赖如 fastapi/pydantic/httpx 保留），避免镜像徒增无用包。
- **CI 覆盖率门禁显式化**：pytest 命令加 `--cov-fail-under=75`，与 `pyproject.toml` 既有 `fail_under` 对齐，使门禁不依赖 pytest-cov 版本行为。
- **CI 纳入训练单元测试**：test job 新增 `python -m train.tests.run_tests` 显式步骤，训练逻辑回归在独立带标签的 gate 下暴露（离线缺失依赖时 skip 非 fail）。
- **`deploy/streamlit/requirements.txt` 加独立说明**：注明该文件为何独立于根 `requirements.txt`/`requirements.lock.txt`（Community Cloud 仅 CPU、pin CPU-only torch，避免把 CUDA/nvidia 19 个包拖进免费档）。

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

- `serve/models/sr_generator_scale2.pt`：首个自训 ×2 超分权重（SRResNet 生成器，AutoDL RTX 4090 batch16；1.2.0 条目原文误记为 SRCNN，2026-11-02 依权重结构与训练日志名更正）。全图验证 **32.35 / 0.917**，胜 bicubic 基线 31.04 +1.31 dB；此前 ×2 请求一直回退经典 bicubic（1.1.1 中"×2 无训练权重"的临时限制解除）。
- `deploy/streamlit/models/sr_generator_scale2.pt`：Streamlit demo 同步上线 ×2 真模型档位（加载逻辑按 `sr_*_scale{scale}.pt` 自动识别，零代码改动）。
- `results/train_log_*_batch16_20261007.csv` ×3 与 `results/logs_batch16_20261007/`：第 ②③ 次训练（batch16）完整日志留档，与第 ① 次（batch8）构成超参对照实验证据。
- `archive_private/docs/MODEL_SELECTION_TODO.md`（已移出主仓库至私有归档）：三次训练按槽位择优的决策记录（含评估口径说明、备份路径坑位与恢复验证）。

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
- `archive_private/docs/OPERATIONS.md`（已移出主仓库至私有归档）：部署形态、健康检查、日志、回滚方式与已知缺口清单。
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
