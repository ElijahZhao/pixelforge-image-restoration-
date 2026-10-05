# PixelForge 深度审计 · 第 17 轮：文档自洽性（叙述层交叉对账）

> 本轮聚焦文档之间、以及文档与真实代码之间的一致性——**讲的是否是同一套故事**。实读了 `README.md`、`DEPLOY.md`、`TOOLS_CHECKLIST.md`、`results/README.md`、`PROGRESS.md`，以及两个真实部署入口 `deploy/streamlit/streamlit_app.py`、`deploy/hf_space/app.py` 与它们各自的部署文档。

## 一、经得起查的部分（先给信用，避免滥诉）

这些被我逐条实读/实测核实，**没有问题**，应如实记录：

- **`/api/health` 端点真实存在且诚实**（已证实 `serve/app.py:54-63`）：返回 `sr_scale2 / sr_scale4 / lowlight` 三项的 `ml` 或 `classical` 引擎状态。其中 `sr_scale2` 因无 scale2 权重被如实标为 `classical`，不做假。
- **`README` 引用的 9 张 demo 图片全部真实存在**（`assets/banner.svg`、`demo_sr_*.jpg`、`demo_lowlight_*.jpg`、`sample_*.png`），"效果演示"章节不会因缺图而破图。
- **快速开始依赖的文件真实齐全**：根 `requirements.txt`、前端 `web/.env.local.example` 均在位，README 的 `pip install -r requirements.txt` 与 `cp .env.local.example .env.local` 可机械执行。
- **两个真实部署入口都默认 ×4（确实有权重 → 走 ML）**：`streamlit_app.py:302` 的 scale radio 默认 `index=1="4"`；`hf_space/app.py:155` 的 scale radio 默认 `value="4"`。对默认路径，"用自训模型驱动"是字面成立的。
- **`results/README.md` 的诚实底线**：`SRCNN 2×` 明确标 `TBD`（未训练），没有把占位当真实结果——这一点和第 5/16 轮对 `method/page.tsx` 的判断一致，需肯定。

## 二、文档之间的矛盾与"选择性诚实"（核心发现）

### A. README 状态表 vs TOOLS_CHECKLIST —— 同一仓库，直接矛盾

| 事实 | README（状态表，line 57-59） | TOOLS_CHECKLIST（line 4 / 67） |
|---|---|---|
| 真实训练权重 | ✅ 已完成 100% | "缺口是真实权重……必须你去 Colab 训，服务才会从 classical 切到 ML" |
| 线上部署 | ✅ 已上线 100% | "缺口是线上部署" |
| 自动化测试 | （隐含完成） | "缺口是自动化测试" |

**两篇文档对"核心交付物（自训权重）是否存在"给出相反结论。** 读 README 的人以为权重已训好随仓库分发；读 TOOLS_CHECKLIST 的人以为权重是空缺口、需自己去训。懂行的人会因此怀疑整个项目状态表的可靠性。

### B. 四份部署文档，描述 2–3 套拓扑，根 `DEPLOY.md` 已废弃却被当部署指南

- **README「部署上线」+ 页眉徽章**：Streamlit Community Cloud 是**唯一公开 Demo**；Vercel + HF Spaces 只是"可选/备选"。
- **根 `DEPLOY.md`（70 行）**：全文只讲 **Vercel（前端）+ HF Spaces（后端）**，**只字未提 Streamlit**。
- **`deploy/streamlit/DEPLOY_STREAMLIT.md`**：明确解释"**为什么换成 Streamlit**"——HF Spaces 自 2026-07 起 Gradio 算 GPU 收费，故改用免费 Streamlit。
- **`deploy/hf_space/DEPLOY_HF.md`**：仍把 HF Spaces 当"方案 B 最省事"，并写"已在沙箱验证的事实：SR ×4: ML · Low-light: ML（非基线兜底）"。

**结论**：根 `DEPLOY.md` 描述的是一个**既非上线方案（Streamlit）、也非推荐方案（按 STREAMLIT 文档已弃用）**的陈旧拓扑。一个按 `DEPLOY.md` 操作的人会部署出一套没人维护的 Vercel+HF 栈。DEPLOY_HF 与 DEPLOY_STREAMLIT 又对"哪个是主路径"互相打架。**四份部署文档 = 至少两套互相矛盾的故事。**

### C. "非基线兜底 / 由自训模型实时驱动" —— 端点诚实，UI 选择性展示

- 后端 `/api/health`（line 59）**诚实**暴露 `sr_scale2: classical`。
- 但两个 UI 徽章都**只展示 ×4 与 low-light 状态**（`streamlit_app.py:285-286` 的 `sr4_tag`/`low_tag`；`hf_space/app.py:126-133` 的 `sr4`/`low`），**完全不展示 scale2 的 classical 回退**。
- 用户选 ×2 时，`predict_sr(img,2)` 返回 `None` → 静默回退 `sr_classical`，**界面无任何提示**（R4/6/16 已指出的 ×2 陷阱，本轮确认它在两个部署入口都存在、且被徽章刻意隐藏）。
- 叠加第 10 轮事实：自训模型**实测劣于**经典基线（SR 22.06 vs bicubic 33.25 dB；低光 12.68 vs gamma 19.62 dB）。"由自训模型驱动"字面真，但模型比它对比的基线还差；"非基线兜底"在默认路径成立，却用徽章把 ×2 的兜底藏了起来。

### D. "低光 U-Net 已验证胜出"（README line 140 / 324）vs 现实

- README：量化严谨性"对比经典基线 vs 自训模型，结论可解释（低光 U-Net **已验证胜出**）"；表格称 U-Net 19.26 dB"已高于该范围"。
- 但 `results/README.md:31` 自己承认：伽马基线 15–17 dB 是**文献参考范围，"非本项目测试划分上自测"**；而第 10 轮在本项目自带合成样本上实测 U-Net（12.68 dB）**输给**伽马（19.62 dB）7 dB，且 19.26 这个数字因仓库无数据/无日志（`results/` 仅含 README）**无法复现**。
- **"已验证胜出"是对一个文献参考带的断言，不是本项目数据上的真头对头胜出；且项目自身样本与之矛盾。**

### E. "SR×4 17.20 dB 低于 bicubic 是刻意取舍、非 bug"（README line 122 / 327）vs 根因

- README 把 SR×4 的劣于基线包装为"VGG 感知损失刻意牺牲像素换观感"。
- 第 1/3/11 轮已实证的真因：`loss = 0.01*loss + percep`（像素项被乘小 100 倍）+ VGG 输入未做 ImageNet 归一化（94% 特征退化为零）。这是**配置错误，不是设计取舍**。把 bug 写成"tradeoff"是叙述对缺陷的掩盖。

### F. "真实评测指标 PSNR/SSIM ✅ 已出 100%"（README line 58）vs 仓库实际

- 该状态行声称指标已出 100%，但：
  1. `results/` 目录**仅含 `README.md`，无任何训练日志 CSV、无测量数据产物**（已核实）。
  2. 同一 README 自身的"评测指标"表格里 `SRCNN 2×` 仍是 `TBD`（未训练）——**状态表说 100% 完成，指标表却有 TBD，README 内部自相矛盾**。
- "真实评测指标已出"在可复现性上不成立：数字不可证伪。

### G. "build-passing" 徽章（README line 10）无 CI 支撑

- 仓库**无 `.github/workflows/`**，没有任何 CI 配置。该徽章是静态装饰，暗示有自动化构建/测试流水线，实际不存在。对申研项目这是小瑕疵，但懂行的人一眼识破。

### H. "4.8M / 2.3M 权重" —— 字节数被读作参数量（与第 11 轮同）

- `DEPLOY_HF.md:17-18`、`README:80` 写 `sr_generator_scale4.pt 4.8M`、`lowlight.pt 2.3M`。这是 **文件字节大小（MB）**，但紧贴文件名读作"4.8M 参数"。真实参数量：SR 1.224M、U-Net 0.565M（第 11 轮实读）。**易让读者高估模型规模。**

## 三、本轮结论

> **项目的"叙述层"由多篇文档拼成，彼此未对齐：状态表与 TOOLS_CHECKLIST 对"权重是否完成"直接打架；四份部署文档描述至少两套矛盾的拓扑，根 `DEPLOY.md` 已废弃却仍被当部署指南；"非基线兜底/已验证胜出"等强结论在端点层面诚实、却在 UI 与指标层面选择性展示或无法复现。**
>
> 能力（health 端点诚实、图片齐全、快速开始可跑、默认路径真用 ML）都在场；**问题是叙述的"一致性与可证伪性"缺失**——读者无法从文档里得到一套自洽、可独立验证的真相。这和第 13 轮"训练—评估—验证闭环不自洽"是同一病灶在文档层的投影。

## 四、校准与修正

- **不滥诉**：先逐条核实"经得起查"的部分（health 端点、图片、依赖、默认路径真用 ML、SRCNN TBD 诚实），再指出矛盾。其中 health 端点暴露 `sr_scale2: classical` 比 UI 徽章更诚实，这点必须点明以免误伤。
- **修正过度指控**：第 5/16 轮称 `method/page.tsx` "完全 stale"——实则其 TBD 占位本身诚实；本轮再次确认 `results/README.md` 的 SRCNN TBD 也是诚实的。矛盾点在"已验证胜出""刻意取舍"这类**强结论**，不在占位本身。

## 五、结论依据

- 所有指控均基于实读（`README.md` 全篇、`DEPLOY.md`、`TOOLS_CHECKLIST.md`、`results/README.md`、`deploy/streamlit/streamlit_app.py`、`deploy/hf_space/app.py`）与命令核查（`/api/health` 源码、`assets/` 文件存在性、`.github/` 缺失、`results/` 内容）。

## 六、遗留盲区

- `PROGRESS.md` 与已发布 Demo 的实际可用性（公开链接是否仍可访问、冷启动行为）；
- `scripts/make_demo.py` 生成的 demo 图是否暗示了"模型胜出"叙事。
