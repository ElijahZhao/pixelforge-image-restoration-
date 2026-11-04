# 论文素材总账（MATERIALS）

> **2026-11-04。** 这份文件回答一个问题：**每一节要论证什么、用哪个数字、那个数字在哪、还缺什么。**
> 三份文件的分工：`MATERIALS.md`（本节要什么）→ `FACTS.md`（数字是多少、出处）→ `OUTLINE.md`（怎么组织）。
>
> 纪律：**写作时不许翻代码、不许翻对话记录。** 本文件指向哪里就去哪里取；取不到就是缺口，记在 §5。

---

## 1. 定位：写什么，不写什么

**写的是**：一篇评测方法学短文。核心命题——*在轻量图像修复里，评测协议的选择能移动报告指标 **3.9 dB**（×4 的 P5→P6；若把 best-of-60 的 P1 也算进来是 19.8 dB），是模型自身改进（0.78 dB）的 **5 倍**（×4；×2 是 3.64×/18.1×，**低光 0.21×——方向相反，必须一起交代**，`FACTS` §6、E57）；只改一个 `antialias` 布尔值，结论从"优于基线 +0.82 dB"翻成"劣于基线 −1.75 dB"。*

> 数字口径：3.9 / 19.8 / 0.78 都是 **×4 任务**的值（×2 是 3.14 / 15.65 / 0.86，低光是 2.25 / 6.07 / 10.74）。摘要里若只报一组，必须标明是 ×4，不能让人读成"三任务平均"。
>
> **3.9 与 19.8 的关系要一句话讲清，否则读者会以为其中一个打错了**：3.8892 是**去掉 P1 之后** 7 个协议的跨度（P5 29.0523 → P6 25.1631）；19.7944 是**含 P1** 的跨度——P1（best-of-60）单独就把 44.9575 拉到比次高的 P5 高 **15.9052 dB**。**主数用 3.9**：P1 是极值统计，不是"协议选择"的常规效应；19.8 只能当"含极端采样协议时的上限"提，且必须说明 P1 是 best-of-60、其基线 n=95（§7.6 与 E21）。

**不写的是**：
- 不主张性能。三个模型都不是 SOTA：按 RGB 口径，低光落后 KinD **2.36 dB**（20.87 as reported, not re-run）。**不写 Retinexformer 的 6.65 dB**（25.16 − 18.51）——那是 Y 减 RGB（`FACTS` §9.3）；⚠️ 原写 6.8，与同句算式对不上，见 E60。
- 不主张系统。**81 项测试全部通过**（`tests/e2e/` 浏览器模块默认整模块跳过，需 `PIXELFORGE_RUN_E2E=1`，不计入）、CI、跨环境数值一致（**四项统计量差 0.0 到 1e-6 精度；验收判据 ≤1e-5 dB；不是 bitwise**）是**证据可信的前提**，放 §4.1，不单列为贡献。
- 不主张"首次发现协议影响指标"。Keleş 2021（聚合，SISR 实测 2.07 dB）、LOL audit 2026（低光标签与测试时设置）、Clean-FID 2022（GAN 侧 resize）都在这条线上。我们主张的是**首个跨任务、跨五维度的统一审计 + 符号翻转这一具体机制**。

**三条贡献（按可防御性排序）**：
1. 符号翻转的直接证据（×4 +0.8179→−1.7531，×2 +1.1456→−1.8889）——在经典 SR 保真评测上无人做过。
2. 统一矩阵：3 任务 × 8 协议 × 5 种子，每一维的孤立效应都可读出。
3. 63 篇近期论文的报告完整性抽样（聚合 1/63、验证 batch 0/63）。

---

## 2. 章节 → 素材映射

### Abstract（150–200 词）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 定位句 | "协议移动 3.9 dB，模型只动 0.78 dB" | `FACTS` §6 | ✅ |
| 翻转 | +0.8179 → −1.7531（×4） | `FACTS` §5.3 | ✅ |
| 文献盲区 | 1/63、0/63 | `FACTS` §8 | ✅ |
| 建议 | **六条**最小报告要求（摘要里可压成四条） | 需自己提炼（§6.2） | ⬜ 待写 |

### §1 Introduction（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 开场场景 | 学生把 bicubic 跑出 28.9、模型 29.0，报告 +0.1 dB；换脚本变 −1.7 | 构造场景，可基于真实数据改写成"27.61 vs 26.83 = +0.78，换算子后 −1.75" | ✅ 数据支撑 |
| 一图证明 | Fig. 1 | `paper/figures/fig1_protocol_matrix.pdf` | ✅ |
| 三条贡献 | 见 §1 | — | ✅ |

> 开场不要写"图像修复很重要"。写那个学生——他没做错任何事，只是没说清三行默认参数。

### §2 Related Work（**1.2 页**：2.1 0.2 / 2.2 0.7 / 2.3 0.3）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 2.1 轻量 SR / LLIE 一条线 | SRCNN→SRResNet→EDSR→RCAN；RetinexNet→KinD→Retinexformer | `paper/refs.bib`（已生成） | ✅ bib 已建 |
| 2.2 协议审计七篇 | 七篇"它做了什么 / 我们补什么" | `archive_private/05-论文规划/查新报告.md` §二 + 本文件 §4.1 | ✅ 表已成，**出处已核完** |
| **2.3 ML 评测批判谱系** | **查新报告缺这条线**，见下。**写法已定**：五篇一句话 + 一句与本文的分界（`OUTLINE` §2.3 有整段与表格），落点是"**五篇都在'训练与选模'这一侧记账（预算 2 篇、方法学 1 篇、方差/极值 2 篇），本文补'评测协议'这一侧**" | 本文件 §4.2（**五篇摘要已读**，用途逐条改写） | ✅ **素材与写法均齐**（2026-11-04） |
| Table 1 | 最近邻对比 | 查新报告改写 | ⬜ |

> **本节最大的增值点**：查新报告的七篇全是"图像修复领域内"的占位。补上"ML 评测批判"这条线（Lucic 2018 / Musgrave 2020 / Dodge 2019 / Bouthillier 2021 / Picard 2021），论文就从"一个领域的经验观察"升级为"接入 ML 评测批判的既有谱系"。**0 实验成本，只多读五篇。**⚠️ **但这五篇不能一律概括成"方差吞掉改进"**：Lucic 与 Dodge 讲的是**预算**，Musgrave 是**方法学**，只有 Bouthillier 与 Picard 讲方差——而 **Picard 的结论恰恰是"方差本身不大"**。概括句必须写成"报告的改进在重新记账后普遍站不住"，再分三条路径（`OUTLINE` §2.3、E58）。

### §3 Protocol Matrix（1.5 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 3.1 五类维度 | ①聚合 ②验证 batch ③采样位置与窗口 ④通道与边界 ⑤退化算子。**注意：8 个协议只覆盖 ①③④⑤ 四类，第 ② 类没有对应轴**——见 §7.10，这不是漏测，是逐图聚合把它消除了 | `FACTS` §4 + §7 + **§7.10** | ✅ |
| 每维"为什么原则上会改变数字" | 一句机制，不展开（展开在 §6.1）。**③ 采样维现在有逐图证据**：`grad_ratio` 与 `P2−P3` 的 **Spearman ρ = −0.9247**、两个 69 张集合**重合 64 张**（`FACTS` §7.4.1）。② 类要给旧口径下的实测：**batch 划分能移动 11.65 dB**（32 个 96² 块、主仓权重，`FACTS` §7.10） | `FACTS` §7.1–7.5 + **§7.4.1** + §7.10 | ✅ |
| 3.2 八个协议 | Table 2，逐列 `degrade/metric/sampling/agg` | `FACTS` §4 表 | ✅ 定义齐 |
| **P5 与 P6 只差一个布尔值** | 同库同函数同尺寸 | `FACTS` §4 注 | ✅ 全文最关键一行 |
| 3.3 分组实现 | 分组键是 **`(degrade, sampling)`**，共 **5 组**：3 次全图（P3/P4/P5 共用 1 次、P6 1 次、P7/P8 共用 1 次）+ P2 的 1 次 96² + **P1 的 60 次 96²**。**P1 的块是独立前向，不是从全图输出上裁**。T4/2040×1356：全图前向 ×4 **0.2683 s / 3.15 GB**、×2 **0.6624 s / 8.37 GB**；96² 单块 0.0034 s。**加速 ×4 = 1.80×**（1.8173→1.0124 s） | 探针 `kag-probe/probe_v1.json` + `FACTS` §4.1 | ✅ **2026-11-04 修正：原写"3 组/3 次前向/2.7×"三处都不准** |
| **66× 已禁写** | **不要写 66×，也不要写"3 次前向"或"2.7×"**。66× 全仓无推导、无单图 CPU 实测做分母，且与 4.2h/0.7h=6× 对不上；"3 次"是**全图**的，总共是 **64 次前向**；分组真实加速 **1.80×**（×4）。总耗时只可说"4.2 h 估算 → 0.86 h 实测，约 5 倍" | `FACTS` §4.1 + §10-E17 + `results/compute_ledger.md` §六 | ⚠️ **三个数字全错，见 FACTS §10-E17** |

### §4 Setup（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 4.1 三个模型 | **1.224 M / 0.069 M / 0.565 M**（可训练参数，实测；不是 1.226 —— 那个含 BN buffer，见 `FACTS` §10-E43） | `FACTS` §1 | ✅ |
| **×4 带感知损失** | 配置事实（config）：`w_pixel=1.0`、`w_percep=0.006`、`objective = 1.0*charbonnier + 0.006*vgg_perceptual`。**实测是惰性的**：成对消融 on 27.61 / off 27.60，**Δ −0.01 dB**，SSIM Δ 0.0001，锐度差 0.26%，参数收敛到同一处。**"1:8–1:17"是估算**（日志只记 total loss）。**禁写"以牺牲 PSNR 换观感"**——本项目没发生 | `FACTS` §3 + §7.9 + §10-E18/E32 | ✅ **必写，但写"设了没生效"** |
| 4.1 可复现性 | **81 项收集、全部通过**（`train/tests` 69 + `tests/` 12：API 冒烟 9、推理集成 3）；`tests/e2e/` 的浏览器 e2e **整模块默认跳过**（收集阶段，需 `PIXELFORGE_RUN_E2E=1`），所以 pytest 的汇总行是 `collected 81 items / 1 skipped`；CI 三门禁（pytest+覆盖率 75%、pip-audit、tsc+next build）；跨环境数值一致 **≤1e-5 dB（实测 1e-6），不是 bitwise** | `FACTS` §9.2 + §10-E10 | ✅ |
| 4.2 数据 | DIV2K 800/100；LOL 485/15；低光**配对无上采样** | `FACTS` §2 | ✅ |
| 4.2 种子与轮数 | 42–46；200/100/200 轮；**Adam + cosine 退火（T_max=总轮数）** | `FACTS` §3 | ✅ 2026-11-04 补：原漏了 LR 调度 |
| 4.3 环境 | T4 训练、本地 CPU 跑 ×2 基线；跨环境**四项统计量差 0.0（到 1e-6 精度）**，验收判据 ≤1e-5，**禁写 bitwise** | `FACTS` §9.2 | ✅ |
| **验证口径交叉验证** | 训练日志 best 对到协议矩阵 1e-3 内，SR 对 P2、低光对 P3 | `FACTS` §9.1 | ✅ 极强的可信度证据 |

### §5 Results（2.5 页，核心）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 5.1 方差分解 | 种子 std（**仅全图协议 P3–P8**）：×4 0.0190–0.0361 / ×2 0.0268–0.0938 / 低光 0.0346–0.0806 | `FACTS` §6 + §5.1 | ✅ |
| 5.1 **采样协议的种子 std 不可混入上一条** | 低光 P1 = **0.4148**、P2 = **0.5303**；×4 P1 = 0.2364、×2 P1 = 0.5584 —— 这些不是"训练随机性"，是**块采样噪声被 best-of-60 放大**（Fig. 2 的同一现象） | `FACTS` §5.1 | ✅ 写 §5.1 时必须分开说，否则表里 0.53 会打自己的脸 |
| 5.1 跨度 | 3.889（×4）/ 3.143（×2）/ 2.250（低光）；**比值三任务并列：4.98× / 3.64× / 0.21×**（含 P1 为 25.3× / 18.1× / 0.57×）——**低光是反例，训练增益是协议跨度的 4.8 倍** | `FACTS` §6 + E57 | ✅ 2026-11-04 补：原只算 ×4 |
| 5.1 最低协议不同 | ×4 最低 P6，×2 最低 P2 | `FACTS` §6 注 | ✅ |
| **5.1/7 绝对值的不确定度** | 低光 n=15：每图 PSNR **12.49–23.72**（极差 11.23），均值 18.509 的 95% CI 半宽 **±2.04 dB**；SR n=100 是 **±0.87** | `FACTS` §7.7 | ✅ **低光只报方向不报幅度** |
| 5.2 符号翻转 | 完整 21 行增益表 | `FACTS` §5.3 | ✅ |
| 5.2 机制数字 | 模型侧 **−3.89/−2.72**，基线侧 **−1.32 / +0.31**（×2 是**涨**的）；剪刀差 2.57 / 3.03 | `FACTS` §7.2 | ✅ 2026-11-04 修正：×2 基线符号原写反 |
| 5.3 跨任务 | ×2 复现且更大（3.03 dB 摆幅） | `FACTS` §5.3 | ✅ |
| **5.3 低光上的反例（必写）** | 低光协议跨度 **2.25 dB**、训练增益 **10.74 dB**，比值 **0.21** —— 本文命题在这个任务上**不成立**，因为 headroom 差一个量级（0.78 vs 10.74）。**不要只写"低光增益与 SR 不可比"就过去**，那读起来像回避；正面写"协议支配是 headroom 的函数"反倒立得住 | `FACTS` §6 | ✅ 2026-11-04 补（E57） |
| 5.3 **P1 不是"另一个翻转"，是"P1 根本不可比"** | ×4 **−0.5793**、×2 **−2.3123**（两边同取 95 张）。5 张 DIV2K 图（`0839/0843/0846/0847/0868`）被 bicubic 在 best-of-60 里**精确重建**，PSNR=∞，故基线 n=95；模型侧对应 5 张是 50–68 dB。**任何网络都不可能赢** | `FACTS` §5.3 + §10-E21 + `results/p1_inf_alignment.csv` | ✅ 2026-11-04 修正 |
| 5.4 文献报告 | **F1 聚合 1/63 [0.3, 8.5]%；F2 验证 batch 0/63 [0.0, 5.7]%；F3 退化可复现 5/44 [5.0, 24.0]%；F4 明说 Y 通道 22/63 [24.3, 47.2]%；F5 shave 边界 0/44 [0.0, 8.0]%；F6 val/test 同集 0/63 [0.0, 5.7]%；F7 给代码 35/63 [43.3, 67.2]%** | `FACTS` §8 | ✅ **七个比例**全带 Wilson CI。⚠️ **F6 是 2026-11-04 才补上的**（编码表一直有这列，文档漏统计），写 §5.4 时别漏 |
| Fig. 1 / Fig. 2 / Table 4 / Table 5 | — | `paper/figures/`、`FACTS` §5 | ✅ 图齐，表待排版 |

### §6 Discussion（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 6.1 机制 | 五个维度逐条解释 | `FACTS` §7.1–7.5 | ✅ |
| **6.1 聚合的量化** | 逐图平均比平均 MSE 高 **1.90/2.20/1.51 dB**（全图），单块放大到 **3.14/3.47/2.77**；面积加权只贡献 **−0.04～+0.07**（P1 为负），**96%** 是 Jensen。**顺带复现 Keleş 2021 的 2.07 dB** | `FACTS` §7.6 + `results/aggregation_gap.csv` | ✅ **原稿引的 1.156 dB 口径不对，已换** |
| 6.1 挡质疑的对照 | 换库 **0.0090–0.0215 dB** vs 换 antialias **2.7242–3.8892 dB**，同列配对是 **127–432 倍**（×2 约 130、×4 约 300–430）。**别写"200 倍"**——没有任何一个同口径能得出 200 | `FACTS` §7.3 | ✅ |
| 6.2 最小报告清单 | **六条**：聚合 / 验证 batch / 采样窗口 / 通道与 shave / 退化算子（含 `antialias`）/ 种子与 best-of-N。**摘要里压成四条**（合并「通道+shave」与「采样+batch」），正文保持六条 | `FACTS` §8 的**七个**比例（F1 1/63、F2 0/63、F3 5/44、F4 22/63、F5 0/44、F6 0/63、F7 35/63）可直接当每条的证据；其中 **F2/F5/F6 是零**，F1 是 1.6%，四条"零或近零"最有力 | ⬜ 待写 |
| 6.3 对审稿流程 | "不要求重跑，只要求一行字" + 可复制模板 | 需自己写 | ⬜ 待写 |

### §7 Limitations（0.5 页）

| 条目 | 数据 | 出处 |
|---|---|---|
| 非 SOTA | 按 **RGB 口径**：KinD **20.87**（as reported, not re-run）vs 我们 P3 **18.51**，差 **2.36 dB**。**不写 Retinexformer 6.65 dB**（25.16 − 18.51）——那是 Y 减 RGB。**口径锚点**：我们的 identity **7.7733** = 文献 "Input Image" **7.77**（差 0.003），证明口径对齐 | `FACTS` §9.3 + §10-E15 |
| 仅 2 任务 3 模型 | — | — |
| 无真实退化 | 全部合成 bicubic | — |
| 验证集小 | DIV2K 100 张、LOL 15 对。**低光均值 95% CI 半宽 ±2.04 dB**（SR ×4 ±0.87、×2 ±0.93）→ 低光侧 <2 dB 的方法间差异不可分辨 | `FACTS` §7.7 |
| ×2 基线晚补 | 同一代码路径、同一数据 | `FACTS` §0 |
| 文献抽样只覆盖 arXiv | 43/63 venue 待核 → 开源倾向偏差 | `FACTS` §8 |
| 单卡单框架 | 跨环境只验过 CPU/GPU + torch 2.10/2.11 | `FACTS` §9.2 |
| **协议支配是有条件的** | 核心命题只在 SR 上成立（×4 4.98×、×2 3.64×），**低光 0.21×**（2.25 vs 10.74）——headroom 差一个量级。不写这条，等于让审稿人从表里自己发现反例 | `FACTS` §6 + E57 |
| ~~感知损失未做消融~~ | ✅ **已做**（2026-11-02）：on 27.61 / off 27.60，Δ −0.01 dB，锐度差 0.26%。**限制**：只测 0.006 一档，不能外推成"感知损失无用"（SRGAN 是 1:1 量级） | `FACTS` §7.9 |

### §8 Conclusion（0.25 页）

回到开场那个学生。**"协议不是脚注，是结论的一部分。"**

---

## 3. 图件现状

| 编号 | 文件 | 尺寸 | 状态 |
|---|---|---|---|
| **Fig. 1** | `paper/figures/fig1_protocol_matrix.pdf`（20 KB）+ `.png` | figsize 7.0×3.9 → **成品 5.83×3.82 in** | ✅ 定稿（2026-11-04 重出：P1 增益改为对齐值 −0.58 / −2.31，增益轴下界放到 −2.95 以容下标签）。6-panel 双栏；两 SR 面板共享增益轴，低光独立轴 |
| **Fig. 2** | `paper/figures/fig2_sampling_noise.pdf`（20 KB）+ `.png` | figsize 3.45×2.30 → **成品 3.43×2.26 in** | ✅ 定稿（2026-11-04 压宽：`bbox_inches="tight"` 的默认 `pad_inches=0.1` 曾把它撑到 3.61 in、超 IEEE 单栏，现设 0.01）。源数据 `paper/data/fig2_sampling_noise.csv`，脚本 `scripts/make_figure2.py` |

### 图注要点（写作时按此写，不在此处写成品）

**Fig. 1** 必须交代：
1. 上排是绝对 PSNR，下排是相对同协议基线的差（论文真正会报的那个数）。
2. **P1 是轴外标记**：上排它比其余协议高 15–19 dB，下排它是**两个负增益之一**（−0.58 / −2.31，另一个是 P6）；画在同一轴上会把 0.8 dB 的效应压平。数值直接印在标记旁，未隐藏。
3. 三个任务各有自己的基线：SR 是 bicubic，低光是 identity（"什么都不做"）。
4. 误差棒是 5 个种子的样本标准差。**两个 SR 面板上几乎看不见**（σ 0.019–0.094 dB，而纵轴跨 4–5 dB），**低光面板上看得见**（P1 σ=0.41、P2 σ=0.53，轴跨 13 dB）。所以措辞要分开：SR 侧"种子方差小于标记尺寸、不是不确定性来源"；低光侧"采样协议的 σ 已经在图上可见"。**别笼统写"误差棒比标记还细"**——低光那三个点不成立。
5. 低光只有 P1–P5：配对任务没有"退化算子"这一轴。

**Fig. 2** 必须交代：
1. 数据来自**单张 444×680 真实图**的 28 个不相交 96² 块的 bootstrap（5000 次），不是数据集级置信区间。
2. 噪声按 1/√n 下降（灰线只过 n=1 一点，不做最小二乘，否则会掩盖系统性偏离）。
3. 单块采样带 **1.597 dB** 噪声，是文献常见 +0.3 dB 声称的 **约 5 倍**（1.597/0.3 = **5.32**，`FACTS` §7.4.2）。⚠️ **"+0.3 dB"是泛指量级、不是某篇的具体数字**，所以只能写"约 5 倍"并带"若某篇声称 +0.3 dB"这个前提；**别写成"文献都只涨 0.3 dB"**——那是把泛指当统计。
4. 要把噪声压到 0.3 dB 以下，**每图需要 28 次抽样**（n=16 时还有 0.395 dB，n=28 才是 0.293 dB）——"每图 1 块 + 多图平均"与"每图多块"在同等预算下不等价。⚠️ **n 是"每图抽样次数"，不是"用了多少块"**：这是**有放回** bootstrap（`σ_pop/√n`），28 次仍有 0.293 dB；而 28 恰等于该图的全部块数，**极易被读成"取全部就精确了"**（`FACTS` §7.4.2 限定 1）。
5. **不要从这个数外推"100 张就没事"**：1.60 dB 是**同一张图内** 28 块之间的 σ。若每图各取一块、取 100 张的均值，噪声按 1/√100 约降到 0.16 dB，但那是**图内 σ 的下界估计**，没算图间差异（DIV2K 逐图 PSNR 极差 23.10 dB，`FACTS` §7.7）。本文用全图评测，**根本不做抽样**，所以不受这条曲线约束——这才是真正的论据，不是"图多了就没事"。

---

## 4. 参考文献清单

### 4.1 协议审计（七篇最近邻，来源：`查新报告.md`）

| # | 条目 | 用途 | 核验 |
|---|---|---|---|
| 1 | Keleş, Yılmaz, Tekalp, **Korkmaz, Dogan**. *On the Computation of PSNR for a Set of Images or Video.* **PCS 2021, 页 1–5**（DOI `10.1109/PCS50896.2021.9477470`）. arXiv:2104.14868（2021-04-30） | 唯一正对"聚合方式"，SISR 实测 2.07 dB、理论上界 2.507 dB | ✅ arXiv API 已核：**是 5 位作者，不是 3 位**（原表漏了 Korkmaz 与 Dogan）；`<arxiv:comment>` 写明 "accepted for publication in PCS 2021"。**页码/DOI 已补齐**（2026-11-04 Crossref：`2021 Picture Coding Symposium (PCS)` 1–5） |
| 2 | Thirumani Srinivasan. *What Does LOL Measure? Hidden Label Shifts in Low-Light Enhancement Benchmarks.* arXiv:2610.38121（2026-10-28） | 低光侧标签偏移：LOLv1 测试标签渲染不同（同一批图在 LOLv2 中饱和度低 1.48×）、LOLv2-Real 测试集是 690–789 连续块且比训练标签暗 14%、test-time 设置足以改变排名 | ✅ arXiv API 已核（**单作者**）。⚠️ **原表"15 图 95% CI 跨 5.2 dB"不可核，已弃用**（摘要无此数，见 `FACTS` §10-E24）；只引上面三条 |
| 3 | Parmar, Zhang, Zhu. *On Aliased Resizing and Surprising Subtleties in GAN Evaluation*（Clean-FID）. CVPR 2022 | 抗锯齿 resize 的叙事先例（GAN 侧） | ✅ 标题/作者/会议确 |
| 4 | Shah, Duarte. *HyperBench: Standardizing and Scaling Synthetic Evaluation for Hyperspectral Super-Resolution.* arXiv:2605.21671（2026-05-20） | 高光谱 Wald 协议实现差异导致排名不保留 | ✅ arXiv API 已核（原表只有编号；**标题是全名，勿简写**） |
| 5 | Zhang, Li, Chen, Qiao, Wu, Dong. **SEAL: *A Framework for Systematic Evaluation of Real-World Super-Resolution.* arXiv:2309.03020（ICLR 2024 Spotlight）** | 随机采样评测不可靠、排名互换 | ✅ **原表 "2409.xxxx 编号不全" 已补全**：是 **2309**.03020，2023-09 挂出、ICLR 2024 Spotlight、XPixelGroup |
| 6 | Su, Rocafort, Xue et al. ***Bridging the Perception Gap in Image Super-Resolution Evaluation.*** arXiv:2503.13074（2025-03-17） | 审计感知质量评价，与本文的保真协议正交 | ✅ **原表标题 "Rethinking Image Evaluation in Super-Resolution" 是错的**，已按 arXiv 原文改正 |
| 7 | Cawley & Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* **JMLR 11(Jul):2079−2107, 2010** | 模型选择偏差的理论支撑 | ✅ 卷期页码已核（jmlr.org/papers/v11/cawley10a） |

> **本表 2026-11-04 用 arXiv API 逐条核过**：1/2/4/5/6 的标题与作者以 `export.arxiv.org/api/query?id_list=` 返回的 `<title>`/`<name>` 为准。原表 5 条"待核"、1 条编号残缺、1 条标题错误，现全部结清。**原缺的 PCS 页码已于 2026-11-04 补齐**（Crossref 实查：Keleş PCS 2021 页 **1–5**，DOI `10.1109/PCS50896.2021.9477470`）——至此七篇的 venue/页码/DOI 全部落实。

### 4.2 建议补充（**查新报告未覆盖，但 §2.3 需要**）

> **2026-11-04 第三次核：这次读的是摘要，不是元数据。** 上一轮只核了标题/作者/venue，五条都标"✅ 已核"，但**"引它来支撑什么"从来没对过摘要**。读完发现自己写的用途有两条是错的（Dodge、Bouthillier），一条过强（Musgrave），见 `FACTS` §10-E51。下表"摘要实际说了什么"一列**只能从 `<summary>` 里取**，不要按标题想象。

| 条目 | 摘要实际说了什么（据 arXiv `<summary>`） | 本文怎么用（含禁区） |
|---|---|---|
| Lucic, Kurach, Michalski, Gelly, Bousquet. *Are GANs Created Equal? A Large-Scale Study.* NeurIPS 2018（arXiv:1711.10337） | "most models can reach similar scores with enough hyperparameter optimization and random restarts… improvements can arise from a higher computational budget and tuning more than fundamental algorithmic changes"；结论还有更硬的一句："**we did not find evidence that any of the tested algorithms consistently outperforms the non-saturating GAN**" | §2.3 的"同等预算下改进消失"。**可用**：它是"预算→改进"这条线的源头。**禁区**：它动的是**训练预算**（超参+重启），我们动的是**评测**，不要写成"我们也证明了 GAN 那件事"。那句话里的"非饱和 GAN 都没被稳定超过"可以引，比"改进消失"更狠 |
| Musgrave, Belongie, Lim. *A Metric Learning Reality Check.* ECCV 2020（arXiv:2003.08505） | "papers from the past four years have consistently claimed great advances… often more than doubling"；复查后发现"flaws in the experimental methodology"，实际改进"**marginal at best**" | §2.3。⚠️ **"归零"过头了**——原文是 "marginal at best"（**至多是边际的**），不是零。写"归零"会被读过原文的审稿人抓。正确措辞："**多年报告的翻倍级改进，在统一实验方法后收缩到边际水平**"。核验：Crossref `10.1007/978-3-030-58595-2_41`（LNCS 2020），页 **681–699**，DBLP `conf/eccv/MusgraveBL20` |
| Dodge, Gururangan, Card, Schwartz, Smith. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP-IJCNLP 2019（arXiv:1909.03004） | "test-set performance scores alone are insufficient for drawing accurate conclusions about which model performs best"；提出"**expected validation performance of the best-found model as a function of computation budget**"；并发现"multiple recent model comparisons where authors would have reached a different conclusion if they had used more (or less) computation" | §2.3 + §6.2。⚠️ **上一版写的"报分布而非单点"是错配**——它讲的不是"报分布"，是"**报预算**"：你那个 best 是用多少搜索预算换来的，必须一起报。这条**直接支撑我们的 best-of-N**：§7.8 的 1.74 dB 就是"没报 N 的 best"；清单里"写明 N"那一项由此而来。核验：Crossref `10.18653/v1/d19-1224`，正式写法 **EMNLP-IJCNLP 2019**（container 含 "and the 9th IJCNLP"），**页 2185–2194**（2026-11-04 补） |
| Bouthillier, Delaunay, Bronzi, Trofimov, Nichyporuk et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021（arXiv:2103.03098） | 建模整个 benchmark 过程，"variance due to **data sampling, parameter initialization and hyperparameter choice** impact markedly the results"；反直觉结论："adding more sources of variation to an imperfect estimator approaches better the ideal estimator at a **51 times reduction in compute cost**"；并研究"the error rate of detecting improvements" | §2.3。⚠️ **必须写清分界**：他们分解的是**训练侧**方差（数据采样/初始化/超参），我们量化的是**评测协议**方差（权重固定、种子固定）。**两者正交**，合起来才是完整图景——这句分界正是本文的位置所在。**禁区**：不要写"我们也做了方差分解"（会让人以为重复劳动）。作者数：proceedings **16 位** vs arXiv **17 位**，`refs.bib` 用 proceedings 版（`AUTHOR_OVERRIDE`）。**MLSys 2021 论文集不排连续页码**（2026-11-04 实查 proceedings.mlsys.org：卷 *Proceedings of Machine Learning and Systems 3*，按论文独立编号），所以这条**没有页码可补，不要再挂缺口** |
| Picard. *torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision.* arXiv:2109.08203（2021-09-16） | "even if the variance is not very large, it is **surprisingly easy to find an outlier** that performs much better or much worse than the average"；扫了 CIFAR-10 上多达 10⁴ 个种子 | §2.3 + §7.8。**这条比上一版写的更有用**：它说的正是"方差不大，但极值好找"——与我们 best-of-60 抬 1.74 dB 同一机制（我们 σ 只有 0.019–0.094 dB）。⚠️ 它是**分类 accuracy**，我们是 PSNR，类比要注明。**纯预印本、无正式发表**；标题是全名（"Torch." 的首字母大写是 arXiv 自动加的，原题小写），引用勿简写 |

### 4.3 模型与数据

| 条目 | 用途 |
|---|---|
| Dong et al. SRCNN, ECCV 2014 | ×2 模型 |
| Ledig et al. SRResNet/SRGAN, CVPR 2017 | ×4 模型 + 感知损失出处（VGG54） |
| Lim et al. EDSR, **CVPRW 2017**（arXiv comment 原文 "To appear in CVPR 2017 workshop"）；Zhang et al. RCAN, ECCV 2018 | §2.1 谱系 |
| **Cai** et al. Retinexformer, ICCV 2023（arXiv:2303.06705） | §2.1 + Limitations 的对比值 |
| Zhang et al. KinD, ACM MM 2019 | §2.1 + Limitations 的对比值 |
| **Wei, Wang, Yang, Liu. *Deep Retinex Decomposition for Low-Light Enhancement.* BMVC 2018** | **一条顶两条**：既提出 RetinexNet，又发布 LOL 数据集（原表把它拆成 "RetinexNet, BMVC 2018" + "LOL dataset, **ICCV** 2018" 两行，后者年份/会议是错的） |
| Agustsson & Timofte. NTIRE 2017 Challenge / DIV2K | 数据集（**只引数据集**）。⚠️ **不要再写"文献惯例 = Y + shave"**：Y 有 22/22 支持（`FACTS` §8），**shave 是 SR 子集 44 篇里 0 篇说明**（5 篇 LLIE 属不适用，不该进分母），没有可引出处。见 `FACTS` §10-E29 |
| Wang et al. SSIM, IEEE TIP 2004 | 指标 |
| Johnson et al. Perceptual Losses, ECCV 2016；Simonyan & Zisserman VGG, ICLR 2015 | 感知损失 |
| Blau & Michaeli. *The Perception-Distortion Tradeoff.* CVPR 2018 | 感知-失真权衡的经典出处。⚠️ **不能用来解释本文的 PSNR**——本文 `w_percep=0.006` 实测只值 −0.01 dB（`FACTS` §7.9），权衡在这一档没有发生。引它只能是 §2.1 的背景 |
| **Kingma & Ba. Adam, ICLR 2015**（arXiv:1412.6980） | **优化器出处**——§4.2 明写三个任务都用 `torch.optim.Adam` 默认超参，必须引。此前 bib 有、本表漏登 |
| **He et al. Deep Residual Learning, CVPR 2016**（arXiv:1512.03385） | **残差结构出处**——×4 模型的 8 个 `res_blocks` 是 ResNet 式残差块，§2.1 谱系与 §4.1 结构描述都要用到。此前 bib 有、本表漏登 |
| **Lai et al. LapSRN, TPAMI 2019**（arXiv:1710.01992，期刊版） | **金字塔式上采样谱系**——与本文的亚像素上采样（`upsample`）是同一问题的两种解法，§2.1 谱系一句带过。⚠️ **venue 已修正**：arXiv 这版是 TPAMI 2019 的期刊版（标题含 "Fast"），CVPR 2017 那篇叫 "Deep LapSRN"（`10.1109/cvpr.2017.618`），别混（`FACTS` §10-E38） |

> **已完成**：§4.1 七条 + §4.2 五条 + §4.3 十五条 = **27 条**，与 `paper/refs.bib` 的 27 个 key 一一对应（2026-11-04 补登 Adam / ResNet / LapSRN 三条——bib 里一直有，本表漏登，导致"27 条"这个数在表里数不出来）（`scripts/build_refs_bib.py` 从 arXiv API 抓取，标题不符即中止）。原缺的 Keleş PCS 页码已补（1–5，DOI `10.1109/PCS50896.2021.9477470`）；Dodge 页 2185–2194 已补；Bouthillier（MLSys）**无连续页码**，实查确认，不再等。
> **NTIRE 官方协议页不再是缺口**——见 §5 第 5 项（原设想用官方文档支撑"Y+shave 惯例"，但 shave 无人写，官方文档也证明不了）。
>
> **编号纪律**：本次复核中，凭记忆写出的 KinD `1909.01917` 与 RetinexNet `1808.10082` **都指向了完全无关的论文**（前者是差分隐私 SQL，后者是隐私保护检测）。正确值是 **1905.04161 / 1808.04560**，按标题检索得到。`refs.bib` 里每个 arXiv 编号都必须过一遍 `id_list=` 校验，`<title>` 与预期不符就退回重查。

---

## 5. 缺口（按阻塞程度排序）

| # | 缺口 | 阻塞什么 | 成本 | 状态 |
|---|---|---|---|---|
| **1** | **作者署名、单位、通讯邮箱** | **一切对外动作**（预印本注册、模板页眉、署名顺序） | 等用户 | ⬜ |
| ~~2~~ | ~~`refs.bib`~~ | — | **已生成 `paper/refs.bib`：27 条**（20 inproceedings + 4 misc + 3 article），由 `scripts/build_refs_bib.py` 从 arXiv API 抓取并**逐条校验标题**，手改无效 | ✅ **2026-11-04 完成** |
| **3** | Table 1（七篇对比） | §2 | 从查新报告改写；7 篇出处已核 | ⬜ **出处已清** |
| ~~4~~ | ~~§2.3 的 ML 评测谱系五篇~~ | — | **五篇摘要已读**（2026-11-04）：用途改了两条（Dodge 是"报预算"不是"报分布"、Bouthillier 要写"训练方差 vs 评测方差"的分界），Musgrave 的"归零"改为"marginal at best"。`OUTLINE` 已补 §2.3 | ✅ **完成**，见 `FACTS` §10-E51 |
| ~~5~~ | ~~NTIRE 官方协议原文~~ | **降级为可选**：原想用它支撑"文献惯例 = Y + shave"，但 shave 部分是 **SR 子集 0/44 说明**（`FACTS` §8-F5），**任何官方文档都不能把"没人写的东西"变成惯例**。改用抽样数据自证，见 §10-E29 | 0 | ✅ **关闭** |
| **6** | 六条最小报告清单的措辞 | §6.2 | 写作时提炼 | ⬜ |
| 7 | Table 2–6 排版 | §3–§6 | 素材齐，纯排版 | ⬜ |
| 8 | 页数/模板确定 | 排版 | 等 Preprints.org 格式 + 署名 | ⬜ |
| **9** | ~~Fig. 2 压到 3.4 in 以内~~ | — | 已由 `pad_inches=0.01` 解决，成品 3.43 in | ✅ **已解决 2026-11-04** |
| ~~10~~ | ~~感知损失"1:8–1:17"~~ | — | 正文只报权重即可（`FACTS` §10-E18）；若要比例，需重跑记 loss 分量 | ⚪ **可删，不再阻塞** |

**不阻塞的**：所有数字、两张图的源数据、三张表的源数据、**27 条参考文献（已成 bib）**。

---

## 6. 现在能写到什么程度

| 章节 | 可写性 | 说明 |
|---|---|---|
| §3 Protocol Matrix | **可立即写** | 定义齐、机制齐、加速比齐，不依赖任何外部输入 |
| §5 Results | **可立即写** | 21 行增益表 + 两张图 + Wilson CI 全齐 |
| §4 Setup | **可立即写** | 模型/数据/训练/环境全齐，含交叉验证 |
| §6 Discussion | 大部分可写 | 机制段齐；6.2/6.3 需提炼措辞 |
| §7 Limitations | 可立即写 | **九条**全部有数据支撑 |
| §8 Conclusion | 依赖 §1 | — |
| §2 Related Work | **可写** | bib 已生成（27 条），七篇对比表出处已核（只差改写成论文表）；**2.3 五篇的写法已定**（`OUTLINE` §2.3），摘要已读、用途已改 |
| §1 + Abstract | **最后写** | 需先知道全文说了什么 |

> 按此顺序：**§3 → §5 → §4 → §6/§7 → §2（等 bib）→ §1/Abstract**。
> 但用户指示先整理素材不写正文，故本文件到此为止；下一步等指令。

---

## 7. 已废弃的素材源

`/root/.codebuddy/artifact/2f8a7d30-9c21-4f0b-1d22-8f607b933000/pf_paper/` 下的 8 个 README（01-引言素材 … 07-可复现性素材）与 `00-索引.md` **已全部过时，不要再引用**：

| 那里写的 | 现状 |
|---|---|
| "待补测 #6 协议矩阵 P1–P6 全量" | ✅ 已完成，8 协议 × 3 任务 × 5 种子 |
| "待补测 #7 种子矩阵" | ✅ 已完成 |
| "待补测 #8 文献查新" | ✅ 已完成（N=63） |
| "39 项测试" | 现为 **81 项、全部通过**（`train/tests` 69、`tests/` 12），另 1 个 e2e 模块默认跳过。见 §4.1 |
| "×2 槽位是 SRResNet" | 部分是：**主仓 `models/` 的 0.781 M 部署权重**是 SRResNet（781,068 可训练参数；写 0.783 是把 state_dict 元素数当参数量）；**种子矩阵 `models/seeds/srcnn_x2/` 的 0.069M** 是真 SRCNN。论文用后者。见 `FACTS.md` §10-E1 |
| 五类维度的"11.85 dB / 11.65 dB" | 条件是 32 个 96² 块，与论文的全图口径不同，**禁止混用**。见 `FACTS.md` §11 |

那份 `00-审查总账.md`（1631 行）**仍有价值**，但只作为过程记录与自我更正档案引用，不作为数字来源。

