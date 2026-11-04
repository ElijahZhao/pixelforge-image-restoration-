# 论文素材总账（MATERIALS）

> **2026-11-04。** 这份文件回答一个问题：**每一节要论证什么、用哪个数字、那个数字在哪、还缺什么。**
> 三份文件的分工：`MATERIALS.md`（本节要什么）→ `FACTS.md`（数字是多少、出处）→ `OUTLINE.md`（怎么组织）。
>
> 纪律：**写作时不许翻代码、不许翻对话记录。** 本文件指向哪里就去哪里取；取不到就是缺口，记在 §5。

---

## 1. 定位：写什么，不写什么

**写的是**：一篇评测方法学短文。核心命题——*在轻量图像修复里，评测协议的选择能移动报告指标 3.9 dB（×4 全协议跨度 19.8 dB），是模型自身改进（0.78 dB）的 5 倍；只改一个 `antialias` 布尔值，结论从"优于基线 +0.82 dB"翻成"劣于基线 −1.75 dB"。*

> 数字口径：3.9 / 19.8 / 0.78 都是 **×4 任务**的值（×2 是 3.14 / 15.65 / 0.86，低光是 2.25 / 6.07 / 10.74）。摘要里若只报一组，必须标明是 ×4，不能让人读成"三任务平均"。

**不写的是**：
- 不主张性能。三个模型都不是 SOTA，低光落后 KinD 约 2.5 dB（文献报告值，未复跑）。
- 不主张系统。81 项测试（+1 skip）、CI、跨环境逐字节一致是**证据可信的前提**，放 §4.1，不单列为贡献。
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
| 建议 | 四条最小报告要求 | 需自己提炼（§6.2） | ⬜ 待写 |

### §1 Introduction（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 开场场景 | 学生把 bicubic 跑出 28.9、模型 29.0，报告 +0.1 dB；换脚本变 −1.7 | 构造场景，可基于真实数据改写成"27.61 vs 26.83 = +0.78，换算子后 −1.75" | ✅ 数据支撑 |
| 一图证明 | Fig. 1 | `paper/figures/fig1_protocol_matrix.pdf` | ✅ |
| 三条贡献 | 见 §1 | — | ✅ |

> 开场不要写"图像修复很重要"。写那个学生——他没做错任何事，只是没说清三行默认参数。

### §2 Related Work（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 2.1 轻量 SR / LLIE 一条线 | SRCNN→SRResNet→EDSR→RCAN；RetinexNet→KinD→Retinexformer | 文献（见 §4） | ⬜ 需建 bib |
| 2.2 协议审计七篇 | 七篇"它做了什么 / 我们补什么" | `archive_private/05-论文规划/查新报告.md` §二 + 本文件 §4.1 | ✅ 表已成，**出处已核完** |
| **2.3 ML 评测批判谱系** | **查新报告缺这条线**，见下 | 本文件 §4.2（五篇出处已核） | ⬜ 待写（素材齐） |
| Table 1 | 最近邻对比 | 查新报告改写 | ⬜ |

> **本节最大的增值点**：查新报告的七篇全是"图像修复领域内"的占位。补上"ML 评测方差"这条线（Lucic 2018 / Musgrave 2020 / Dodge 2019 / Bouthillier 2021 / Picard 2021），论文就从"一个领域的经验观察"升级为"接入 ML 评测批判的既有谱系"。**0 实验成本，只多读五篇。**

### §3 Protocol Matrix（1.5 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 3.1 五类维度 | ①聚合 ②验证 batch ③采样位置与窗口 ④通道与边界 ⑤退化算子 | `FACTS` §4 + §7 | ✅ |
| 每维"为什么原则上会改变数字" | 一句机制，不展开（展开在 §6.1） | `FACTS` §7.1–7.5 | ✅ |
| 3.2 八个协议 | Table 2，逐列 `degrade/metric/sampling/agg` | `FACTS` §4 表 | ✅ 定义齐 |
| **P5 与 P6 只差一个布尔值** | 同库同函数同尺寸 | `FACTS` §4 注 | ✅ 全文最关键一行 |
| 3.3 分组实现 | **按退化算子分 3 组**（`tv_bicubic_aa` / `tv_bicubic_noaa` / `pil_bicubic`）→ **3 次全图前向**，P1 的 60 块与 P2 的单块从同一张输出图上裁；T4 上 x4 全图前向 **0.2683 s / 峰值 3.15 GB**（x2 是 0.6624 s / 8.37 GB） | 探针 `archive_private/06-实验脚本与输出/kag-probe/probe_v1.json` | ✅ |
| **66× 的正确口径** | 66× 是 **T4 相对 CPU 的全图前向加速比**，不是"分组省下的倍数"。分组复用本身是 **8 协议 → 3 次前向（约 2.7×）** | `results/compute_ledger.md` §六（"4.2 h 拍脑袋估算 → 0.7 h 实测"）+ `archive_private/06-实验脚本与输出/README.md` | ⚠️ **极易误用，见 FACTS §10-E17** |

### §4 Setup（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 4.1 三个模型 | 1.226M / 0.069M / 0.565M（实测） | `FACTS` §1 | ✅ |
| **×4 带感知损失** | 硬事实（config 实测）：`w_pixel=1.0`、`w_percep=0.006`、`objective = 1.0*charbonnier + 0.006*vgg_perceptual`；**"有效贡献 1:8–1:17" 是估算，不是实测**（日志只记 total loss，未记分量） | `FACTS` §3 + §10-E18 | ✅ **必写**，但比例只能当估算说 |
| 4.1 可复现性 | **81 项通过 + 1 跳过**（`train/tests` 69 + `tests/` 12：API 冒烟 9、推理集成 3）；CI 三门禁（pytest+覆盖率 75%、pip-audit、tsc+next build）；跨环境逐字节 | `FACTS` §9.2 + §10-E10 | ✅ |
| 4.2 数据 | DIV2K 800/100；LOL 485/15；低光**配对无上采样** | `FACTS` §2 | ✅ |
| 4.2 种子与轮数 | 42–46；200/100/200 轮 | `FACTS` §3 | ✅ |
| 4.3 环境 | T4 训练、本地 CPU 跑 ×2 基线、跨环境 <1e-6 dB | `FACTS` §9.2 | ✅ |
| **验证口径交叉验证** | 训练日志 best 对到协议矩阵 1e-3 内，SR 对 P2、低光对 P3 | `FACTS` §9.1 | ✅ 极强的可信度证据 |

### §5 Results（2.5 页，核心）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 5.1 方差分解 | 种子 std（**仅全图协议 P3–P8**）：×4 0.0190–0.0361 / ×2 0.0268–0.0938 / 低光 0.0346–0.0806 | `FACTS` §6 + §5.1 | ✅ |
| 5.1 **采样协议的种子 std 不可混入上一条** | 低光 P1 = **0.4148**、P2 = **0.5303**；×4 P1 = 0.2364、×2 P1 = 0.5584 —— 这些不是"训练随机性"，是**块采样噪声被 best-of-60 放大**（Fig. 2 的同一现象） | `FACTS` §5.1 | ✅ 写 §5.1 时必须分开说，否则表里 0.53 会打自己的脸 |
| 5.1 跨度 | 3.889（×4）/ 3.143（×2）/ 2.250（低光）；比值 4.98× / 3.64× | `FACTS` §6 | ✅ |
| 5.1 最低协议不同 | ×4 最低 P6，×2 最低 P2 | `FACTS` §6 注 | ✅ |
| 5.2 符号翻转 | 完整 21 行增益表 | `FACTS` §5.3 | ✅ |
| 5.2 机制数字 | 模型侧 −3.89/−2.72，基线侧 −1.32/−0.31 | `FACTS` §7.2 | ✅ |
| 5.3 跨任务 | ×2 复现且更大（3.03 dB 摆幅） | `FACTS` §5.3 | ✅ |
| 5.3 **P1 不是"另一个翻转"，是"P1 根本不可比"** | ×4 **−0.5793**、×2 **−2.3123**（两边同取 95 张）。5 张 DIV2K 图（`0839/0843/0846/0847/0868`）被 bicubic 在 best-of-60 里**精确重建**，PSNR=∞，故基线 n=95；模型侧对应 5 张是 50–68 dB。**任何网络都不可能赢** | `FACTS` §5.3 + §10-E21 + `results/p1_inf_alignment.csv` | ✅ 2026-11-04 修正 |
| 5.4 文献报告 | **F1 聚合 1/63 [0.3, 8.5]%；F2 验证 batch 0/63 [0.0, 5.7]%；F3 退化可复现 5/44 [5.0, 24.0]%；F4 明说 Y 通道 22/63 [24.3, 47.2]%；F7 给代码 35/63 [43.3, 67.2]%** | `FACTS` §8 | ✅ 五个比例全带 Wilson CI |
| Fig. 1 / Fig. 2 / Table 4 / Table 5 | — | `paper/figures/`、`FACTS` §5 | ✅ 图齐，表待排版 |

### §6 Discussion（1 页）

| 要写 | 用什么 | 出处 | 状态 |
|---|---|---|---|
| 6.1 机制 | 四个维度逐条解释 | `FACTS` §7.1–7.5 | ✅ |
| 6.1 挡质疑的对照 | 换库 0.01–0.02 dB vs 换 antialias 2.7–3.9 dB，差 200 倍 | `FACTS` §7.3 | ✅ |
| 6.2 最小报告清单 | 六条（聚合/batch/采样窗口/通道与 shave/退化算子含 antialias/种子与 best-of-N） | 需自己提炼 | ⬜ 待写 |
| 6.3 对审稿流程 | "不要求重跑，只要求一行字" + 可复制模板 | 需自己写 | ⬜ 待写 |

### §7 Limitations（0.5 页）

| 条目 | 数据 | 出处 |
|---|---|---|
| 非 SOTA | 低光落后 KinD ~2.5 dB、Retinexformer ~6.8 dB（**as reported, not re-run**） | `FACTS` §10-E15 |
| 仅 2 任务 3 模型 | — | — |
| 无真实退化 | 全部合成 bicubic | — |
| 验证集小 | DIV2K 100 张、LOL 15 对；15 张的 95% CI 跨度约 5 dB | `FACTS` §10-E16 |
| ×2 基线晚补 | 同一代码路径、同一数据 | `FACTS` §0 |
| 文献抽样只覆盖 arXiv | 43/63 venue 待核 → 开源倾向偏差 | `FACTS` §8 |
| 单卡单框架 | 跨环境只验过 CPU/GPU + torch 2.10/2.11 | `FACTS` §9.2 |
| **感知损失未做消融** | ×4 带 `w_percep=0.006`，无成对消融 | `FACTS` §3 |

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
2. **P1 是轴外标记**：上排它比其余协议高 15–19 dB，下排它是**唯一的两个负增益之一**（−0.58 / −2.31，另一个是 P6）；画在同一轴上会把 0.8 dB 的效应压平。数值直接印在标记旁，未隐藏。
3. 三个任务各有自己的基线：SR 是 bicubic，低光是 identity（"什么都不做"）。
4. 误差棒是 5 个种子的样本标准差——**比标记还细**，说明种子方差不是这里的不确定性来源。
5. 低光只有 P1–P5：配对任务没有"退化算子"这一轴。

**Fig. 2** 必须交代：
1. 数据来自**单张 444×680 真实图**的 28 个不相交 96² 块的 bootstrap（5000 次），不是数据集级置信区间。
2. 噪声按 1/√n 下降（灰线只过 n=1 一点，不做最小二乘，否则会掩盖系统性偏离）。
3. 单块采样带 **1.597 dB** 噪声，是文献常见 +0.3 dB 声称的 **5.3 倍**。
4. 要把噪声压到 0.3 dB 以下，**每图需要 28 块**（n=16 时还有 0.395 dB，n=28 才是 0.293 dB）——"每图 1 块 + 多图平均"与"每图多块"在同等预算下不等价。
5. 100 张图的均值会把噪声再压 10 倍（1.60 → 0.16 dB）——这正是我们用全图而非依赖这条曲线的原因。

---

## 4. 参考文献清单

### 4.1 协议审计（七篇最近邻，来源：`查新报告.md`）

| # | 条目 | 用途 | 核验 |
|---|---|---|---|
| 1 | Keleş, Yılmaz, Tekalp. *On the Computation of PSNR for a Set of Images or Video.* PCS 2021. arXiv:2104.14868（2021-04-30） | 唯一正对"聚合方式"，SISR 实测 2.07 dB、理论上界 2.507 dB | ✅ arXiv API 已核（标题/三位作者/日期全对）；页码待排版时补 |
| 2 | Thirumani Srinivasan. *What Does LOL Measure? Hidden Label Shifts in Low-Light Enhancement Benchmarks.* arXiv:2610.38121（2026-10-28） | 低光侧标签偏移/测试时设置，luma vs RGB 差 1.9–3.7 dB；15 图 95% CI 跨 5.2 dB | ✅ arXiv API 已核（**单作者**，原表"待核作者"已补） |
| 3 | Parmar, Zhang, Zhu. *On Aliased Resizing and Surprising Subtleties in GAN Evaluation*（Clean-FID）. CVPR 2022 | 抗锯齿 resize 的叙事先例（GAN 侧） | ✅ 标题/作者/会议确 |
| 4 | Shah, Duarte. *HyperBench: Standardizing and Scaling Synthetic Evaluation for Hyperspectral Super-Resolution.* arXiv:2605.21671（2026-05-20） | 高光谱 Wald 协议实现差异导致排名不保留 | ✅ arXiv API 已核（原表只有编号；**标题是全名，勿简写**） |
| 5 | Zhang, Li, Chen, Qiao, Wu, Dong. **SEAL: *A Framework for Systematic Evaluation of Real-World Super-Resolution.* arXiv:2309.03020（ICLR 2024 Spotlight）** | 随机采样评测不可靠、排名互换 | ✅ **原表 "2409.xxxx 编号不全" 已补全**：是 **2309**.03020，2023-09 挂出、ICLR 2024 Spotlight、XPixelGroup |
| 6 | Su, Rocafort, Xue et al. ***Bridging the Perception Gap in Image Super-Resolution Evaluation.*** arXiv:2503.13074（2025-03-17） | 审计感知质量评价，与本文的保真协议正交 | ✅ **原表标题 "Rethinking Image Evaluation in Super-Resolution" 是错的**，已按 arXiv 原文改正 |
| 7 | Cawley & Talbot. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* **JMLR 11(Jul):2079−2107, 2010** | 模型选择偏差的理论支撑 | ✅ 卷期页码已核（jmlr.org/papers/v11/cawley10a） |

> **本表 2026-11-04 用 arXiv API 逐条核过**：1/2/4/5/6 的标题与作者以 `export.arxiv.org/api/query?id_list=` 返回的 `<title>`/`<name>` 为准。原表 5 条"待核"、1 条编号残缺、1 条标题错误，现全部结清。**只剩 1 的 PCS 页码**（会议论文集，arXiv 版无页码）留到排版时补。

### 4.2 建议补充（**查新报告未覆盖，但 §2.3 需要**）

| 条目 | 为什么必须引 | 核验 |
|---|---|---|
| Lucic, Kurach, Michalski, Gelly, Bousquet. *Are GANs Created Equal? A Large-Scale Study.* NeurIPS 2018（arXiv:1711.10337） | "同等调参预算下，改进消失"——本文符号翻转的最近亲 | ✅ 已核 |
| Musgrave, Belongie, Lim. *A Metric Learning Reality Check.* ECCV 2020（arXiv:2003.08505） | 公平基线后，多年"改进"归零。**最能给本文定位的一篇** | ✅ 已核 |
| Dodge, Gururangan, Card, Schwartz, Smith. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP-IJCNLP 2019（arXiv:1909.03004） | 报告规范（报分布而非单点）——直接支撑 §6.2 清单 | ✅ 已核 |
| Bouthillier, Delaunay, Bronzi, Trofimov, Nichyporuk et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021（arXiv:2103.03098） | 方差来源分解的方法论 | ✅ 已核 |
| Picard. *torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision.* arXiv:2109.08203（2021-09-16） | 种子敏感性——本文种子 std 0.02–0.09 dB 的对照 | ✅ 已核（标题是全名，引用勿简写） |

### 4.3 模型与数据

| 条目 | 用途 |
|---|---|
| Dong et al. SRCNN, ECCV 2014 | ×2 模型 |
| Ledig et al. SRResNet/SRGAN, CVPR 2017 | ×4 模型 + 感知损失出处（VGG54） |
| Lim et al. EDSR, CVPR 2017；Zhang et al. RCAN, ECCV 2018 | §2.1 谱系 |
| **Cai** et al. Retinexformer, ICCV 2023（arXiv:2303.06705） | §2.1 + Limitations 的对比值 |
| Zhang et al. KinD, ACM MM 2019 | §2.1 + Limitations 的对比值 |
| **Wei, Wang, Yang, Liu. *Deep Retinex Decomposition for Low-Light Enhancement.* BMVC 2018** | **一条顶两条**：既提出 RetinexNet，又发布 LOL 数据集（原表把它拆成 "RetinexNet, BMVC 2018" + "LOL dataset, **ICCV** 2018" 两行，后者年份/会议是错的） |
| Agustsson & Timofte. NTIRE 2017 Challenge / DIV2K | 数据集 + **"文献惯例 = Y + shave"的官方出处** |
| Wang et al. SSIM, IEEE TIP 2004 | 指标 |
| Johnson et al. Perceptual Losses, ECCV 2016；Simonyan & Zisserman VGG, ICLR 2015 | 感知损失 |
| Blau & Michaeli. *The Perception-Distortion Tradeoff.* CVPR 2018 | **解释为什么带感知损失的模型 PSNR 偏低**——`FACTS` §3 那条必须写的设置，理论上靠这篇 |

> **待办**：§4.3 的 10 条仍需核 DOI/页码/作者全名（§4.1/§4.2 已核完）。之后统一生成 `paper/refs.bib`。NTIRE 官方协议页需抓存档（查新报告提到但未存链接）。

---

## 5. 缺口（按阻塞程度排序）

| # | 缺口 | 阻塞什么 | 成本 | 状态 |
|---|---|---|---|---|
| **1** | **作者署名、单位、通讯邮箱** | **一切对外动作**（预印本注册、模板页眉、署名顺序） | 等用户 | ⬜ |
| **2** | `refs.bib`（25 条） | §2 无法定稿 | **2 小时**——出处已全部核完（§4），只剩转 BibTeX + 补 DOI | ⬜ **成本已降** |
| **3** | Table 1（七篇对比） | §2 | 从查新报告改写；7 篇出处已核 | ⬜ **出处已清** |
| **4** | §2.3 的 ML 评测谱系五篇 | 论文的学术定位高度 | 读五篇摘要 | ⬜ **出处已核** |
| **5** | NTIRE 官方协议原文 | §2、§6 的"文献惯例"出处 | 需抓网页存档 | ⬜ |
| **6** | 六条最小报告清单的措辞 | §6.2 | 写作时提炼 | ⬜ |
| 7 | Table 2–6 排版 | §3–§6 | 素材齐，纯排版 | ⬜ |
| 8 | 页数/模板确定 | 排版 | 等 Preprints.org 格式 + 署名 | ⬜ |
| **9** | ~~Fig. 2 压到 3.4 in 以内~~ | — | 已由 `pad_inches=0.01` 解决，成品 3.43 in | ✅ **已解决 2026-11-04** |
| **10** | 感知损失"有效贡献 1:8–1:17"是估算 | 只在正文引比例时 | 重跑 s42 并记 loss 分量，或正文只报权重不报比例 | ⬜ **新增** |

**不阻塞的**：所有数字、两张图的源数据、三张表的源数据、25 条文献的出处。

---

## 6. 现在能写到什么程度

| 章节 | 可写性 | 说明 |
|---|---|---|
| §3 Protocol Matrix | **可立即写** | 定义齐、机制齐、加速比齐，不依赖任何外部输入 |
| §5 Results | **可立即写** | 21 行增益表 + 两张图 + Wilson CI 全齐 |
| §4 Setup | **可立即写** | 模型/数据/训练/环境全齐，含交叉验证 |
| §6 Discussion | 大部分可写 | 机制段齐；6.2/6.3 需提炼措辞 |
| §7 Limitations | 可立即写 | 八条全部有数据支撑 |
| §8 Conclusion | 依赖 §1 | — |
| §2 Related Work | **卡在 bib** | 表已成，**25 条出处全核完**，只剩转 BibTeX |
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
| "39 项测试" | 现为 **82 项收集：81 通过 + 1 跳过**（`train/tests` 69、`tests/` 12）。见 §4.1 |
| "×2 槽位是 SRResNet" | 部分是：**主仓 `models/` 的 0.783M 部署权重**是 SRResNet；**种子矩阵 `models/seeds/srcnn_x2/` 的 0.069M** 是真 SRCNN。论文用后者。见 `FACTS.md` §10-E1 |
| 五类维度的"11.85 dB / 11.65 dB" | 条件是 32 个 96² 块，与论文的全图口径不同，**禁止混用**。见 `FACTS.md` §11 |

那份 `00-审查总账.md`（1631 行）**仍有价值**，但只作为过程记录与自我更正档案引用，不作为数字来源。

