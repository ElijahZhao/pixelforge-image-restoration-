# Preprints.org 投稿元数据（2026-11-04）

> 本文件是**投稿表单的填写稿**，不是论文正文。字段全部经外部核实，来源逐条列在文末。
> 阻塞状态：**已解除**（此前 `FACTS` §12 / `MATERIALS` §5 第 1 项 / `OUTLINE` §6 第 6 项都记为"等用户"）。

---

## 1. 署名与身份

| 字段 | 填写值 | 核实依据 |
|---|---|---|
| 姓名 | **Yulin Zhao** | ORCID 记录的姓名（given-names = Yulin, family-names = Zhao） |
| 别名 | Elijah Zhao（"also known as"） | ORCID 记录的 "Also known as" 条目；SciProfiles 页显示为 "Mr. Elijah Zhao" |
| ORCID | `https://orcid.org/0009-0005-4623-8465` | 实查：该 ORCID 页面上方即显示此姓名与别名 |
| 通讯邮箱 | `20233407131@stu.huel.edu.cn` | 官方 Instructions 明写"应尽可能使用机构邮箱"；该邮箱是学生邮箱（`stu.huel.edu.cn`），而 ORCID 记录的 **verified email domain 正是 `huel.edu.cn`**——机构归属闭环 |
| 机构 | **Henan University of Economics and Law**（河南财经政法大学，缩写 HUEL），中国郑州 | 学校章程第一条明文："英文名称为 'Henan University of Economics and Law'，英文缩写 'HUEL'" |
| 学院 | **不填** | 用户决定（2026-11-04）。Preprints.org 要求 affiliations，但学校层面已满足 |
| 稿件类型 | **Article** | 官方列出的适合类型；本文是原创研究，非 review / conference paper |

⚠️ **一个必须避开的坑**：ORCID 记录里机构写成 **"Henan University of Economic and Law"**（少一个 s）。那是 ORCID 上的笔误，**与学校章程不符**。投稿一律用章程里的 **Economics and Law**。已同步写入 `CITATION.cff` 的注释。

### 1.1 已写入仓库的位置

`CITATION.cff` 已更新并通过 **CFF 1.2.0 官方校验器**（`cffconvert`）验证：

```bash
python3 -c "
from cffconvert import Citation
c = Citation(open('CITATION.cff').read()); c.validate(); print('PASS')
print(c.as_bibtex('bibtex'))"    # -> author = {Zhao, Yulin}
```

写入的字段：`given-names: Yulin` / `alias: Elijah Zhao` / `orcid` / `email` / `affiliation`（CFF 1.2.0 的 `person` 实体支持这五项，已核对 schema）。

---

## 2. Preprints.org 官方要求（逐条核实，2026-11-04）

来源：<https://www.preprints.org/instructions-for-authors>（`Instructions for Authors` 页原文）

### 2.1 首屏必须包含（原文）

> The first page of your manuscript must contain the title, author list, abstract, keywords, corresponding author contact details, and affiliations for all authors.

→ 对应 LaTeX 首屏六项：**标题 / 作者列表 / 摘要 / 关键词 / 通讯作者联系方式 / 全部作者单位**。

### 2.2 提交文件格式（原文 —— 与本文档旧记载相反，见 §3）

> Files should be submitted in Microsoft Word or LaTeX format. For LaTeX files, ensure that all files necessary, including the .bib file, if applicable, to recreate the PDF are included in a zip or similar archive.

→ **PDF 与 LaTeX 源码都收**。若走 LaTeX 路线，`.tex` + `refs.bib` + 图 + 全部 `.cls/.sty` 一起打包成 zip。
→ **本文的投稿包应包含 `paper/refs.bib`**（27 条，`scripts/build_refs_bib.py` 生成）。

### 2.3 图形摘要（推荐，非强制）

> we recommend producing a graphical abstract ... The graphic should be provided as a JPG or PNG file. Please note that a screenshot of the textual abstract does not qualify as a graphical abstract. Files submitted as graphical abstracts that do not meet the required criteria will be removed prior to the manuscript's posting.

→ **候选现成件**：`paper/figures/fig1_protocol_matrix.png`（1749×1144 px @ 300 dpi）。
→ ⚠️ **不能直接交 Fig. 1**：它是论文里的六面板矩阵图，不是为"一页读懂"设计的图形摘要；"文本摘要的截图"明确不算。**提交前需另做一张**——这是**待办**，不是已解决项。

### 2.4 数据可得性（原文，硬性）

> Preprints.org requires that authors make all data associated with their submission available where there are no legal or confidentiality constraints in doing so.

> Make sure research data are available.

→ **现状核查（2026-11-04，本地实测）**：
  - ✅ `results/` 已在公开仓库（git 已跟踪 61 个文件，含 `protocol_matrix_fig1_wide.csv`、`aggregation_gap.csv`、`p1_inf_alignment.csv`、`seeds/` 下 37 个种子聚合文件）。
  - ✅ `paper/data/fig2_sampling_noise.csv`、`paper/figures/`、`paper/refs.bib` 均已公开。
  - ✅ **逐图指标 CSV 已公开化（2026-11-04 完成）**：新目录 **`results/per_image/`**，**126 个文件、286 KiB**，每个文件是一个协议单元的逐图 `image,psnr,ssim`。含 `README.md` 说明命名规则、∞ 值的处理口径与复算方法。
  - ⚠️ **发布的是 126 个而非早先估的 186 个**，这是**修正而不是缩水**（见 §2.4.1）。

### 2.4.1 为什么是 126 个而不是 186 个（E76）

早先的估算（186 个、0.3 MB）是按"归档里所有 `per_image` 目录的文件数"数的，**没有核对这些文件是否就是支撑论文数字的那一批**。归档里混着三代评测产物：

| 归档目录 | 是否支撑论文 | 说明 |
|---|---|---|
| `kernel-output-protocol-full-v2/{generator_x4_x4,srcnn_x2_x2}` | ✅ 是 | ×4 / ×2 的 5 seed × 8 协议 |
| `kernel-output-protocol-full-v4/{lowlight_unet_x1,identity_ll}` | ✅ 是 | 低光 5 seed × 5 协议 + 恒等基线 |
| `kernel-output-protocol-full-v3/bicubic_sr_x4` | ✅ 是 | ×4 双三次基线 |
| `local-baseline-bicubic-sr-x2` | ✅ 是 | ×2 双三次基线（目录名与 job 名不同） |
| `kernel-output-protocol-full-v3/{bicubic_ll_x2,lowlight_unet_x2}` | ❌ 否 | 被取代的旧版本（`lowlight_unet_x2` ≠ 权威的 `lowlight_unet_x1`） |
| `kernel-output-protocol-repo-v1-failed/` | ❌ 否 | **失败跑**，文件名写着 `sr_generator_scale4__P3` 之类，与权威 checkpoint 名不符 |
| `kernel-output-protocol-repo-v3/` | ❌ 否 | 早期 protocol repo 的残留 |

**判定方法不是靠目录名，而是靠数值**：`scripts/verify_per_image_mapping.py` 先从文件名解析出它声称的 `(job, checkpoint, protocol)`，再把该文件重新求平均，与 `results/matrix_runs/<job>/protocol_matrix.json` 里对应 cell 的 `psnr_mean` 比对，容差 1e-3 dB。

实测结果：**126 个文件全部一一对应 126 个权威 cell，无重复占用、无遗漏**；其余 57 个判定为 stale（名字指向的 cell 不存在，或数值对不上）。

两个必须记下的坑：
- **∞ 值**：双三次上采样能精确重建平坦块，MSE=0 → PSNR=+∞。权威矩阵的均值是**剔除 ∞ 后**的均值（P1 的双三次基线是 95 张图的均值，模型侧是 100 张，见 E21）。求平均时必须先滤掉非有限值，否则整列变成 `inf`。
- **不能只靠数值匹配**：P3 与 P4 只差 4 像素 shave，均值相差约 0.01 dB，纯容差匹配会让两个文件同时"匹配"到同一个 cell。必须先用文件名定位、再用数值验证。

**发布后已验证（`scripts/verify_public_release.py`）**：只用 `results/per_image/` 里的公开 CSV 重新求平均，**126/126 个 cell 全部复现**；再按种子求均值与跨种子样本标准差，得到 **21 行（3 任务 × 协议）均值与 std，与 `results/protocol_matrix_fig1_wide.csv` 逐位对账，21/21 零偏差**。

### 2.5 其余硬性条款（原文要点）

| 条款 | 原文要点 | 对本文的含义 |
|---|---|---|
| 许可 | 通过后按 **CC BY 4.0** 上线，**作者保留版权** | 无需额外授权谈判 |
| DOI | 每个版本各发一个 DOI（`10.20944/preprints<YYYYMM>.<NNNN>.v1`） | 已实测该前缀属 **MDPI AG**；改稿要发新版本，不能"改掉旧的" |
| 筛选 | 编辑团队 24 小时内完成（多数情况） | 时间可控 |
| 账号 | 需要 **MDPI 账号**登录后提交 | 投稿前必须先注册 |
| 数据链接 | 可在提交时填外链数据地址；**提交后要补链接需联系编辑** | → **数据链接要在提交时一次填对** |
| 撤稿 | 预印本上线后**无法完全删除**，只能撤下 | 发前须确认 |
| 合作者 | 所有作者须知晓并同意 OA 发布 | 单作者论文，需本人确认即可 |
| 期刊政策 | 若后续投期刊，需确认该刊接受预印本 | 已在 `OUTLINE` §6 记录平台选择时同步考虑 |

### 2.6 学科分类

主类 **Computer Science and Mathematics**；二级类 **Artificial Intelligence and Machine Learning**
（该二级类 URL：`preprints.org/computer-science-and-mathematics/artificial-intelligence-and-machine-learning`）
→ **分类与已有 4 篇 preprint 一致**（既有作品的 `group-title` 字段实测为 "Computer Science and Mathematics"）。

---

## 3. 本文档对旧记载的更正（E73）

| 位置 | 旧写法 | 实测 | 处置 |
|---|---|---|---|
| `OUTLINE` §6 第 5 项 | "它不收 LaTeX 源码而收 PDF，所以排版自由度更高" | 官方 Instructions 明写 **Word 或 LaTeX 都收**，源码须连 `.bib` 一起打包 | 已改为"两种都收"（E73） |
| `OUTLINE` §6 第 4 项 | "Preprints.org 收 PDF，用通用 `article` 即可" | 结论仍成立（用 `article` 即可），但理由错了——不是"只能收 PDF" | 已补上准确理由（E73） |

**教训**：平台要求属于**外部事实**，不能凭印象写。本条的原始错误源于早期规划文档的一句顺口推断，未经核验就写进了两份文档——与 E52（"没读文献就写用途"）同类。**平台规则必须查官方 Instructions 原文。**

---

## 4. 投稿前的待办（按阻塞顺序）

| # | 事项 | 成本 | 状态 |
|---|---|---|---|
| 1 | 写正文（8–10 页，双栏） | 主成本 | ✅ **已完成**（2026-11-04）：`paper/main.tex`，**9 页双栏**，实测 pdflatex **0 错误 / 0 Overfull / 0 undefined reference**，**27 条参考文献全部被引**；正文 Table 1–7 与 `paper/tables/table1_`–`table7_` **编号一一对应**（E82） |
| 2 | 补一张**图形摘要**（JPG/PNG，非文本摘要截图） | 中 | ✅ **已完成**（2026-11-04）：`paper/figures/ga_protocol_audit.{png,jpg,pdf}`，2889×1470 px @ 300 dpi，三面板（协议跨度对比 / 符号翻转 / 报告率），与 Fig. 1 / Fig. 2 均非同一图 |
| 3 | 把逐图 CSV 公开化 | 小 | ✅ **已完成**（2026-11-04）：**126 个文件、286 KiB** → `results/per_image/`，含 README；已验证可复现全部 cell 与 21 行种子级统计（§2.4.1） |
| 4 | 注册 MDPI 账号 | 小 | ⬜ 待办 |
| 5 | 打包 LaTeX 投稿 zip（`.tex` + `refs.bib` + 图 + 样式文件） | 小 | ✅ **已完成**（2026-11-04）：`scripts/make_submission_package.py` → `dist/preprints_submission.zip`（**12 个条目、54 KiB**：`main.tex` / `refs.bib` / **7 张表** / 2 张图 / `README.md`）；**隔离目录从零编译验证通过**（9 页），且双向校验「正文 `\input` ↔ 包内清单」 |
| 6 | 填投稿表单：标题 / 摘要 / 关键词 / 学科分类 / ORCID / 通讯邮箱 | 小 | ⬜ 待做（元数据本文档已备齐；标题取候选 A） |
| 7 | 提交时填数据链接（仓库地址） | 小 | ⬜ 待做，**提交后无法自补**（§2.5） |
| 8 | 定标题 | 小 | ✅ **已定**：**候选 A** —— *The Protocol Is the Result: How Evaluation Choices Flip the Verdict on Lightweight Image Restoration* |
| 9 | **排版体检**（字号 / 浮体顺序 / 清单一致） | 小 | ✅ **已完成**（2026-11-04）：`scripts/check_layout.py`——量 PDF 每个文本行的行高（阈值 9 px ≈ 7 pt）、逐页定位「Table N:」与「Table N」引用、核对正文 `\input` 与 `paper/tables/*.tex` 双向一致。**结果：字号全部达标、无表出现在引用之前**（E84） |

### 4.1 Data Availability 声明（投稿表单逐字稿）

> **Data Availability Statement:** All per-image evaluation values underlying every protocol cell reported in this work are publicly available in the accompanying repository under `results/per_image/`, together with the aggregated protocol matrix (`results/matrix_runs/`), the derived tables (`results/*.csv`), and the scripts that regenerate them. Each released CSV contains one row per test image with its PSNR and SSIM; re-averaging these files reproduces every cell reported in the paper, and the repository includes a script (`scripts/verify_public_release.py`) that performs this check. The per-image release supports the distributional analyses reported in the paper, including the per-image correlation between gradient ratio and protocol-induced shift. The DIV2K validation split and the LOL dataset are not redistributed here; they are publicly available from their original sources and are referenced accordingly.

**提交时必须填的数据链接**（§2.5：提交后无法自补）：

```
https://github.com/ElijahZhao/pixelforge-image-restoration-/tree/main/results/per_image
```

⚠️ 该链接必须在**推送 `results/per_image/` 到远端之后**再填入表单——否则读者点开是 404。

**投稿前最后确认**：所有作者知晓并同意 OA 发布；确认后续目标期刊接受预印本；确认工作不侵犯任何第三方权利（本文引用了 Clean-FID、KinD 等他人成果并做对比——**只报文献值、不复跑，且标注 `as reported, not re-run`**，不复制其图表数据）。
