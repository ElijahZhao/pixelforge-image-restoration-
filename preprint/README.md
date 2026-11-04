# Preprints.org 相关文件

本目录集中存放投稿到 [Preprints.org](https://www.preprints.org/) 所需的全部文件。
元数据填写稿见 [`paper/PREPRINT_SUBMISSION.md`](../paper/PREPRINT_SUBMISSION.md)（字段逐条核实过官方 Instructions）。

- 目标期刊平台：Preprints.org（MDPI 运营），稿件类型 **Article**
- 学科分类：**Computer Science and Mathematics** → **Artificial Intelligence and Machine Learning**
- 许可：CC BY 4.0（作者保留版权）
- 数据可得性是硬性要求，本项目已满足（见下文"数据链接"）

---

## 一、需要在 Preprints.org 下载的文件

**这些不是要上传的，是投稿过程中平台侧产生、或本地需要准备下载的：**

| 文件 | 从哪里来 | 用途 |
|---|---|---|
| `preprints_submission.zip` | 本目录 `submission/` | **上传项**（LaTeX 源码包）。平台要求："For LaTeX files, ensure that all files necessary, including the .bib file, if applicable, to recreate the PDF are included in a zip or similar archive." |
| `manuscript.pdf` | 本目录 `submission/` | **上传项**（PDF 版稿件），由 `paper/main.tex` 编译而来，12 页 |
| `graphical_abstract.jpg` | 本目录 `submission/` | **上传项**（图形摘要）。2889×1470 px @ 300 dpi，三面板，与正文 Fig. 1 / Fig. 2 均不同图 |
| ORCID 记录页 | <https://orcid.org/0009-0005-4623-8465> | 投稿表单里 ORCID 一栏需填；该页同时显示姓名 Yulin Zhao 与别名 Elijah Zhao |
| MDPI 账号 | <https://www.mdpi.com/> 注册 | **必须先有账号**——平台要求"You will need to have an MDPI account and log in to submit" |
| 数据链接 | 本仓库 `results/per_image/` | 提交时填外链数据地址。**提交后要补链接需联系编辑**，所以提交时必须一次填对 |

> **图形摘要的硬性判据**（官方原文）："The graphic should be provided as a JPG or PNG file. Please note that a screenshot of the textual abstract does not qualify as a graphical abstract. Files submitted as graphical abstracts that do not meet the required criteria will be removed prior to the manuscript's posting."
> → 所以 **不要**上传 `fig1_protocol_matrix.png`（六面板矩阵图，非图形摘要）；上传 `graphical_abstract.jpg`。

---

## 二、需要在 Preprints.org 提交的文件（表单要填什么）

### 2.1 上传的文件（3 个）

| # | 文件 | 对应本目录 | 说明 |
|---|---|---|---|
| 1 | Manuscript (PDF) | `submission/manuscript.pdf` | 12 页双栏，首屏含标题/作者/摘要/关键词/通讯邮箱/单位 |
| 2 | LaTeX source (zip) | `submission/preprints_submission.zip` | 15 项：`main.tex` + `refs.bib` + 10 张表 + 2 张图 + `README.md`；已验证从零隔离编译通过 |
| 3 | Graphical abstract (JPG) | `submission/graphical_abstract.jpg` | 平台独立上传项 |

### 2.2 表单字段（逐字稿见 `paper/PREPRINT_SUBMISSION.md`）

| 字段 | 填写值 |
|---|---|
| Title | **The Protocol Is the Result: How Evaluation Choices Flip the Verdict on Lightweight Image Restoration** |
| Manuscript type | **Article** |
| Author | **Yulin Zhao**（别名 Elijah Zhao） |
| ORCID | `0009-0005-4623-8465` |
| Corresponding author email | `20233407131@stu.huel.edu.cn` |
| Affiliation | **Independent Researcher**（独立研究者，无机构归属） |
| Subject area | Computer Science and Mathematics → Artificial Intelligence and Machine Learning |
| Keywords | evaluation protocol; reproducibility; image super-resolution; low-light enhancement; PSNR; reporting standards |
| Data link | <https://github.com/ElijahZhao/pixelforge-image-restoration-/tree/main/results/per_image> |

### 2.3 Data Availability Statement（逐字稿）

> All per-image evaluation values underlying every protocol cell reported in this work are publicly available in the accompanying repository under `results/per_image/`, together with the aggregated protocol matrix (`results/matrix_runs/`), the derived tables (`results/*.csv`), and the scripts that regenerate them. Each released CSV contains one row per test image with its PSNR and SSIM; re-averaging these files reproduces every cell reported in the paper, and the repository includes a script (`scripts/verify_public_release.py`) that performs this check. The per-image release supports the distributional analyses reported in the paper, including the per-image correlation between gradient ratio and protocol-induced shift. The DIV2K validation split and the LOL dataset are not redistributed here; they are publicly available from their original sources and are referenced accordingly.

---

## 三、目录内容

```
preprint/
├── README.md                      ← 本文件
├── submission/                    ← 要上传的 3 个文件（+ 归档件）
│   ├── manuscript.pdf             上传项 1：PDF 稿件
│   ├── preprints_submission.zip   上传项 2：LaTeX 源码包
│   └── graphical_abstract.jpg     上传项 3：图形摘要
└── archive/                       ← 版本归档（投稿后放带 DOI 的版本）
```

### `submission/` 里各文件的来源

| 文件 | 由谁生成 | 可重现命令 |
|---|---|---|
| `manuscript.pdf` | `paper/main.tex` | `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` |
| `preprints_submission.zip` | `scripts/make_submission_package.py` | `python3.11 scripts/make_submission_package.py`（自带隔离编译校验） |
| `graphical_abstract.jpg` | `scripts/make_graphical_abstract.py` | `python3.11 scripts/make_graphical_abstract.py` |

三者的内容与仓库根目录 `deliverables/` 下的同名文件逐位一致（md5 相同）。

> **一致性保障**：`scripts/make_submission_package.py` 在重建投稿包后会逐条目比对本目录
> `submission/preprints_submission.zip` 与 `dist/preprints_submission.zip` 的**内容**
> （不是字节——zip 每次重建都会写入新的时间戳，比字节会每次都报假警），
> 不一致即报错退出。该检查已接入 CI。
> `manuscript.pdf` 与 `graphical_abstract.jpg` 由同一批源文件生成，
> 改论文后需重新执行：
> ```bash
> python3.11 scripts/make_submission_package.py
> cp dist/preprints_submission.zip preprint/submission/
> cp paper/main.pdf                  preprint/submission/manuscript.pdf
> cp paper/figures/ga_protocol_audit.jpg preprint/submission/graphical_abstract.jpg
> ```

---

## 四、参考文献有效性（2026-11-04 全量外部核验）

`paper/refs.bib` 共 **27 条**，全部通过 `scripts/check_references.py` 的 **83 项检查**：

| 检查 | 结果 |
|---|---|
| 24 条带 arXiv 号 | 逐条经 arXiv API 抓取，**标题逐字匹配**，年份口径自洽，`arxiv.org/abs/` 页面全部可达 |
| 2 条带 DOI | 在 Crossref **已注册**，注册标题与年份与 bib 吻合，`doi.org` 可提供元数据 |
| 1 条无 eprint/doi（Cawley & Talbot, JMLR 2010） | 期刊落地页可达 |
| `\cite` 键闭合 | 27 个引用键全部存在于 `refs.bib`，且 **27 条全部被引用**，无悬空引用、无未引用条目 |

**另做了一层"引文内容"核验**（读原文，不只看元数据），这一层查出并已修正一处实质错误：
**SEAL（ICLR 2024）讲的是"从巨大降级空间随机抽样降级案例"，全文没有 "crop" 一词**，
原稿把它写成 "random-crop evaluation" 属引用内容错误，已改正（见 `paper/FACTS.md` E91）。

其余文献的引用断言逐条核对无误：Keleş 2021 的 **2.0719 dB**（原文 Table I）与 2.5068 dB 理论极限、
HyperBench 的 **5→13 dB**、LOL 文献的 **1.48× 饱和度 / 14% 更暗 / pixel-identical 输入**、
Bridging 的"部分度量与人类感知呈负相关"、CleanFID 的抗锯齿、Musgrave 的 "marginal at best"、
Lucic 的 "no consistent advantage over the non-saturating GAN"、Bouthillier 与 Dodge 的方差/预算主张。

---

## 五、投稿前剩余的本人操作

1. 注册 MDPI 账号并登录
2. 上传 §二 的 3 个文件
3. 按 §2.2 填表单字段（含数据链接，**提交后无法自补**）
4. 粘贴 §2.3 的 Data Availability Statement
5. 确认合作者知情同意 OA 发布；确认后续目标期刊接受预印本

> **不可撤回**：平台原文 "preprints cannot be completely removed once online"，上线前请确认。
