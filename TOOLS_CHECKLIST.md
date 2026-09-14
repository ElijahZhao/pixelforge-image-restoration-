# PixelForge · 使用工具清单（内置专家 / 技能 / 连接器）

> 盘点 WorkBuddy 内置的"专家、技能、连接器"，挑出**对本项目（AI 计算机视觉作品集，用于申请美国研究生）真正有用**且能在沙箱里调动的部分。
> 项目现状：代码闭环（训练/推理/前端/文档）已完成并本地跑通，缺口是"真实权重、线上部署、申请材料包装、自动化测试"。本清单围绕这些缺口展开。

---

## 一、总览表（按价值排序）

| 工具 | 类型 | 在本项目中的用途 | 是否需授权 / 外部资源 |
|---|---|---|---|
| **发布为应用** | 技能 | 一键把网站发布成**在线分享链接**，导师直接打开 demo | 需你当轮明确说"发布"（每次重新授权） |
| **github-connector** | 连接器 | 把 `/workspace` 干净仓库 **push 到 GitHub** 作公开作品集源码 | 需 GitHub OAuth 授权 |
| **docx / pdf** | 技能 | 生成英文 **SOP / CV / 项目报告** | 本地生成，无需授权 |
| **pptx** | 技能 | 生成**项目答辩 / 面试幻灯片** | 本地生成，无需授权 |
| **xlsx** | 技能 | 生成 **PSNR/SSIM 指标对比表、训练超参记录表** | 本地生成，无需授权 |
| **humanizer** | 技能 | 把 AI 味的 SOP/简历英文**润色成自然表达** | 本地，无需授权 |
| **summarize** | 技能 | 总结 SRCNN/ESRGAN/Retinex 等**论文**，用于 Method 页与 SOP 综述 | 本地，无需授权 |
| **agent-browser / playwright-cli** | 技能 | **E2E 自动化测试**网站（上传→滑块→API）并截图留证 | 本地浏览器自动化，无需授权 |
| **openai-image-gen / nano-banana-pro** | 技能 | 生成网站 **hero 图 / OG 图**（⚠️ 不用于冒充模型效果） | 可能需 API key |
| **资料库（library）** | 技能 | 把报告/文档存个人空间并生成**可分享链接**（备选托管） | 本地，无需授权 |
| **figma-connector** | 连接器 | 若要做精致 UI 设计稿再转代码（可选） | 需 Figma OAuth 授权 |
| **Explore / Plan / general-purpose** | 专家 | 审查代码、设计 B/C 方向、跑复杂多步任务 | 本地，无需授权 |

> **明确排除**：tencent-docs / tencent-local-office（腾讯文档，申美研用处低）、cnb / gongfeng（腾讯内网 git，非公开作品集）、automation-task-manager（定时任务，本项目无需）、apple-notes/imsg/trello 等个人效率工具、skill-creator / skill-vetter（过度工程）。

---

## 二、分模块详述

### A. 部署与发布
- **发布为应用（最优先）**：支持 Node.js / Python(FastAPI) / 静态站点**单端口** HTTP 服务。本项目前端已同源反代，可单端口发布；但后端此时仍在沙箱，导师访问需后端在线。**长期公开仍走 Vercel(前端) + HF Spaces(后端)**（见 `DEPLOY.md`）。需你当轮明确授权，绝不静默发布。
- **github-connector**：把 `/workspace` 推成公开仓库——招生官查看"代码可复现、训练流水线完整"的**主证据**。需 GitHub OAuth。

### B. 申请材料包装（申研核心）
- **docx / pdf**：英文 SOP 段落、CV 项目描述、技术报告（`web/app/method/page.tsx` 已是现成英文素材）。
- **pptx**：项目答辩 / 面试幻灯片。
- **xlsx**：PSNR/SSIM 量化对比、训练超参（lr / batch / epochs / GPU 时长）结构化表格。
- **humanizer**：去 AI 味，更自然。
- **summarize**：抓论文要点，写进 Method 页 "Related Work"。

### C. 质量与可信度
- **agent-browser / playwright-cli**：自动化打开网站、上传图、拖滑块、调 `/api/predict` 并截图，作为 "demo 真实可用" 取证。
- **openai-image-gen / nano-banana-pro**：仅做装饰性配图，**绝不用 AI 生成图冒充模型真实效果**。

### D. 知识沉淀（备选）
- **资料库（library）**：报告/文档存个人空间并生成可分享链接，作 GitHub 之外备选托管。

### E. 专家子代理
- **Explore**：审查代码、找改进点（Grad-CAM 可视化、模型量化）。
- **Plan**：设计候选 B（可解释医学分类）或 C（YOLO 检测）。
- **general-purpose**：执行复杂多步任务（如小规模合成数据跑通真实 checkpoint 证明流水线）。

---

## 三、推荐执行顺序（组合拳）

1. **补真实内容**：`summarize` 读论文 → 写进 Method 页；`xlsx` 做指标表（先占位，训完填真值）。
2. **包装申请材料**：`docx`/`pdf` 出 SOP+CV → `humanizer` 润色 → `pptx` 出幻灯片。
3. **自动化质控**：`agent-browser` 跑 E2E 测试截图，放进 README。
4. **对外发布**：`github-connector` 推公开仓库 → `发布为应用` 出 demo 链接（或按 `DEPLOY.md` 走 Vercel+HF）。

---

## 四、必须说清楚的限制

- ⚠️ **任何内置工具都替代不了"真实 GPU 训练"**：沙箱无 GPU，自训权重必须你去 **Kaggle / Colab 免费 GPU** 跑 `train/train.py` + `export.py`，把 `.pt` 放进 `serve/models/`，服务才会从 "Classical baseline" 切到 "Trained PyTorch model"。
- ⚠️ **发布为应用单端口限制**：长期公开建议 Vercel + HF Spaces，发布为应用只作快速 demo 链接。
- ⚠️ **AI 生成图不能充当模型效果**：只用于装饰，真实效果用你训练后跑测试集的输出图。
- ⚠️ 需 OAuth 授权的（GitHub / Figma）和需当轮授权的（发布为应用），不会静默操作。
