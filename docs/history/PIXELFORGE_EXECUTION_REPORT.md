# PixelForge 提升计划 · 执行报告

> 本报告记录《修复方案》（`PIXELFORGE_FIX_PLAN.md`）的实际执行结果。
> **所有修复均附验证证据**——这正是 19 轮审计发现的核心缺失环节。

---

## 一、执行总览

| 编号 | 修复项 | 状态 | 验证结果 |
|---|---|:---:|---|
| F1 | 轮换并移除 .git/config 暴露的 token | ✅ | config 中明文凭据已清除 |
| F2 | 感知损失补 ImageNet 归一化 | ✅ | 特征幅值恢复 + 对扰动敏感度 4.4× |
| F3 | 损失权重重平衡 | ✅ | 像素项不再被 0.01 抹除，显式可读 |
| F4 | SR 重训去感知损失 | ⏸️ 待用户 | 需 GPU，命令已就绪 |
| F5 | 训练目标与评估目标对齐 | ✅ | 目标+best 判据写入同一日志 |
| F6 | 补测可复现真实指标 | ⏸️ 待用户 | 需先重训（F4）产生产物 |
| F7 | 消除 SR×4 隐藏的 4 倍假放大 | ✅ | 输出 = lr*scale，非 orig*scale |
| F8 | ×2 静默回退显式化 | ✅ | 后端 note + 两 UI 徽章明示 |
| F9 | 数据管线鲁棒性补丁 | ✅ | 尺寸不匹配/损坏/数量不一致→跳过告警 |
| F10 | 叙述层对齐（取舍/胜出/100%） | ✅ | README/PROGRESS/results 已修正 |
| F11 | 统一部署文档 | ✅ | 根 DEPLOY.md 标注历史+指向主路径 |
| F12 | demo 图可复现 | ✅ | 脚本产 .jpg，与 README 引用一致 |
| F13 | 推理服务加输入/资源限制 | ✅ | 413 生效（超大文件/像素） |
| F14 | 测试从形状升级到正确性 | ✅ | 20/20，变异测试拦截率 4/4 |

**已完成 12 项；2 项（F4/F6）因需真实 GPU 训练而标记为待用户执行。**

---

## 二、逐项证据

### F1 · token 轮换
- **改动**：`git remote set-url origin https://github.com/ElijahZhao/pixelforge-image-restoration-.git`
- **验证**：`grep -E "ghp_|oauth2:" .git/config` → 无输出 ✓
- **⚠️ 用户待办**：GitHub 上**撤销旧 token**（本地无法代劳；该 token 曾穿第三方镜像，必须视为已泄露）。

### F2 · 感知损失归一化
- **改动**：`train/train.py` 的 `VGGPerceptualLoss` 增加 `_feat()`，提取特征前做 `(x-mean)/std`。
- **验证**（用户画像关键）：
  - 高纹理图特征均值：0.0286 → 0.0742（**2.6×**）
  - **对"轻微模糊"扰动的感知距离：0.00529 → 0.02337（4.4×）**——这才是感知损失恢复有效的本质证据。
- **修正**：我原定判据"零占比<5%"**被实验推翻**——ReLU 固有稀疏性使零占比恒高，非有效判据。改用"特征幅值 + 扰动敏感度"。

### F3 · 损失权重重平衡
- **改动**：`loss = 0.01*loss + percep` → `loss = w_pixel*pix + w_percep*per`（默认 1.0 / 0.006，可 CLI 配置）。
- **验证**：两项贡献比从"像素 0.2%"变为"像素 99.8%"（显式可读）✓

### F5 · 训练/评估对齐
- **改动**：CSV 日志头部写入 `# objective=` 与 `# best_criterion=val_psnr`，使"优化什么"与"选 best 看什么"同文件可见。

### F7 · 消除假放大
- **改动**：`serve/model_loader.py` + 两个部署入口的 `predict_sr` 返回模型真实输出；`serve/app.py` 的 `before` 改为同一 LR 的 bicubic 上采样（与 after 同分辨率）。
- **验证**：输入 512、scale=4 → 输出 **512**（=lr*scale），而非旧的虚假 2048 ✓

### F8 · ×2 回退显式化
- **改动**：`serve/app.py` 的 note 如实说明"无对应权重"；streamlit 徽章增 `SR ×2` 状态；hf_space `_engine_status` 增 `SR ×2`。
- **验证**：SR×2 请求 → `engine=classical`，`note="Classical baseline in use — no trained weight found for sr x2"` ✓

### F9 · 数据管线鲁棒性
- **改动**：`LowLightDataset` 按**文件名 stem 配对**、校验**尺寸一致**、跳过损坏文件并 WARNING；val 改为**确定性**（可复现）。
- **验证**：数量不一致→跳过(样本数=1)✓；尺寸不同→跳过并告警(不再静默错位)✓；损坏→跳过✓；val 两次取样相同✓

### F10/F11/F12 · 文档层
- **F10**：README/results/PROGRESS 的"刻意取舍""已验证胜出""指标✅100%"全部改为可证伪表述（"属待修复问题""未自测""待复现"）。
- **F11**：根 `DEPLOY.md` 顶部加"历史/备选方案"声明，指向实际主路径 `deploy/streamlit/DEPLOY_STREAMLIT.md`。
- **F12**：`make_demo.py` 改为产出 `.jpg`；**验证**：脚本输出的 6 个文件名与 README 引用的 6 个**完全一致** ✓

### F13 · 服务端限制
- **改动**：`serve/app.py` 加 `MAX_UPLOAD_BYTES=10MB`、`MAX_INPUT_PIXELS=4MP`（可 env 覆盖）。
- **验证**：11MB 文件→**413**✓；5MP 图→**413**✓；正常图→200✓

### F14 · 正确性测试（本轮核心）
- **改动**：新增 `train/tests/test_correctness.py`（8 个正确性测试），注册进 `run_tests.py`。
- **验证**：套件 **20/20 通过**。
- **变异测试**（决定性的验收）：注入 4 个真实缺陷类 → 新套件**全部拦截（4/4）**：
  - 去掉 VGG 归一化 → ✅拦截
  - 恢复 `0.01` 权重 → ✅拦截
  - 恢复 `orig*scale` 假放大 → ✅拦截
  - 去掉尺寸一致性校验 → ✅拦截
  - （对比第 9 轮：旧"形状测试"对这类错误**全部漏掉**。）

---

## 三、全量回归

| 检查 | 结果 |
|---|---|
| 单元测试 | **20/20 passed** |
| 后端端到端 | health 诚实；SR×4→ml；SR×2→classical+明示 note |
| 4 个入口 predict_sr 一致性 | 均无 orig*scale 假放大 |
| 9 个改动 Python 文件语法 | 全部 OK |
| demo 脚本可复现性 | 产出与 README 引用文件名一致 |

---

## 四、改动文件清单

**代码（8 个）**：`train/train.py`、`train/datasets.py`、`train/tests/run_tests.py`、`train/tests/test_correctness.py`(新)、`serve/app.py`、`serve/model_loader.py`、`scripts/make_demo.py`、`deploy/streamlit/streamlit_app.py`、`deploy/hf_space/app.py`

**文档（5 个）**：`README.md`、`PROGRESS.md`、`results/README.md`、`DEPLOY.md`、`assets/demo_*.jpg`(重新生成)

**未改动**：`.git/config`（已通过 `git remote` 命令清理，非源码文件）

---

## 五、下一步（需用户决策）

1. **🔴 撤销 GitHub 旧 token**（安全，仅你可在 GitHub 操作）。
2. **🟠 在 AutoDL 上用修复后的代码重训 SR×4 与低光**（`train/train.py` 已含 F2/F3/F5 修复）——这是让"修复真正体现为指标提升"的唯一途径。
3. **🟡 重训后执行 F6**：提交 `results/` 训练日志 + 评测脚本，让指标可复现。
4. **🟢 重新部署**：把修复后的 `streamlit_app.py` 推到 Streamlit Cloud，去掉徽章里的旧表述。
