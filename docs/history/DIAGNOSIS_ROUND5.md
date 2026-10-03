# PixelForge 深度诊断（第五轮·文档与一致性版）

> **本轮仍为只读审计，未改任何代码。** 全部结论由沙箱直接读源码/权重/文档实测。
> 前四轮：损失权重 → 低光失败 → VGG未归一化 → 前端与契约。
> 本轮专攻：**唯一未读的前端文件、跨文档一致性、命名与元数据、陈旧产物**。
> 生成日期：2026-10-03。

---

## 〇、本轮新增发现速览

| # | 发现 | 严重度 | 类别 |
|---|---|---|---|
| **D1** | `web/app/method/page.tsx` **完全过时**：说"无权重/TBD/HF Spaces"，与线上 README 全面矛盾 | 🔴🔴 高 | 文档矛盾 |
| **D2** | 低光 checkpoint 命名 `lowlight_srcnn_scale2_best.pth` **误导**——实际是 U-Net，不是 SRCNN | 🔴 中 | 命名 |
| **D3** | `train.py` 的 `--model` 对 lowlight **被静默忽略**（永远建 U-Net），但文件名/文档仍在传 `srcnn` | 🟡 中 | 参数语义 |
| **D4** | `DEPLOY.md` 描述的是**已废弃的部署方案**（Vercel + HF Spaces），与实际 Streamlit Cloud 不符 | 🟡 中 | 文档陈旧 |
| **D5** | method page 里的训练命令用 `--model srcnn`（对 SR 是错的默认）会**误导复现者** | 🟡 中 | 文档误导 |
| **D6** | `.env.local.example` 等部署文件残留，指向不存在的后端 | 🟢 低 | 陈旧 |

**核心结论**：项目存在**两套互相矛盾的"叙事"**——README/线上说"已训好、已上线"，
而 `method/page.tsx` 和 `DEPLOY.md` 还停留在"未训练、待部署"的旧状态。**文档层已经分裂。**

---

## 一、D1：`method/page.tsx` 全面过时（🔴 最严重）

### 1.1 矛盾清单

`web/app/method/page.tsx` 与线上 README/实际状态逐条对照：

| 内容 | method page 说 | 实际（README/实测） | 矛盾 |
|---|---|---|---|
| 指标表 | SRCNN/SRResNet/UNet 全 **TBD** | SR×4=17.20、lowlight=19.26（已测） | 🔴 |
| 权重 | "Until real weights are trained"（未训） | 权重已随仓库分发、线上已加载 | 🔴 |
| 部署 | "Next.js (Vercel) + FastAPI (Hugging Face Spaces / VPS)" | **Streamlit Cloud 单页** | 🔴 |
| 标题文案 | "the 'ours' rows are honestly marked TBD" | 早已填真实值 | 🔴 |

### 1.2 严重性

`method/page.tsx` 是**网站上的"如何实现"页面**——它本该是项目技术叙事的正式出口。
但它现在展示的是一份 **"项目还没开始训练"的声明**，与首页/README/线上 Demo **正相反**。

后果：
- 任何人点进 `web/` 的 method 页（若 web 被部署）会看到"未训练"自述 → **自相矛盾**
- 即便 web 未部署，仓库里同时存在"已训好"和"未训练"两种说法 → **审查者会困惑哪个是真的**
- 这也是"把失败粉饰"的反面：**README 改新了，但 method page 忘了同步** → 留下自相矛盾的双叙事

> 这印证了前几轮的判断：项目**缺乏单一的真相来源（single source of truth）**，各处文档各自漂移。

---

## 二、D2/D3：低光模型的命名与参数语义混乱

### 2.1 D2：文件名 `srcnn` 却是 U-Net

各处文档（`train/train.py` docstring、`train/export.py`、`README.md:296-300`）都写：

```
models/lowlight_srcnn_scale2_best.pth
```

**但实测** `serve/models/lowlight.pt` 的结构：

```
模块: ['', 'enc1', 'enc1.0'..., 'enc2', ...]   ← 是 U-Net
含 enc1: True
含 conv1: False                                 ← 不是 SRCNN
```

`build_model('lowlight')` **永远返回 `LowLightUNet()`**，与 `--model` 无关：

```python
if task == "lowlight":
    return LowLightUNet()      # ← 忽略 --model
```

所以 **"lowlight_srcnn"** 这个名字从头到尾就是错的——它既不是 SRCNN，`--model srcnn` 也没起作用。
这是**命名与实际实现脱节**，会让任何复现者困惑（"我训的到底是啥？"）。

### 2.2 D3：`--model` 对 lowlight 是"死参数"

`train.py` 第 71-72 行：

```python
model = build_model(args.task, scale=args.scale, advanced=(args.model == "generator"))
```

但对 `task == "lowlight"`，`build_model` 里 `advanced` 参数**被完全忽略**。也就是说：
用户传 `--model generator` 或 `--model srcnn` 对低光**毫无区别**。这是**静默的无效参数**。

---

## 三、D4：`DEPLOY.md` 描述的是废弃方案（🟡）

`DEPLOY.md` 全文讲的是：

```
1. Frontend (Next.js) → Vercel
2. Backend (FastAPI) → Hugging Face Spaces
```

而**实际情况**：

- HF Spaces 因改收费政策被放弃（项目历史中有记录）
- 最终部署是 **Streamlit Cloud 单页**（`deploy/streamlit/`）
- 真正的部署步骤在 `deploy/streamlit/DEPLOY_STREAMLIT.md`

于是仓库里**同时存在**：
- `DEPLOY.md` → 讲 Vercel + HF（废弃）
- `deploy/streamlit/DEPLOY_STREAMLIT.md` → 讲 Streamlit（现役）
- `deploy/hf_space/DEPLOY_HF.md` → 讲 HF（废弃，但保留）

**三份部署文档，两份过时，没有一份标注"已废弃"。**

---

## 四、D5：method page 的训练命令会误导复现者

`method/page.tsx` 第 135-138 行：

```
# low-light (default --model srcnn --scale 2)
python train/train.py --task lowlight --data_root data --epochs 200
python train/export.py --checkpoint models/lowlight_srcnn_scale2_best.pth ...
```

问题：
1. 注释 `default --model srcnn` —— 对低光**这个参数根本无效**（D3）
2. checkpoint 名带 `srcnn` —— 与实际 U-Net 不符（D2）
3. 与 README 里**已经更新过的真实命令**（含 `--batch_size 8 --lr 2e-4`）不一致

复现者照此操作会得到**与文档描述不同的东西**。

---

## 五、五轮诊断完整合并

### 5.1 全部发现（约 28 项）

| 轮次 | 主题 | 关键发现 |
|---|---|---|
| 1 | 训练损失 | SR 感知损失权重失衡（占 99.2%） |
| 2 | 模型质量 | 低光输给基线+过曝；demo SR 输入 bug；val 随机 crop；基准不同口径 |
| 3 | 内核算子 | VGG 未归一化（特征 91.6% 为零）；serve before 语义错；无上传限制；BN 退化；数据划分无校验 |
| 4 | 前端契约 | 滑块缩放错位；前端 before 语义错（第3次）；e2e 形同虚设；Streamlit 2× 陷阱；默认值不一致 |
| 5 | 文档一致 | method page 全面过时；低光命名误导；`--model` 死参数；DEPLOY.md 废弃 |

### 5.2 三类系统性问题（这是最深的洞见）

把 28 项发现归类，我看到的**不是 28 个孤立 bug，而是三类系统性缺陷**：

**类型 A：核心算法/训练原理错误**
- SR 损失权重 0.01（感知占 99.2%）
- VGG 未归一化
- 低光退化解（过曝）
→ **共同点：都指向"这是训练失败，而被误当成设计选择"**

**类型 B：输入/输出契约从未被定义**
- SR 的 before/after 语义，在 4 处各错一次（make_demo / serve / web / streamlit）
- 低光 32 倍数约束（靠 pad 补丁兜）
→ **共同点：数据在管线中"流经多个环节"，但每个环节对'输入是什么'的理解都不同**

**类型 C：缺乏"可信度验证"机制**
- 随机 crop 的 val（指标不可复现）
- 文献基准冒充自测对照
- e2e 只验长度不验内容
- 文档分裂（README 新 / method 旧）
→ **共同点：没有任何机制能保证"声称的"等于"实际的"**

### 5.3 一句话总结（第五轮修正版）

> **PixelForge 是一个"工程外壳完整、但核心技术叙事不可信"的项目。它不缺代码，缺的是'让声称与事实一致的机制'——从训练损失的配置、到评测的口径、到文档的同步，每一层都允许'看起来对'掩盖'实际不对'。**

---

## 六、附：本轮全部结论的复现命令

```bash
# D1  method page 过时（TBD + HF Spaces vs README 已训好+Streamlit）
cd /workspace && grep -n "TBD\|Hugging Face" web/app/method/page.tsx
cd /workspace && grep -n "已上线\|自训模型" README.md | head -4

# D2  低光命名误导（lowlight.pt 是 U-Net 不是 SRCNN）
cd /workspace && python3.11 -c "
import torch
m=torch.jit.load('serve/models/lowlight.pt',map_location='cpu')
n=[x for x,_ in m.named_modules()]
print('含 enc1:', any('enc1' in x for x in n), '| 含 conv1:', any('conv1' in x for x in n))"

# D3  --model 对 lowlight 无效
cd /workspace && sed -n '154,160p' train/models.py

# D4  DEPLOY.md 描述废弃方案
cd /workspace && head -15 DEPLOY.md

# D5  method page 训练命令误导
cd /workspace && sed -n '135,138p' web/app/method/page.tsx
```
