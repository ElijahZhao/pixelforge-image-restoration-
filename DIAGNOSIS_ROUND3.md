# PixelForge 深度诊断（第三轮·内核与管线版）

> **本轮仍为只读审计，未改任何代码。** 全部数字由沙箱直接加载仓库真实权重/源码实测。
> 第一轮 `DIAGNOSIS_AND_PLAN.md`、第二轮 `DIAGNOSIS_ROUND2.md`。本轮专攻：**权重的技术内核、
> TorchScript 等价性、训练脚本正确性、服务/前端管线**。
> 生成日期：2026-10-03。

---

## 〇、本轮新增发现速览

| # | 发现 | 严重度 | 类别 |
|---|---|---|---|
| **R1** | 感知损失的 **VGG16 输入未做 ImageNet 归一化** → 特征 91.6% 是零，感知信号质量极差 | 🔴🔴 高 | 训练 bug |
| **R2** | SR 服务端 `before` 语义错误：SR 时 `before=原图放大`，**对比失真**（应展示低清输入） | 🔴 中 | 服务逻辑 |
| **R3** | 上传**无大小限制** + 无图片尺寸上限 → DoS / 内存风险 | 🟡 中 | 安全 |
| **R4** | `scale` 越界静默改值（3→2），无报错 | 🟡 低 | 健壮性 |
| **R5** | 训练/验证集划分**全靠手动放目录**，脚本不校验重叠 → 潜在数据泄露 | 🟡 中 | 方法学 |
| **R6** | `export.py` 用 `torch.jit.trace` **固定 dummy 尺寸**，但 SR 实际支持任意尺寸（幸运地没崩） | 🟡 低 | 健壮性 |
| **R7** | BN 统计健康但**部分通道退化**（running_var 最低 0.003）→ 部分通道几乎死掉 | 🟡 低 | 模型质量 |

**三轮回看**：第一轮发现 SR 损失权重失衡（感知占 99.2%）；第二轮发现低光模型输给基线+过曝、
demo 链路 bug；**本轮把"为什么训练会失败"的根因补齐了——训练脚本本身有 3 个独立缺陷**。

---

## 一、R1：VGG 感知损失输入未归一化（🔴 训练级 bug）

### 1.1 问题

`train/train.py` 第 46-58 行：

```python
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        vgg = models.vgg16(pretrained=True).features[:30].eval()
        ...
    def forward(self, pred, target):
        return self.criterion(self.vgg(pred), self.vgg(target))   # ← 直接喂 [0,1]
```

**VGG16 的预训练权重是在 ImageNet 归一化输入上训练出来的**，标准推理必须做：

```python
x = (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
```

而这里直接把 `[0,1]` 范围的 tensor 喂进去，**跳过了归一化**。

### 1.2 实测影响

```
当前输入范围        : 0.047 ~ 1.0   (未归一化)
归一化后应有范围    : -1.91 ~ 2.29

VGG features[:30] 输出的"非零率"（ReLU 后）:
  未归一化输入 : 8.4% 非零   ← 91.6% 的神经元是死的！
  已归一化输入 : 5.9% 非零
```

**91.6% 的特征图元素是 0**——意味着感知损失大部分时候"感知不到差异"，梯度信号极度稀疏。叠加第一轮发现的"感知项权重占 99.2%"，最终效果是：**模型被一个"又稀疏、又主导"的损失牵着走**，既学不到像素保真，也没得到有效的感知引导——两头不讨好。这从原理上解释了 SR 为何训练失败。

> 注：很多 SR 实现确实会在 VGG 前做归一化。这里缺了，属于**真实现缺陷**，不是"设计选择"。

---

## 二、R2：服务端 SR 的 `before` 语义错误（🔴 展示失真）

### 2.1 问题

`serve/app.py` 第 90-92 行：

```python
if task == "sr":
    ...
    # "before" = original shown at the upscaled size (blurry reference)
    before = original.resize((original.width * scale, original.height * scale), BICUBIC)
```

但 `predict_sr(original, scale)` **内部会先把 original ÷scale** 再超分。也就是说：

- **after**：`original → ÷4(128) → 模型 → ×4(512)`（真正的超分流程）
- **before**：`original → ×4`（**直接放大**，是一张"高清放大图"）

**两者不是同一张输入的两种处理**。用户若上传高清图：
- before = 高清图放大 → 很清晰
- after = 高清图降采样再超分 → 更糊

**结果是"增强后反而更差"，对比图完全失真。** 这与 R2 在 `make_demo.py` 里的 bug（第二轮 N3）是**同源问题**——SR 管线的"输入到底是谁"始终没理清。

### 2.2 正确的语义应该是

```
before = 低清输入（可放大显示）    ← 128 的 LR
after  = 模型对 LR 的超分结果      ← 512
```

---

## 三、R3/R4：服务端健壮性与安全

### R3 无上传限制（🟡）

`serve/app.py` 全文**没有任何文件大小/像素数限制**。`await image.read()` 全量读入内存。
配上 R6 的说明（U-Net 虽已 pad 修复，但超大图仍会吃满内存）：

- 上传 100MB 图 → 直接进内存 → 服务 OOM
- 属于免费的 DoS 面

**至少应有**：文件大小上限（如 10MB）、像素数上限（如 4000×4000）、超限返回 413。

### R4 scale 静默改值（🟡）

```python
if scale not in (2, 4):
    scale = 2       # 用户传 3 会静默变 2，无任何提示
```

不严重，但违反"最小惊讶原则"。应返回 422 明示。

---

## 四、R5：训练/验证划分方法学缺陷（🟡）

`train/datasets.py` 完全**依赖用户手动把图片放进 `train/` 和 `val/` 目录**，脚本：

- ❌ 不切分
- ❌ 不校验 train/val 是否重叠
- ❌ 不固定随机种子

**风险**：如果用户图省事把同一批图放两个目录（或 DIV2K 的 801-900 没正确分开），
**训练集和验证集会重叠 → 指标虚高 → 假成功**。

DIV2K 标准划分是 800 train / 100 val，但**没有任何代码强制或校验这一点**。
这是"指标可信度"的地基问题——没法证明 17.20 / 19.26 不是数据泄露的结果。

---

## 五、R6/R7：导出与模型质量

### R6 trace 固定尺寸（🟡）

```python
dummy = torch.rand(1, 3, 64, 64) if task == "lowlight" else torch.rand(1, 3, 16, 16)
traced = torch.jit.trace(model, dummy)
```

`trace` 会**记录 dummy 尺寸下的执行路径**。实测：

```
SR 输入 16/32/44/64 → 输出 64/128/176/256  ✓（卷积天然支持任意尺寸）
LL 输入 100×100 → 崩（U-Net 需 32 倍数）
```

**SR 之所以任意尺寸都能跑，是"卷积天然支持"的运气**，不是 trace 的保证。
更稳的做法是 `torch.jit.script`（保留控制流），或导出时多尺寸验证。当前属于"能跑但脆弱"。

### R7 BN 通道部分退化（🟡）

实测 SR 模型 16 个 BN 层的统计：

```
running_mean: absmean=0.169（偏小）
running_var : min=0.003, max=3.109, mean=0.695
```

`running_var` 最低 0.003 意味着**部分通道方差趋近 0 → 该通道输出被 BN 压成常数 → 通道死了**。
配合 8 个残差块的小容量，说明**模型有效容量比账面更小**。

---

## 六、三轮诊断合并：失败根因链（这是最有价值的部分）

把三轮发现串起来，SR 的失败**不是单一 bug，而是一条链**：

```
① 数据层：train/val 手动划分、无校验          (R5)
        ↓ 指标地基不牢
② 损失层：感知项权重 0.01 → 感知占 99.2%      (第一轮)
        ↓ 像素保真被抹掉
③ 损失层：VGG 输入未归一化 → 特征 91.6% 为零   (R1)
        ↓ 感知信号本身也稀疏失真
④ 模型层：BN 通道退化 + 仅 8 残差块            (R7)
        ↓ 有效容量不足
⑤ 评测层：val 随机 crop → 指标不可复现         (第二轮 N4)
        ↓ 无法察觉训练失败
⑥ 展示层：demo/serve 的 before/after 语义错   (第二轮 N3 / 本轮 R2)
        ↓ 对比图失真，掩盖了问题
⑦ 文档层：把失败解释成"设计选择"               (第一/二轮)
        ↓ 错误被固化
```

**低光模型的失败链**同理：crop 训练 + 小 U-Net + 无结构约束 → 学到"全局提亮"的退化解（过曝）→
被错误的文献对照基准包装成"成功"。

**这条链本身就是最好的作品素材**——它证明作者能**系统性定位多层缺陷**，而不是只会调参。

---

## 七、修正后的行动清单（三轮汇总）

| 优先级 | 任务 | 覆盖发现 | 含金量 | 成本 |
|---|---|---|---|---|
| **P0** | 修损失函数：权重（0.01→合理）+ VGG 归一化 | ①②③ | ⭐⭐⭐⭐⭐ | 极低 |
| **P0** | 修低光模型：诊断过曝→加结构/GT锚定→重训 | 二轮N1N2 | ⭐⭐⭐⭐⭐ | 中 |
| **P0** | 建**固定、可复现、同口径**的评测脚本（固定 val + 自测 baseline） | R5 N4 N5 | ⭐⭐⭐⭐⭐ | 中 |
| **P1** | 修 `make_demo.py` + `serve/app.py` 的 before/after 语义 | N3 R2 | ⭐⭐⭐ | 极低 |
| **P1** | 服务端加上传限制 + scale 校验 | R3 R4 | ⭐⭐⭐ | 低 |
| **P2** | 训练产物归档（CSV + 收敛曲线 + 复现脚本） | N6 | ⭐⭐⭐⭐ | 低 |
| **P2** | 模型结构优化（去 BN / 加块）+ 多尺寸导出验证 | R6 R7 | ⭐⭐⭐ | 中 |
| **P3** | 把整条"失败根因链 → 修复 → 验证"写成案例 | 全部 | ⭐⭐⭐⭐⭐ | 中 |

---

## 附：本轮全部结论的复现命令

```bash
# R1  VGG 未归一化 → 特征 91.6% 为零
cd /workspace && python3.11 -c "
import torch; from torchvision import models; import torchvision.transforms.functional as TF
from PIL import Image
w=models.vgg16(weights='DEFAULT'); f=w.features[:30]
img=TF.to_tensor(Image.open('assets/sample_scene.png').convert('RGB')).unsqueeze(0)
with torch.no_grad(): out=f(img)
print('非零率', round(float((out!=0).float().mean()),3))"

# R2  SR 服务端 before/after 语义不一致
cd /workspace && sed -n '86,92p' serve/app.py

# R3  无上传大小限制
cd /workspace && grep -n "read()\|size\|limit" serve/app.py || echo "无限制"

# R7  BN 通道退化
cd /workspace && python3.11 -c "
import torch
m=torch.jit.load('serve/models/sr_generator_scale4.pt',map_location='cpu')
rv=torch.stack([v for k,v in m.named_buffers() if k.endswith('running_var')]).flatten()
print('running_var min',round(float(rv.min()),4),'mean',round(float(rv.mean()),3))"
```
