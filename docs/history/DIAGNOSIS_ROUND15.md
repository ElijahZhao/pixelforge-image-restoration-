# PixelForge 深度审计 · 第 15 轮

**主题：ML 核心架构的论文保真度（经得起懂行人对照经典论文盯着看吗？）**
**方法：实读源码 → 逐条对照 Ledig 2017 / Johnson 2016 / Dong 2014 → 用代码枚举验证，避免凭记忆下结论**
**硬约束：只读不改，零代码改动**

---

## 0. 本轮的立场：先校准，再指控

前面 14 轮挖出大量"运行期/训练期"缺陷。但一个懂行的人看项目，第一步往往不是看 bug，而是看**你到底有没有真懂这套方法**——架构是不是照论文忠实实现，还是照猫画虎拼装。

这一轮我把 ML 核心（`SRGenerator` / `SRCNN` / `LowLightUNet` / `VGGPerceptualLoss`）逐条和经典论文对照。**关键原则：能证明"忠实"的就明说忠实，能证明"偏差但可接受"的就标偏差，只有真正破坏方法前提的才指控为缺陷。** 这一轮我主动推翻了自己两个预设（见 §4），这正是审计最可信的地方。

---

## 1. 对照总表

| 组件 | 声称/暗示的来历 | 实际实现 | 保真度判定 |
|---|---|---|---|
| `SRGenerator` 残差块 | SRResNet / Ledig 2017 | `Conv-BN-PReLU-Conv-BN` + `x+body(x)` | ✅ **忠实**（含 BN、PReLU、残差） |
| `SRGenerator` 深度 | "SRResNet-style" | `num_blocks=8`（论文 16） | ⚠️ 半容量"轻量版"，可接受 |
| `SRGenerator` 上采样 | — | 单次 `PixelShuffle(scale)` | ✅ **EDSR 式**单步上采样，可接受 |
| `SRGenerator` 全局跳跃 | SRResNet 有全局残差 | **无**（学绝对 HR，非相对 bicubic 残差） | ⚠️ 偏离 Ledig，但 **EDSR 也去掉全局跳跃**，可接受 |
| `VGGPerceptualLoss` 层 | docstring "relu5_4" | `features[:30]` = **relu5_3** | ⚠️ **层选得对**（=SRGAN VGG54），但 docstring 写错层名 |
| `VGGPerceptualLoss` 归一化 | Johnson/SRGAN 都要求 | **完全缺失** | ❌ **真正的保真度缺陷**（方法前提被丢弃） |
| `SRCNN` 结构 | Dong 2014 三层 | 9-5-5 + 前两层 ReLU + 无 BN | ✅ **忠实** |
| `SRCNN` 上采样位置 | 论文：网络外先上采样 | 折进网络内部（`self.upscale`） | ⚠️ 可接受变体 |
| `LowLightUNet` | "inspired by Retinex，非论文" | 标准 U-Net + skip concat | ✅ 诚实声明，无保真度声称 |
| 低光 checkpoint 文件名 | `lowlight_srcnn_scale2_best.pth` | 实为 U-Net | ❌ 标签造假（第 5 轮） |

---

## 2. `SRGenerator`：结构基本忠实，两个"可接受偏差"

**忠实的部分（值得肯定）：**
- `ResidualBlock`：`Conv2d → BatchNorm2d → PReLU → Conv2d → BatchNorm2d`，输出 `x + body(x)`。这与 Ledig et al. 2017 的 SRResNet 残差块**完全吻合**（BN + PReLU + 残差）。
- 上采样用单次 `PixelShuffle(scale)`。虽然 Ledig 用两次 ×2 subpixel，但 **EDSR（Lim 2017）正是用单次 PixelShuffle**，属主流可接受实现。

**偏差但可接受：**
- `num_blocks=8` vs 论文 `B=16`。文档自称"Lightweight"，半容量。这**部分解释**了为什么实测能力弱——容量只有参考的一半，**但叠加感知损失缺陷才是主因**（见第 11/13 轮）。
- **无全局跳过连接**：`forward`（86–92 行）里原始输入 `x` 仅在 `self.entry(x)` 出现一次，从未加回输出。网络学的是**绝对 HR**，而非"相对 bicubic 上采样的残差"。Ledig 的 SRResNet 有全局跳跃（`output = net(lr) + bicubic_up(lr)`），但 **EDSR 明确去掉了全局跳跃、学绝对映射**。所以这是"风格偏离 SRResNet、对齐 EDSR"，**不是 bug**，只是命名"SRResNet-style"略宽松。

> 结论：`SRGenerator` 是**有理解地实现**的——残差块、BN、PReLU 都对，偏差都可追溯到某个主流变体（EDSR）。这本身就说明作者读得懂方法，不是照抄。

---

## 3. `VGGPerceptualLoss`：层选对了，但丢了方法的前提

**这是本轮最该讲清楚的点，也是唯一的"真缺陷"。**

**(a) 层选择其实是对的——我原本的准备指控被推翻。** 代码用 `vgg16.features[:30]`。我**枚举确认**：`features[28]=Conv5_3`、`[29]=ReLU5_3`、`[30]=Pool5`，所以取的是 **ReLU5_3**。

- Johnson et al. 2016 用 **relu2_2**；
- Ledig et al. 2017（SRGAN）用 **relu5_3（所谓 VGG54，第 5 个 maxpool 前的特征）**。

也就是说，项目的 relu5_3 **恰好对应 SRGAN 的 VGG54 选择**，并非"错用了最深层"。**所以"感知损失用了最深层是错误的"这个直觉是错的——不能这么指控。** 我之前没细查就差点这么写，实读+枚举纠正了。

**(b) 真正的缺陷：完全没有 ImageNet 归一化。** Johnson 2016 和 Ledig SRGAN 的感知损失都**明确要求**把输入做 ImageNet 均值/标准差归一化（VGG 是在归一化数据上预训练的）。项目 `forward` 直接 `self.criterion(self.vgg(pred), self.vgg(target))`，喂的是 `[0,1]` 原始像素。

> 这等于"照抄了 SRGAN 感知损失的**层选择**，却**漏掉了它之所以成立的前提条件**"。第 3/11/13 轮已量化后果：未归一化输入让 VGG 特征 **94% 退化为零**、均值只有归一化时的 1/4.8——感知项在数值上几乎失效。

**(c) docstring 与代码 off-by-one。** 类注释写"`relu5_4`"，代码实际是 `relu5_3`（差一个池化层）。属于"文档没和代码对齐"的小瑕疵，但**懂行的人读源码会立刻看到这条注释与实现的矛盾**，削弱可信度。

**(d) 单一层、无逐层权重/归一化。** SRGAN 原文用的是单层 VGG54 特征 + L1，所以"单层"本身不扣分；但 Johnson 的多层加权（relu1_2/2_2/3_3/4_3/5_3）更稳。属可改进项，非缺陷。

> **专家视角的定性结论**：感知损失"不是选错层，而是缺了归一化这个前提"。修复方向极其明确——保留 relu5_3，**只补 `normalize = (x - mean)/std` 这一行**就能让感知项从"94% 死亡"恢复到"有意义"。这是整个项目里性价比最高的单点修复。

---

## 4. 两个被我亲手推翻的预设（本轮的校准）

1. **"感知损失用了最深层 relu5_4 是错的"** → 错。枚举证明是 relu5_3，且这正是 SRGAN 的 VGG54 合法选择。真正的问题是缺归一化，不是层选错。
2. **"SRGenerator 有全局跳过连接"**（我的脚本一度误报）→ 错。函数签名里的 `x` 干扰了判断；实读 forward 确认原始 LR 从未加回输出，网络学绝对映射（EDSR 式，可接受）。

这两处都是"先立结论、再实读/枚举证伪"——和前几轮（第 11 轮损失量级、第 12 轮数据配对、第 14 轮 next.config 缺失）同一传统。**能推翻自己的假设，才是审计可信的根。**

---

## 5. `SRCNN` 与 `LowLightUNet`：一个忠实，一个诚实

**SRCNN（Dong et al. 2014）——忠实。** 三层 `9×9-64 / 5×5-32 / 5×5-3`，前两层 ReLU，无 BN——与论文一致。唯一偏差：把 bicubic 上采样折进了网络（`self.upscale`），而论文是"网络外先上采样再进网络"。这是合理的自包含变体（也接近 FSRCNN 思路），非缺陷。

**LowLightUNet——诚实。** docstring 明确写"inspired by Retinex-theory... but implemented as a direct learning-based mapping"，**没有声称实现任何具体论文**。标准 U-Net + 编码器/解码器 skip concat，实现正确。这里**不存在保真度问题**，只有第 2/10 轮查过的"训练目标/过曝"问题（那是训练，不是架构）。

**但有个标签造假（第 5 轮）：** 低光 checkpoint 命名为 `lowlight_srcnn_scale2_best.pth`，实际是 U-Net。这不属于架构保真度，属于**命名/叙述的真实性**——懂行的人看文件名会以为用了 SRCNN，打开却是 U-Net。

---

## 6. 本轮结论

> **ML 核心在"结构"层面基本经得起对照**：残差块、BN、PReLU、SRCNN 三层、U-Net skip 都对得上经典论文或主流变体；感知损失的层选择（relu5_3=VGG54）也站得住。
>
> **真正的保真度缺陷只有一个，且是"前提性"的**：感知损失丢了 ImageNet 归一化——这是 SRGAN/Johnson 方法的必需预处理，项目的实现把它整段漏掉了，导致感知项数值上近乎失效（94% 死特征）。
>
> **叙述层面的"宽松标签"有三处**：docstring 写 relu5_4 实为 relu5_3；"SRResNet-style"实际是 SRResNet 骨架 + SRGAN 感知损失的混合；低光文件名写 srcnn 实为 unet。

一句话给懂行的人：**"架构是读得懂的人写的，但感知损失少写了归一化这一行，且几处标签把混合方案说成了单一方案。"** 这比"架构全是抄的"更精确，也比"完全忠实"更诚实。

---

## 7. 硬约束遵守

- 本轮**零代码改动**。`git diff train/ serve/ web/` 无输出。
- 仅新增本文件 `DIAGNOSIS_ROUND15.md`。
- 所有结论基于实读源码（lines 25–149 of `models.py`、46–58 of `train.py`）与 `vgg16.features` 枚举验证（确定性，可复现）。
- `assets/sample_dark.png` 的 `M` 为早期跑 demo 脚本副产物，非本轮改动。

---

## 8. 下一步

三条路你定：
1. **继续扫盲区** — 还剩 `web/app/method/page.tsx` 的 stale 文案（TBD / HF Spaces，第 5 轮提过未深挖）、`web/components/CompareSlider.tsx` 比例错位（第 4 轮）、`gradio_demo.py` 完整边界。
2. **做"修复方案"文档** — 把已确认根因汇总成可执行的修复清单（含"感知损失只补归一化一行"这种高性价比项），纯方案、不动代码。
3. **收网** — 把 15 轮合并成《PixelForge 工程复盘》，主线建议：**"诚实查清了什么、哪些是我自查推翻的、为什么'能力在场但闭环不自洽 + 叙述略松'才是这个项目的真问题"**。仍然只写文档，不动代码。

你说方向。
