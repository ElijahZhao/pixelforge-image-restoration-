# PixelForge 深度审计 · 第 16 轮

**主题：对外展示与叙述层的诚实度（项目展示的东西，有没有诚实地反映模型真实能力？）**
**方法：实读 `web/components/CompareSlider.tsx`、`web/app/page.tsx`、`web/app/method/page.tsx`、`serve/gradio_demo.py` + 加载逻辑实测**
**范围：仅分析，不改动代码**

---

## 0. 本轮立场：先纠正一个旧结论

第 5 轮我写过 "`web/app/method/page.tsx` completely stale (TBD values, HF Spaces)"。本轮实读后**必须纠正**：该页的 **TBD 占位反而是诚实的**——它明确写 "no placeholder numbers are shown as if they were real results"。所以"完全 stale"是过了。**真正的问题不是 TBD 本身，而是 TBD 叙述与"已提交权重 + 绿色徽章"之间的叙述矛盾**。这一轮把这条线做透，并区分"已经诚实的"和"仍不诚实的"。

---

## 1. 叙述与实况的矛盾（本轮最深的发现）

`method/page.tsx` 的两条核心叙述：

- 第 52–54 行：「Until real weights are trained, the 'ours' rows are honestly marked **TBD** — no placeholder numbers are shown as if they were real results.」
- 第 144–145 行：「The service automatically **falls back to classical baselines when no trained weights are present**, so the demo always runs.」

**但实况是：权重已经真实提交在 `serve/models/`**（`sr_generator_scale4.pt` 5.0MB、`lowlight.pt` 2.3MB，`git ls-files` 确认被跟踪）。而且：

- 第 10 轮实测：这些权重的真实表现**劣于经典基线**（SR 22.06dB vs bicubic 33.25dB；低光 12.68dB vs gamma 19.62dB）。
- 主页面 `page.tsx:130–135` 给这些权重打的是**绿色 "Trained PyTorch model" 徽章**，不披露它们弱于基线。

> **矛盾的本质**：方法页用"诚实的 TBD / 无权重回退基线"讲了一个"我们还没训出结果、但优雅兜底"的故事；而实际发布的产物是"带着已知坏结果的训练权重 + 绿色 ML 徽章"。懂行的人把方法页和 demo 对照看，会立刻发现：**叙述说没有结果，产品却在自信地展示一个比基线还差的'训练好的模型'。**

这不是 TBD 的错（TBD 是诚实的），是**叙述层与部署层脱节**——和前面第 13 轮"训练/评估目标脱钩"、第 14 轮"契约层不完整"是同一类病灶：**能做的都在做，但各层之间讲的故事对不上**。

---

## 2. `CompareSlider.tsx`：潜在比例脆弱 + 误导注释（非活动缺陷）

实读源码（行 44–59）：

- **After（底层）**：`<img ... className="block w-full" />` —— 无高度约束，容器高度由 After 的自然宽高比决定。
- **Before（覆盖层）**：`<img ... className="absolute inset-0 h-full w-full object-cover" />` —— 绝对定位填满容器，`object-cover` 意味**若 before 与容器（=After 尺寸）比例不同，会被裁剪**。

组件注释（第 13 行）声称「Uses clip-path so both layers stay **perfectly aligned regardless of size**」——**这句是错的/夸大的**：它只在 before/after **同尺寸**时对齐；一旦比例不同，before 会被 `object-cover` 裁切，破坏对齐。

**校准（保持公正）**：在**当前 demo 的调用方式下，bug 是休眠的**——`page.tsx:142–143` 传入的 `before`/`after` 都是同一输出尺寸（SR 的 before=原图插值到输出尺寸、after=模型输出；低光两者同尺寸），所以 `object-cover` 不会裁切。**结论：组件脆弱、注释夸大，但当前 demo 未触发视觉破坏。** 这是"潜在缺陷 + 误导性注释"，不是"滑块已坏"。

---

## 3. Next.js 的 SR "before" 语义（确认第 5/10 轮）

`page.tsx:142`：`beforeLabel={task === "sr" ? "Original (upscaled)" : "Low-light"}`。

- SR 的 before 是 **"原图再插值 N 倍"**（api 返回的 `original upscaled to output size`），**不是真原图**。
- 由于模型输出（SR）实测**劣于 bicubic**（第 10 轮 22 vs 33dB），滑块把"bicubic 上采样"标成 before、"模型输出"标成 after，相当于**把一个比基线还差的模型包装成"在增强"**。视觉上若模型锯齿/模糊，观众会以为是"原图差、模型救回来了"，实则是模型不如它正在对比的那个基线。

这属于**语义错位掩盖负增益**，确认并强化第 5/10 轮结论。

---

## 4. `gradio_demo.py` 完整边界

实读 `serve/gradio_demo.py`（30 行），三条发现：

**(a) SR 默认 scale=2 → 静默经典基线（强指控，已实测）。**
- `gr.Radio(["2","4"], value="2")` —— 默认 2×。
- 实测 `get_sr_model(2)=False`、`get_sr_model(4)=True`（只有 scale4 权重被提交）。
- 因此 `process` 中 `predict_sr(image, 2)` 返回 `None` → `or run_classical(image, "sr", 2)` → **默认路径跑的是 bicubic 经典基线，不是 ML 模型**。
- 描述写「Uses trained models if exported, otherwise classical baselines」——技术上"会回退"没错，但**默认体验是经典基线，且用户无任何提示自己看的是 bicubic 而非 ML**。这与第 4/6 轮的 Streamlit ×2 陷阱是同一类（只有 scale4 权重却默认 2×）。
- 旁证：`model_loader.py:4` 注释引用 `serve/models/sr_srcnn_scale2.pt`，**该文件从未生成**——代码预期一个从未训练的 scale2 权重。

**(b) "After" 含第 10 轮的隐藏 4× 假放大。**
- `predict_sr` 返回 `out.resize((img.width*scale, img.height*scale), BICUBIC)`——模型只产出 `原图/scale` 的分辨率，最后 PIL 插值放大 `scale` 倍（第 10 轮证实）。
- Gradio 的 `process` 返回 `image`（真原图，小）作 before、`out`（被插值放大的大图）作 after。于是**展示上把插值细节当成模型能力**，且 before/after 尺寸悬殊（如 512 vs 2048），视觉对比失真。

**(c) 无大小限制 / DoS。** 与 `serve/app.py` 同病（第 14 轮）：接受任意尺寸图片，`predict_sr` 的 `scale` 倍放大 → 响应无上限。

---

## 5. 标签过度 / 误导（叙述层小项）

| 位置 | 文案 | 实况 | 判定 |
|---|---|---|---|
| `page.tsx:171` footer | "Models: **SRCNN** / SRResNet ... and a U-Net" | 提交的 SR 权重只有 `sr_generator_scale4.pt`（SRResNet 式 generator）；**SRCNN 无提交权重**（第 13 轮） | ⚠️ 暗示部署了 SRCNN，实际没有 |
| `method/page.tsx:135-136` | 低光命令 "--model **srcnn** --scale 2" | `build_model('lowlight')` 永远返回 U-Net，`--model` 是死参数（第 5 轮） | ❌ 暗示低光用 SRCNN，实为 U-Net |
| `method/page.tsx:45` | U-Net "+ residual learning (predicts an enhancement residual)" | U-Net forward 学**绝对映射** + skip concat（第 12 轮），非残差预测 | ⚠️ 描述不准确 |
| `method/page.tsx:142-143` | 部署 "Hugging Face Spaces / small VPS" | DEPLOY.md 描述已废弃的 Vercel+HF 方案（第 5 轮） | ⚠️ 可能 stale |

---

## 6. 自我纠正清单（本轮校准）

1. **第 5 轮 "method page 完全 stale" → 纠正**：TBD 占位本身是诚实的，问题在 TBD 叙述与已提交权重/绿色徽章的矛盾，而非 TBD 本身。
2. **第 4 轮 "CompareSlider 比例错位导致视觉破坏" → 精确化**：组件脆弱、注释夸大，但当前调用传入同尺寸图，**bug 休眠、未实际破坏视觉**。

两处都遵循本审计传统：先立结论、再实读/实测校准，宁可推翻自己。

---

## 7. 本轮结论

> **展示与叙述层的诚实度，介于"已经诚实"和"仍不诚实"之间：**
> - **已经诚实的**：method 页的 TBD 占位、明确声明"不把占位当真实结果"。
> - **仍不诚实的（核心）**：叙述层讲"无权重/回退基线"，部署层却提交已知弱于基线的权重、并打绿色 "Trained PyTorch model" 徽章，把负增益藏在自信的 UI 后面；SR 的 before 语义把"比基线差"包装成"在增强"；Gradio 默认 2× 静默跑经典基线却描述成"用训练模型"。
>
> **一句话给懂行的人**："它诚实地在表格里写了 TBD，却把已知比基线还差的训练权重包装成绿色'训练好的模型'发给用户，且 SR 对比滑块的 before 不是真原图。诚实停在表格，没停在产品。"

---

## 8. 结论依据

- 关键指控均经实测：`git ls-files serve/models/`、`get_sr_model(2)=False / (4)=True`、`predict_sr` 的 `resize(...*scale)` 逻辑（行 63）、`gradio_demo.py` 默认 `value="2"`。
- `assets/sample_dark.png` 的 `M` 为早期跑 demo 副产物，与本轮无关。

---

## 9. 遗留盲区

- `web/app/method/page.tsx` 的部署段（HF Spaces）是否 stale；
- `serve/app.py` 完整异常处理边界（第 14 轮提过但未逐行）；
- `README.md` 整体叙述与实况的对账。
