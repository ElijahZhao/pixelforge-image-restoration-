# PixelForge 深度审计 · 第 14 轮｜serve / web 的异常处理与契约边界

> 审计方法：只读探查 + 端到端代码通读 + 量化实验 + 自我证伪。
> 本轮聚焦两条此前未深挖的线：`serve/app.py` 的异常处理边界，以及 `web/` 前端与 FastAPI 的契约一致性。
> 注：第 4、5、6、9 轮已分别触及 CompareSlider 比例错位、method 页 stale、DoS、gradio 语义，本轮在"边界防护"和"契约闭环"层面补全。

---

## 0. 自我修正（先认一个误判）

本轮开头我在**项目根目录**找 `next.config.mjs`（`cat next.config.mjs` → 报错），差点下"生产代理缺失、部署断裂"的结论。实读发现：文件在 **`web/next.config.mjs`**，且 `rewrites()`（第 4–12 行）逻辑正确——未设 `NEXT_PUBLIC_API_URL` 时把 `/api/*` 代理到 `http://localhost:8000/api/*`。

**纠正**：前端→后端的代理契约是成立的，不是断裂。这又一次印证本审计的方法论——**靠实读压住路径误判，而不是靠"在根目录没找到"下结论**。

---

## 1. DoS 无上限（确定，本轮量化补全）

`serve/app.py:80` `await image.read()` 把整个上传读进内存；`predict_sr` 输出再 `_img_to_b64` 成 base64 返回。**全程无** `Content-Length` 检查、无最大尺寸、无超时。

量化放大系数（SR ×4，base64 ≈ 3 字节/px × 1.33）：

| 输入 | 输出像素 | base64 响应 |
|---|---|---|
| 512×512 | 4 MP | ~17 MB |
| 2000×1500 | 48 MP | ~192 MB（第 6 轮实测 39s） |
| 4000×3000 | 192 MP | **~766 MB** |
| 6000×4000 | 384 MP | **~1.5 GB** |

输入完全由客户端决定，**输出随输入线性放大、无上限**。一次大请求即可让 worker OOM。

> 复现（读代码即可确认，无需起服务）：`grep -n "image.read\|max_size\|Content-Length\|timeout" serve/app.py` → 零命中。

---

## 2. CORS 默认 `*` + 无认证/无限速（确定）

`serve/app.py:35-38`：

```python
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")   # 默认通配
```

默认 `allow_origins=["*"]`、`allow_methods=["GET","POST"]`、`allow_headers=["*"]`。再叠加：
- 无身份认证；
- 无速率限制；
- 第 1 条的 DoS 无上限；

=> **一个开放、无认证、无限制、可被任意域调用的推理端点**。任何人都能无成本地打满你的算力（尤其部署在按量计费的 GPU/Serverless 上）。文档注释说"生产设 `ALLOWED_ORIGINS`"，但**默认值是通配**——这是"默认不安全"的典型。

---

## 3. 前端无超时 / 无 AbortController（确定）

`web/lib/api.ts:26`：

```ts
const res = await fetch(endpoint, { method: "POST", body: fd });   // 无 signal
```

- 后端因大图卡住（39s+）或 OOM 时，前端 `loading` 永久转，**用户无法取消、无超时反馈**；
- 错误分支只 `throw new Error(status + msg)`，把后端原始 JSON 文本当消息显示（非结构化）；
- `res.json()` 对**超大响应无校验**——若后端真返回 GB 级 base64，浏览器解析 JSON 即卡死/OOM。

前端是 DoS 的**放大器**（无客户端护栏，把任意大请求无脑打到后端，再无脑解析响应）。

---

## 4. 前端无上传大小限制（确定）

`web/app/page.tsx:99-104` 的 `<input type="file" accept="image/*">` 只限**类型**，不限**大小**；`api.ts` 直接 `fd.append("image", file)`。客户端不挡超大图，全压后端。

---

## 5. `classical.py:63` 的 `raise ValueError(task)` 是死代码（确定）

`run_classical` 的两个分支（`sr`/`lowlight`）之后有 `raise ValueError(task)`（第 63 行）。
但调用方 `serve/app.py:89/96` 只在 `task ∈ {"sr","lowlight"}` 时调用它——而 `app.py:72` **已经**把 `task` 限制在此集合内。

=> 该 `raise` **永远不可达**。两层校验冗余，第二层是幽灵分支。懂行的人看代码会立刻注意到这个矛盾（要么 app.py 不该先过滤，要么 classical.py 不该再防）。

---

## 6. `scale` 校验静默重置（低优先，确定）

`serve/app.py:76-77`：

```python
if scale not in (2, 4):
    scale = 2          # 非法值静默改为 2，不报错、不告知
```

前端 UI 允许选 2×/4×，若传非法值（如 3），后端悄悄用 2× 跑，客户端无感知。不利于调试，但非安全缺陷。

---

## 7. 模型推理异常冒泡为 500（确定）

`serve/app.py:79-84` 的 `try` **只包 `Image.open`**。`predict_sr` / `predict_lowlight`（第 87/94 行）在 `try` 之外——模型推理若崩（OOM、尺寸不整除、traced 图与输入通道不符），异常直接冒泡成 500，没有结构化错误体。前端 `api.ts` 能显示状态码+文本，但不优雅、不利于排障。

---

## 8. 部署契约的隐性陷阱（确定，值得强调）

`web/next.config.mjs:9-11`：仅当**未设** `NEXT_PUBLIC_API_URL` 时才走 same-origin 代理到 `localhost:8000`。

含义：在 Vercel/HF 等 **serverless/无 localhost 服务**的部署下，若**不设** `NEXT_PUBLIC_API_URL`，`/api/predict` 会被代理到一个不存在的 `localhost:8000` → **生产 404/连不上**。DEPLOY.md 说了"生产要设该变量"，但没强调**"不设就断"**。这是部署契约里一个会咬人的隐性前提。

---

## 9. SR "before" 语义错位（延续第 5 轮，本轮在契约层确认）

`serve/app.py:91-92` 与 `web/app/page.tsx:142` 协同：slider 左侧 `before = original.resize(w*scale, h*scale)`（放大模糊版），标签 `"Original (upscaled)"`。用户滑到的"原图"并非真实输入，而是**插值放大版**。模型承诺"×N 超分"，但对比基线却是"原图再插值 N 倍"——这模糊了"模型增益 vs 插值基线"的边界（第 10 轮已证模型实际增益为负，此语义错位进一步掩盖了这一点）。

---

## 10. 本轮结论

> `serve`/`web` 这一闭环的缺陷，与前几轮的"训练—评估—验证"不自洽（第 13 轮）是**同类病灶**在不同层的投影：
>
> - **边界防护全面缺失**（上传无上限、CORS 通配、无认证限速、前端无超时/无大小校验）——资源与安全的护栏不存在；
> - **错误处理不结构化**（异常冒泡 500、`raise` 死代码、`scale` 静默重置）——出错时既无清晰信号也无优雅降级；
> - **契约前提隐性**（生产不设 env 就断、before 语义错位）——把"能跑"当作"契约完整"。
>
> 能力（代理配置、classical 兜底、health 自检）都在，但**护栏层与契约层是后补的、不完整的**。这正是"懂行的人盯着看"会发现的第二层——不仅模型学不对，连"把模型安全地交给用户"这件事也还没闭环。

## 11. 诚实校准

- 误判并纠正：`next.config.mjs` 不在根目录、实际在 `web/` 且正确（差点冤枉）。
- 量化补全：DoS 放大系数（4000×3000→766MB，6000×4000→1.5GB）。
- 延续第 9/11/12/13 轮传统：**先立结论、再实读证伪**。

## 12. 复现命令（只读）

```bash
# 1. 确认无任何大小/超时/限速防护
grep -n "image.read\|Content-Length\|max_size\|timeout\|rate" serve/app.py   # 零命中

# 2. CORS 默认通配
grep -n "ALLOWED_ORIGINS" serve/app.py                                       # 默认 "*"

# 3. 前端无超时
grep -n "AbortController\|signal\|timeout" web/lib/api.ts                    # 零命中

# 4. 死代码确认：task 已被 app.py:72 过滤，classical.py:63 不可达
grep -n "task not in\|raise ValueError" serve/app.py serve/classical.py
```

---

*本审计严格遵守"只读、不改代码"约束。`git status` 仅新增本报告；`train/`、`serve/`、`web/` 零改动（本轮未修改任何文件）。*
