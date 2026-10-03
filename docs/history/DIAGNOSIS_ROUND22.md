# ROUND 22 — 全项目审查：漏洞、风险与改善

> 本轮任务不是加功能，而是**逐寸排查**：找出真实的 bug、安全隐患、以及在并发/异常/边界条件下会咬人的地方，然后**修缮**。
> 方法：先用两个探索 agent 通读全部代码，再**逐条亲自复现**——agent 的结论一律当作"待验证的线索"，不当结论。
> 下面每条都标注了是**真 bug**（附复现数据）还是**驳回的误报**。

---

## 一、真实缺陷（已修复）

### 1. PIL 解压炸弹：61 KB 上传 → 385 MB 内存 → 可 OOM 打满服务 🔴 P0

**问题**：`serve/app.py` 原来只用 `MAX_UPLOAD_BYTES`（字节数）限制上传。字节数**完全不能约束解码后的内存**——PNG 的 IHDR 头可以声明任意尺寸，攻击者用一个几十 KB 的文件就能让 `img.convert("RGB")` 按声明的 W×H 分配缓冲区。

**实测**：

| 上传字节数 | 声明尺寸 | 解码峰值 RSS |
|---|---|---|
| 66 B | 8000 × 8000（64 MP） | **385 MB** |

重复发起这个请求就是最廉价的 DoS。

**修复**：把校验**提前到解码之前**。PIL 的 `.size` 只读文件头、不解码，所以先拿到尺寸做像素数硬校验，再决定是否付解码的代价；同时把 `Image.MAX_IMAGE_PIXELS` 对齐到同一条上限，让"头信息与真实尺寸不一致"的情况也在解码时被 PIL 自己拦下。

```python
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS   # 4 MP
...
probe = Image.open(io.BytesIO(raw))   # 只读头，不分配像素缓冲
pw, ph = probe.size
if pw * ph > MAX_INPUT_PIXELS:  -> 413
```

**验证**：`bomb → 413`、正常图 `→ 200`、非图片 `→ 422`。

> 顺带修掉了同类问题里一个**语义错误**：此前把 `Image.MAX_IMAGE_PIXELS` 设好后，`Image.open()` 会先抛 `DecompressionBombError`，导致"文件太大"分支**永远不可达**，用户拿到的是 422（"不是有效图片"）而不是 413（"太大"）。现在按异常类型分流：炸弹 → 413，坏文件 → 422。

---

### 2. 事件循环被 CPU 密集推理阻塞：一个重请求卡死全站 🟠 P1

**问题**：`predict` 声明为 `async def`，但内部直接调用同步的 torch 推理。同步代码跑在事件循环线程上，**推理跑多久，所有其他请求（包括 `/api/health`）就被冻住多久**。

**修复**：`await run_in_threadpool(_predict_sync, await image.read(), task, scale)`——把阻塞负载挪到 worker 线程。

**验证（真实 uvicorn 服务器）**：

```
4 concurrent SRx4 -> [200, 200, 200, 200]
health during load -> [200, 200, 200]      # 负载进行中，健康探针仍秒回
```

---

### 3. 模型缓存的并发竞态：8 个并发请求 → 8 次 `torch.jit.load` 🔴 P0

**这是我自己在第 2 条引入的**——改成线程池后，两个请求可以同时命中缓存读取，检查-写入之间存在窗口。实测：

```
concurrent get_lowlight_model calls into _load: 8  (理想: 1)
```

冷启动时 N 个并发请求会**各自完整加载一遍模型**，把 N 份网络权重同时驻留内存——正是缓存本该消除的浪费。

**修复**：模块级 `threading.Lock` + **锁内二次检查**（double-checked locking）。快速路径（缓存已热）不碰锁。

**验证**：

```
concurrent lowlight loads: 1 (expect 1)
concurrent sr loads:       1 (expect 1)
200 warm concurrent cache hits: 0.006s   # 热路径未被锁串行化
```

---

### 4. 低光权重损坏时**每次请求都重载** 🟠 P1

**问题**：`_lowlight_model = None` 同时承担了两个语义——"没找到权重"和"加载失败"。加载失败后哨兵仍是 `None`，于是**下一个请求又完整 `torch.jit.load` 一遍并再次抛错**。

**实测**：损坏的 `lowlight.pt` → 3 次调用触发 3 次加载（SR 路径同样场景只有 1 次，因为它用 `_sr_cache[key] = None` 正确缓存了失败）。

**修复**：引入 `_LOWLIGHT_UNSET = object()` 作独立哨兵，无论成功/失败都替换掉哨兵。

**验证**：`corrupt file -> 3 calls: 1 load（原为 3）`；`missing file -> 0 loads`；真实模型正常加载。

> **同一 bug 在 `deploy/hf_space/app.py` 里也存在**（`_lowlight_model = None` 且无锁），一并修好了。

---

### 5. `get_sr_model` 漏缓存"未命中"：每请求重复 glob 文件系统 🟡 P2

**问题**：权重文件不存在时，`_sr_cache[key]` 从未被写入，导致**每个请求都重新 `glob()` 磁盘**。

**修复**：把"未命中"也缓存为 `None`（`serve/` 与 `deploy/hf_space/` 同步修）。

---

### 6. 训练不可复现：种子不完整 + DataLoader 无 generator/worker 播种 🟠 P1

**问题**：原来只有一句 `torch.manual_seed(42)`。但——

- `random` / `numpy` 没播种；
- DataLoader 用 `shuffle=True, num_workers=4`，既没传 `generator` 也没传 `worker_init_fn`；worker 内部的 numpy/random 状态**不受父进程影响**（PyTorch 会从父进程 RNG / OS 熵推导 worker 基种子）。

结果是：**相同参数两次训练结果不同**。

**修复**：`seed_everything()` 播种 torch / numpy / random / cuda；`get_dataloader(..., seed=)` 传入 `generator` + `worker_init_fn`（在 worker 内把 numpy/random 也按 worker 种子重置）。

**验证（真实数据集 + 真实 DataLoader，2 workers）**：

```
two seeded runs -> identical batches: True
different seed  -> different batches: True
```

一个诚实的补充：**单靠 `worker_init_fn` 不够**。我单独测过——若父进程 numpy 未播种，即使加了 `worker_init_fn`，同种子两次仍不一致（因为基种子源自父进程 RNG 状态）。**父进程播种 + worker 内再播种**两者齐备才真正闭合。这也是为什么修复必须同时改 `train.py` 和 `datasets.py`。

---

### 7. `metrics.py` 用张量做 `if` 判断 🟡 P2

**问题**：`if mse == 0:` 中 `mse == 0` 得到的是**0 维布尔张量**。它"能用"纯属巧合（单元素），且会在验证循环里**隐式触发一次设备同步**；若哪天 `reduction` 变了就会直接抛。

**修复**：`if mse.item() == 0.0:`，并把 `inf` 建在 `pred.device` 上。

> 保留 `inf` 语义是**刻意**的：`inf > best_psnr` 恒真，不改变最优检查点选择；不要图省事换成"一个很大的有限数"。

---

### 8. `export.py`：`--out model.pt` 直接崩溃 🟡 P2

**问题**：`os.makedirs(os.path.dirname(out))`，当 `out` 不含目录时 `dirname` 返回 `''`，而 `os.makedirs('')` 抛 `FileNotFoundError`。

**实测**：`dirname('model.pt') = ''` → `RAISES FileNotFoundError`。

**修复**：`if out_dir: os.makedirs(...)`。

---

### 9. `gradio_demo.py`：文档里的启动方式直接 ImportError，且缺低光守卫 🟠 P1

**问题 A**：文档写 `python -m serve.gradio_demo`，但代码是裸的 `from classical import ...`——包方式运行必崩。

**问题 B**：这是**第三个**推理入口，却没有 Round 20/21 的曝光门控和输出守卫，所以它**仍然会"越增强越暗"**——修好的逻辑只覆盖了 `app.py` / Streamlit。

**修复**：`try: from .classical import ... except ImportError: from classical import ...` 同时支持两种入口；并补齐门控 + 输出守卫。

---

### 10. e2e 测试：断言泄漏进程 + 低光路径被静默跳过 🟡 P2

**问题 A**：用 `assert` 做校验，抛出的 `AssertionError` 会**越过 `_cleanup()`**，把 uvicorn(8000) 和 pnpm(3000) 两个子进程留成孤儿占住端口；而且 `python -O` 下断言会被静默禁用。

**问题 B**：`if not failures: run_flow("lowlight", ...)`——只要 SR 出一次失败，**整个低光测试被跳过**，恰恰在最需要测的时候不测。

**修复**：失败改为追加进 `failures` 列表；`main()` 用 `try/finally` 保证 `_cleanup()` 在任何退出路径都执行；低光**无条件执行**；顺手清掉未用 import 与死代码。

---

### 11. 前端：两个真实交互缺陷 🟡 P2

**A. `CompareSlider` 用 `object-cover` 做对比图层**：底层"after"是 `block w-full`（由自身宽高比撑开容器），覆盖层却是 `absolute inset-0 object-cover`。当两图宽高比不同（SR 输出与 before 尺寸不一致时会发生），`object-cover` 会**裁剪**覆盖层——滑动对比就变成**错位比对**，而组件注释还声称"perfectly aligned regardless of size"。

修复：改用 `object-fill`，与后端"强制 before/after 同尺寸"保持一致；注释写清真实约束。

**B. `dragging` 卡死**：只在 `onPointerUp` 复位，指针在元素外抬起、或浏览器取消手势（触摸滚动接管）时，`dragging` 一直是 `true`，滑块会**继续吸附光标**。

修复：补 `onPointerCancel` / `onPointerLeave` / `onLostPointerCapture`，加 `touch-none`。

**C. 切换任务/倍率后结果未清空**：SR 出图后切到 Low-Light，旧结果仍在屏上，却套上了低光的标签和徽章——**张冠李戴**。修复：`pickTask` / `pickScale` 里清空 `result`。

---

## 二、驳回的误报（查过，不是问题）

诚实记录，避免"为了改而改"：

| 线索 | 结论 |
|---|---|
| `web/app/layout.tsx` 用 `lang="en"` 但内容是中文 | **误报**。逐文件扫描 CJK：`page.tsx`/`method/page.tsx`/`layout.tsx`/`globals.css` 中文串数均为 **0**。Next.js 站点是纯英文；中文界面在 Streamlit 那份（独立的部署面）。`lang="en"` 正确。 |
| `np.pad(mode="reflect")` 在小图上会因 pad ≥ 轴长而抛错 | **实测不成立**。`(5,5)`/`(1,1)`/`(2,2)`/`(33,33)` 全部正常；连 `len=1,pad=31`、`len=3,pad=29` 等极端组合也不抛。无需改。 |
| `classical.py` 有问题 | 干净，无发现。 |
| Streamlit 版模型加载有并发竞态 | **不适用**。它用 `@st.cache_resource`（Streamlit 自带的资源缓存），且是独立进程模型，不存在本文第 3 条那样的竞态。 |

---

## 三、改动清单

| 文件 | 改动 |
|---|---|
| `serve/app.py` | 解码前像素校验（防解压炸弹）；413/422 语义分流；`run_in_threadpool` 卸载阻塞推理 |
| `serve/model_loader.py` | `_LOWLIGHT_UNSET` 哨兵（失败也缓存）；`_MODEL_LOCK` 双检锁；`get_sr_model` 缓存未命中 |
| `serve/gradio_demo.py` | 双入口导入兼容；补齐低光门控 + 输出守卫 |
| `deploy/hf_space/app.py` | 同步上述哨兵 + 锁 + 缓存未命中修复 |
| `train/datasets.py` | `seed` 参数 → `generator` + `worker_init_fn`；`pin_memory` 仅 CUDA 时开 |
| `train/train.py` | `seed_everything`；VGG 改用 `VGG16_Weights` 枚举（旧 `pretrained=True` 已废弃） |
| `train/metrics.py` | `mse.item() == 0.0`，`inf` 建在正确 device |
| `train/export.py` | 无目录 `--out` 不再崩溃 |
| `tests/e2e/e2e.py` | 去断言式校验；`finally` 保证清理；低光无条件跑 |
| `web/components/CompareSlider.tsx` | `object-fill` 对齐；指针取消/离开复位 |
| `web/app/page.tsx` | 切换任务/倍率清空旧结果 |

## 四、回归验证

| 检查 | 结果 |
|---|---|
| `pyflakes` 全量（serve/train/scripts/tests/deploy） | **ALL CLEAN** |
| `train.tests.run_tests` | **20/20 passed** |
| API 全路径（健康/正常×3/非法任务/坏文件/炸弹/超大字节） | **8/8 passed** |
| 真实服务器并发（4×SR×4 + 健康探针） | 全部 200，健康探针未被阻塞 |
| 可复现性（真实 DataLoader，2 workers） | 同种子批次逐字节相同 |
| 模型加载并发（8 线程） | 各 1 次加载（原为 8） |

## 五、未处理 / 已知遗留

- **API 全路径冒烟测试已内联验证，但尚未固化为 `pytest` 用例**。这是"扩展"而非"修缮"，且本轮限定不做扩展；建议后续单独一轮补上，避免这些边角回归。
- **`CompareSlider` 的对齐修复依赖后端保证 before/after 同尺寸**。若将来有前端自行拼接 before/after 的路径，需一并维持该不变量——注释里已写明。
- **模型加载失败目前只 `warnings.warn`**。生产环境若要可观测性，应接日志/metrics；属扩展范畴。
- **`torch.jit.trace` 用固定 dummy 输入**（`export.py`）会把动态尺寸固化。当前低光/ SR 路径都走 `_pad_to_multiple` 规避，未出问题；若将来支持任意尺寸输入需改为 `torch.jit.script` 或动态 shape trace。
