# ROUND 23 — 第二轮全项目审查：部署契约、测试有效性、死代码

> 接上篇。本轮继续逐项排查，重点补上此前没细读的面：895 行的 Streamlit 应用、HF Space 全文件、method 页、Next 配置、`scripts/` 工具、以及**测试本身是否真的有效**。
> 方法：每条结论均实际复现后再记录；被驳回的误报同样留档。

---

## 一、真实缺陷（已修复）

### 1. `streamlit>=1.30` 会让整个结果区崩溃 🔴 P0（本轮最重要）

**问题**：`deploy/streamlit/requirements.txt` 写 `streamlit>=1.30`。但代码里有 **8 处** `st.image(..., width="stretch")`：

```python
st.image(img, width="stretch")     # 835, 840, 845, 858, 862, 867, 874, 878
```

`width=` 接受字符串 `"stretch"` 是**新 API**。若依赖解析到 1.30–1.48，`st.image` 会拒绝该参数，**用户一上传图片，整个结果面板就抛异常**。

**不是靠猜——我下载了三个版本的 wheel 直接比对源码签名**：

| Streamlit 版本 | `st.image` 的 width 签名 | `"stretch"` 可用？ |
|---|---|---|
| 1.40.0 | `width: int \| None = None` | ❌ 只收 int |
| 1.48.0 | `width: int \| None = None` | ❌ 只收 int |
| **1.49.0** | **`width: Width = "content"`** | ✅ |

1.40 只是给 `st.image` 加了 `use_container_width`；`width: Width = "content"` 这种能收 `"stretch"` 的写法**到 1.49.0 才出现**（对应上游 PR #11952，2025-08-14 合入）。

**修复**：`streamlit>=1.49`，并在 requirements 里写明**为什么不能降回去**。

---

### 2. `_ml()` 每次调用都重新加载模块 → 模型加载两次 🟡 P2

**问题**：`scripts/eval_baseline.py` 的 `_ml()` 用 `spec_from_file_location` + `exec_module` 按路径加载 `serve/model_loader.py`。

它**没有注册进 `sys.modules`**，所以每次调用都执行出一个**全新的模块对象**，带着自己独立的缓存字典：

```
module __name__: ml
in sys.modules? False
same as serve.model_loader? False
cache dicts shared? False
```

`_ml()` 在 `eval_sr`(L75) 和 `eval_lowlight`(L121) 各调一次——**同一个进程里跑 SR + 低光就会把两个模型各加载两遍**，且两份缓存互不可见。

**修复**：改成正常 `from serve import model_loader` 并 memoise。

**验证**：`same module object: True`｜`shared with package cache: True`。

---

### 3. 中文模式下浏览器标签页标题仍是英文 🟡 P2

**问题**：`TEXTS["zh"]["page_title"]` = "PixelForge · 图像修复" 定义了却**从没被用过**——`st.set_page_config` 里硬编码了英文字符串。中文用户看到的中文界面，标签页却是英文。

（这类"定义了没用"的键通常无害，但这里对应的是一处**真实的用户可见不一致**。）

**修复**：`set_page_config` 读当前语言。注意顺序约束——`set_page_config` 必须是脚本里第一个 Streamlit 调用，而 `session_state` 默认值在它**之后**才 `setdefault`，所以改用 `.get("lang", "en")` 兜底。

**验证**：`en/zh × dark/light` 四种组合 AppTest 全部无异常。

---

### 4. 测试"假通过"：断言的是测试自己的常量 🟠 P1

**问题**：`test_loss_weights_are_explicit_and_sane` 里把权重**硬编码**成 `1.0` / `0.006`：

```python
pix_contrib, per_contrib = 1.0 * pix, 0.006 * per   # 从注释抄来的
```

它根本没读 `train.py` 的真实默认值。**若有人把 `--w_pixel` 默认值改坏，这个测试照样 PASS**——正是"代码坏了测试还是绿的"。

**修复**：在 `train.py` 暴露 `DEFAULT_W_PIXEL` / `DEFAULT_W_PERCEP` 常量（`parse_args` 也引用它们），测试改为读取真实常量；并补一条"两个权重必须都为正"的断言。

**变异测试证明有效**（注入缺陷 → 测试必须失败）：

| 注入 | 结果 |
|---|---|
| `DEFAULT_W_PIXEL = 0.0` | ✅ 被捕获（ratio=5.3e8） |
| `DEFAULT_W_PIXEL = 0.001` | ✅ 被捕获 |
| `DEFAULT_W_PERCEP = 0.0` | ✅ 被捕获 |
| `DEFAULT_W_PERCEP = 1.0` | ⚪ 仍通过 |

最后一条**我诚实保留**：该情况下数值上像素项仍占优，测试声明的性质（"像素项没被压垮"）**确实成立**，为它加断言反而会过拟合一个良性改动。

> 顺带发现并修掉了一个我自己在本轮引入的语法错误：一次编辑把 `objective = (...)` 和下一行 `with open(...)` 粘成了一行，导致 20/20 掉到 19/20。已修回。

---

### 5. `test_batch_invariance` 平凡为真，无法证伪任何缺陷 🟡 P2

**问题**：原测试用 `torch.zeros` 输入，只断言 `y[0] == y[1]`。对**任何确定性模块**都成立——我实测**连一个单层 Conv2d 的 stub 都能"通过"**。它的注释自己也承认是 "trivial check"。

**修复**：换成三条**可以失败**的性质——
1. eval 模式下对同一输入确定性；
2. **不同输入必须给出不同输出**（防"网络塌成常量/死 ReLU/权重归零"）；
3. 批量推理结果 == 逐样本推理结果（这才是"batch invariance"的本义）。

**变异验证**：把模型改成返回常量 `0.5`，新断言**确实失败**（即真的能捕获塌缩网络）。

顺带补齐了两个 scale4 测试缺失的 `_assert_range`（scale2 版本有、scale4 没有，覆盖不一致）。

---

### 6. 两个 VGG 测试依赖联网，离线 CI 会"假失败" 🟡 P2

**问题**：测试用 `tvm.vgg16(pretrained=True)`——既是被废弃的 API，又会**触发 ImageNet 权重下载**。干净 CI 上无网络时，测试**报错**（看起来像真回归），而它其实是环境缺失。

**修复**：
- 改用 `VGG16_Weights.IMAGENET1K_V1`（与 `train.py` 一致，pin 住具体权重）；
- 抽 `_vgg_features()` helper，权重取不到时抛 `unittest.SkipTest`；
- 给 `run_tests.py` 加 **SKIP 支持**（原来任何异常都算 FAIL）。

**验证**：模拟下载失败 → 两个测试都正确走 SKIP 分支，而非 FAIL。

---

### 7. 死代码清理（已核实后删除）

| 位置 | 内容 | 核实 |
|---|---|---|
| `streamlit_app.py` | `--pf-pink` 两个主题都有，全文件 **0 处引用** | 删。注意它和 `--pf-btn-shadow-hover` 的 rgba **不是同一个颜色**（亮色主题下分别为 #be185d vs rgb(219,39,119)），所以不能"合并"，只能删 |
| `web/tailwind.config.js` | `panel` 色、`fontFamily.mono` 全仓库 0 处使用 | 删 |
| `streamlit_app.py` | `lang_label` | 改为**使用**它（语言选择器的 label 原来硬编码） |

---

## 二、驳回的误报（查过，不是问题）

| 线索 | 结论 |
|---|---|
| `TEXTS` 的 `en`/`zh` 键集不一致 | **误报**。AST 精确比对：两边各 **34 键，完全一致**；代码引用 32 个键全部已定义 |
| CSS 中有未定义的变量引用 | **误报**。所有 `var(--pf-*)` 均能找到定义；`--pf-text*` 只在亮色表定义、也只被亮色 fixes 使用，逻辑自洽 |
| CSS 把侧栏折叠按钮隐藏了 | **误报**。ROUND 22 已修：现在显式保留 `stExpandSidebarButton` / `stSidebarCollapsedControl` 等，只隐藏叶子装饰元素 |
| `.gitignore` 漏忽略或误忽略大文件 | **误报**。`git ls-files` 中无 `.pyc`/`.zip`/screenshots；`models/*.pt` 是**有意提交**（各约 2.3/5 MB） |
| `models.py` 架构有缺陷 | **误报**。逐项核对：ResidualBlock 残差、SRGenerator 长跳连接、U-Net 三段通道拼接（3→32→64→128→…→35→3）全部对得上；`PixelShuffle(scale)` 与 export/trace 契约一致 |
| `build_model("lowlight", scale=...)` 忽略 scale | **设计使然**（LLIE 不改变分辨率），非缺陷 |
| `web/package.json`、`DEPLOY.md` | 无发现 |

---

## 三、改动清单

| 文件 | 改动 |
|---|---|
| `deploy/streamlit/requirements.txt` | `streamlit>=1.30` → **`>=1.49`**（含原因注释） |
| `scripts/eval_baseline.py` | `_ml()` 改为 memoise 的包导入，共享模型缓存 |
| `deploy/streamlit/streamlit_app.py` | 标签页标题跟随语言；语言选择器用 `lang_label`；删死变量 `--pf-pink` |
| `train/train.py` | 暴露 `DEFAULT_W_PIXEL` / `DEFAULT_W_PERCEP` |
| `train/tests/test_correctness.py` | 权重断言读真实常量；VGG 改用 weights 枚举 + 离线 SKIP |
| `train/tests/test_models.py` | 重写 `test_batch_invariance` 为可证伪的三性质；补 scale4 range 断言 |
| `train/tests/run_tests.py` | 支持 SKIP 报告 |
| `web/tailwind.config.js` | 删未使用的 `panel` / `fontFamily.mono` |

## 四、回归验证

| 检查 | 结果 |
|---|---|
| `pyflakes` 全量（serve/train/scripts/tests/deploy） | **ALL CLEAN** |
| `train.tests.run_tests` | **20/20 passed** |
| 变异测试（4 组权重注入） | 3/4 捕获，1 组确认为良性 |
| 模型塌缩变异（常量输出） | 新断言捕获 ✅ |
| 离线模拟（VGG 下载失败） | 正确 SKIP ✅ |
| API 全路径 | **8/8 passed** |
| Streamlit AppTest（en/zh × dark/light） | 4/4 无异常 |
| `tsc --noEmit` + tailwind 配置校验 | 通过 |

## 五、未处理 / 遗留

- **`eval_baseline.py` 的指标口径**：它刻意**不过曝光门控**（评估就该看到未门控的真实数字），这是正确的。但与线上行为不同，已在注释中说明。
- **requirements 无上界 pin**：三个 requirements 都只给下界。对作品集项目可接受；若要长期可复现应加上界或锁文件——属扩展范畴。
- **`pixelforge_fixed_train.zip`（6.7 MB）仍留在工作区**：已被 `.gitignore` 忽略，未提交。属清理项，非缺陷。
- 上一轮遗留的"API 冒烟测试未固化为 pytest 用例"仍然存在——仍归为扩展，本轮未做。
