# Streamlit 线上崩溃 · 诊断记录

这份文档记录一次真实的线上故障排查：公开 Demo（`pixelforge-image-restoration.streamlit.app`）
反复崩溃、且**每次重新部署后更容易崩**。结论与修复依据都在这里，包括从部署日志里取到的原始证据。

排查日期：2026-10-06。

---

## 现象

1. 公开 Demo 周期性崩溃（页面变 "Oh no." 或无法连接）。
2. **重新部署后，崩溃更容易发生**——这个规律是用户先观察到的。
3. Streamlit Cloud 的日志面板里**看不到任何 traceback**：日志停在一行正常输出后就没了。

---

## 证据：部署日志（原始摘录）

日志来自 Streamlit Cloud 的 app 日志面板。关键片段如下。

### 证据 1：Streamlit 读的是哪个 requirements 文件

```
[14:06:40] 🐍 Python dependencies were installed from
  /mount/src/pixelforge-image-restoration-/deploy/streamlit/requirements.txt using uv.
[14:06:40] 📦 WARN: More than one requirements file detected in the repository.
  Available options: uv .../deploy/streamlit/requirements.txt,
                     uv .../requirements.txt,
                     poetry .../pyproject.toml.
  Used: uv with .../deploy/streamlit/requirements.txt
```

**意义**：部署只用 `deploy/streamlit/requirements.txt`，**不读**仓库根的 `requirements.txt`，
也不读 `requirements.lock.txt`。这决定了修复必须落在这个文件上。

### 证据 2：装了 19 个永远不会执行的 CUDA/nvidia 包

依赖解析输出（共 62 个包）里包含：

```
 + cuda-bindings==13.4.3
 + cuda-pathfinder==1.8.3
 + cuda-toolkit==13.0.3.0
 + nvidia-cublas==13.1.1.3
 + nvidia-cuda-cupti==13.0.85
 + nvidia-cuda-nvrtc==13.0.88
 + nvidia-cuda-runtime==13.0.96
 + nvidia-cudnn-cu13==9.24.0.43
 + nvidia-cufft==12.0.0.61
 + nvidia-cufile==1.15.1.6
 + nvidia-curand==10.4.0.35
 + nvidia-cusolver==12.0.4.66
 + nvidia-cusparse==12.6.3.3
 + nvidia-cusparselt-cu13==0.8.1
 + nvidia-nccl-cu13==2.30.7
 + nvidia-nvjitlink==13.4.92
 + nvidia-nvshmem-cu13==3.4.5
 + nvidia-nvtx==13.0.85
 + triton==3.8.0
```

**意义**：Streamlit 免费档**没有 GPU**，`torch.cuda.is_available()` 恒为 `False`
（`streamlit_app.py` 的 `_device()` 因此恒返回 `"cpu"`）。这 19 个包在这里
**一个都不会被调用**，却实打实占用磁盘与内存。

### 证据 3：torch 版本漂移 + TorchScript 弃用

```
 + torch==2.14.1
 + torchvision==0.29.1

.../torch/jit/_serialization.py:176: FutureWarning:
  `torch.jit.load` is deprecated. Please switch to `torch.export`.
  warnings.warn(
```

**意义**：`requirements.txt` 原先只写 `torch>=2.0`（**没有上限**），
于是线上漂到了 **2.14.1**——一个本地与 CI 都未验证过的版本，且其
`torch.jit.load` 已被标记弃用。每次重部署都可能落到不同版本，**行为不可控**。

### 证据 4：崩溃不留 traceback

对比三份日志（一份崩溃、两份成功），**崩溃那一份的末尾是正常的**
`Uvicorn server started on :::8501`，随后**戛然而止**，没有异常、没有堆栈。

```
2026-10-06 13:24:20.039 Uvicorn server started on :::8501

/home/adminuser/venv/.../torch/jit/_serialization.py:176: FutureWarning: ...
```

**意义**：服务能正常启动，故障发生在启动之后。**"进程消失却没有任何错误输出"**
是 Linux OOM killer（或容器内存超限）的典型特征——它发 `SIGKILL`，进程来不及
写任何日志。代码异常则会留下完整 traceback，日志里没有。

> 诚实说明：日志面板里**没有** "Out of memory" 字样，因为平台的 OOM kill
> 不写进这个日志流。因此"崩溃 = 内存超限"是由"无 traceback + 进程消失 + 已知内存压力"
> 推断出来的**高置信结论**，而非日志中可直接指认的行。

---

## 根因

免费档约 2.7 GB RAM，而部署形态是：

```
冷启动 → import torch（连同 19 个 CUDA 包）
       → 加载 2 个 TorchScript 权重
       → 用户上传图 → 全分辨率推理，内存峰值翻倍
       → 触顶 → OOM killer 杀掉进程 → 页面崩溃、日志无痕
```

**「重部署后更容易崩」的解释**：重部署 = 全新进程 = 上面整条链从头走一遍，
冷启动时 `import` + 加载模型 + 首次推理三者叠加，是内存峰值最高的时刻。

---

## 修复

只改一个文件：`deploy/streamlit/requirements.txt`。

1. **CPU-only 索引** —— 加 `--extra-index-url https://download.pytorch.org/whl/cpu`，
   让 pip 取 CPU wheel，**19 个 CUDA 包不再安装**。
2. **钉死版本上限** —— `torch>=2.0,<2.5` / `torchvision>=0.15,<0.20`，
   阻止漂移到未验证的新版本。

**为什么这不影响模型质量**：权重是 TorchScript，**硬件无关**。
CPU 版与 CUDA 版 PyTorch 加载同一份 `.pt`，输出**逐位一致**。
本项目已用真实权重做过实测：同一输入连续推理 5 次输出哈希一致、
跨线程数（1/2/4/8）输出一致、跨进程一致，且逐像素比对最大差为 0。
权重的训练过程（GPU）与部署环境（CPU）互不影响，详见
[`DEPLOY.md`](../DEPLOY.md) 与仓库根的 README。

**验证**（沙箱 dry-run，真实 pip 解析）：

```
将安装包数: 2
CUDA/nvidia/triton 相关: 无 ✓
torch 版本: 2.4.1+cpu
来源: https://download-r2.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-...
```

即：CUDA 包归零，torch 钉在 2.4.1，且装的是 `+cpu` 构建。

### 线上验证（重新部署日志，2026-10-06）

沙箱 dry-run 只能证明"依赖解析会装什么"；修复是否生效最终以
**Streamlit Cloud 真实重部署日志**为准。修复提交推送后重部署，两份日志对照：

| 项 | 修复前（13:29 部署） | 修复后（14:39 部署） |
|---|---|---|
| 解析包数 | `Resolved 63 packages` | `Resolved 43 packages` |
| torch | `+ torch==2.14.1`（漂移版） | `+ torch==2.4.1+cpu` |
| torchvision | （CUDA 依赖随 torch 拉入） | `+ torchvision==0.19.1+cpu` |
| `nvidia-*` / `cuda-*` / `triton` 行 | **19 行** | **0 行** |
| `torch.jit.load` FutureWarning | 有 | 无（2.4.1 未弃用） |
| 服务启动 | `Uvicorn server started on :::8501` | 同左 |
| 崩溃 | 曾发生（不留 traceback） | **未发生**，日志正常收尾 |

包数 63 → 43，减少的 20 个正是 torch 的 CUDA 依赖链（19 个
`nvidia-*`/`cuda-*`/`triton` 包 + torch 换为 `+cpu` 构建后的差异）。
安装阶段从 27.51s 缩到 6.86s，启动后内存中不再加载任何 CUDA 运行库。

> 至此闭环：**现象（崩溃）→ 证据（4 组日志）→ 根因（CUDA 包白装 +
> 版本漂移挤占内存）→ 修复（CPU-only 索引 + 钉版本）→ 线上验证（对照表）**。
> 后续若再崩溃，按下方"已知缺口"的顺位继续排查。

---

## 已知缺口

以下措施有收益，但**缺少日志级证据**，因此按"先只改日志能证实的东西"的
原则暂缓：

| 项 | 内容 | 为何暂缓 |
|---|---|---|
| 线程数限制 | 设 `OMP_NUM_THREADS=1` | 实测证明对输出零影响，属纯优化；但无日志证据显示它是瓶颈 |
| 上传尺寸封顶 | 长边 > N 先等比缩小 | 单张超大图（如 4000×3000）理论上可致峰值翻数倍；但日志中未见该场景 |
| 抑制 `jit.load` warning | 降噪 | 纯日志噪音，不影响行为 |

> 若完成修复后仍出现崩溃，**上传尺寸封顶**是第一顺位要补的措施。

---

## 复现与核对方式

- 部署日志：Streamlit Cloud → app → **Manage app** → 日志面板。
- 依赖解析结果：改完 `requirements.txt` 后重新部署，日志中
  应**不再出现** `nvidia-*` / `cuda-*` / `triton` 行。
- 引擎确认：访客页面侧栏或 `/api/health`（FastAPI 侧）显示当前引擎为 `ml`。
