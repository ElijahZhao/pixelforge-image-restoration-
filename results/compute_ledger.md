# 算力台账：27 次内核、22.9 小时机上时间

> 这份台账记录 2026-11-01 → 11-03 在 Kaggle 免费 T4 上跑过的**每一次**内核。
>
> 时长不是估的：每一行都由 `kaggle kernels logs` 返回的日志逐条时间戳算出（取最后一条事件的相对秒数），
> 起点用 `kaggle kernels files` 的产物创建时刻。复算脚本是 `scripts/kernel_ledger.py`，
> 逐行数据在 `results/compute_ledger.csv`，原始证据在 `results/kernel_runs.json`。

## 一、四个数字

| 指标 | 数值 | 说明 |
|---|---|---|
| 内核 run 次数 | **27 次完成** + 1 次进行中 | 覆盖 10 个内核、27 个版本 |
| 墙上时间合计 | **22.87 h** | GPU 15.54 h + CPU 7.33 h |
| 其中失败或作废 | **6.82 h**（9 次，占 30%） | 见第四节 |
| 训练净时长 | **10.11 h** | 1 100 个 epoch，扣掉环境与数据下载后的纯训练时间 |

Kaggle 的周配额只扣 GPU 会话。**对账内核（`protocol-repo`）刻意关掉了 GPU**（`enable_gpu: false`），
因为它要证明"任何人 clone 下来都能复现"，不能依赖显卡——这 7.33 h 一分 GPU 配额都没花。

## 二、逐次明细

时间为北京时间（UTC+8）。

| # | 起 | 内核 · 版本 | 时长 | 硬件 | 在干什么 | 结局 |
|---|---|---|---|---|---|---|
| 1 | 11-01 20:11 | `smoke` v1 | 18 s | T4 | 环境自检：torch / CUDA / 磁盘 / 内存 | ✅ 通过 |
| 2 | 11-01 20:15 | `smoke` v2 | 24 s | T4 | 环境自检（补齐 GPU 型号与显存） | ✅ 通过 |
| 3 | 11-01 20:30 | `smoke2` v1 | 6 s | T4 | 挂载探测：找 `/kaggle/input` 固件 | ❌ 输入目录未挂载 |
| 4 | 11-01 20:30 | `smoke2` v2 | 30 s | T4 | 挂载探测：确认 working 目录可写 | ✅ 通过 |
| 5 | 11-01 21:48 | `io-probe` v1 | 18 s | T4 | IO 探测：写盘位置 / 软链 / 缓存复用 | ✅ 通过 |
| 6 | 11-01 21:55 | `eval-protocol` v1 | 6.6 min | T4 | 首次对外评估 SR×2 / SR×4，100 张 DIV2K val | ⚠️ 拿到 27.47 / 32.35，随后低光分支崩溃 |
| 7 | 11-01 22:08 | `eval-protocol` v2 | 7.2 min | T4 | 重跑评估，补齐 cross-check | ❌ LOL val 路径不存在 |
| 8 | 11-01 22:31 | `eval-protocol` v3 | 1.1 min | T4 | 重跑评估：改从 HF parquet 取 LOL | ❌ `train/datasets.py` 遮蔽了 HF `datasets` |
| 9 | 11-01 22:34 | `eval-protocol` v4 | 6.9 min | T4 | 最终评估：三任务 + 逐图明细 | ✅ 5 个产物，此后所有表格引用它 |
| 10 | 11-02 09:24 | `protocol-y` v1 | 5.5 min | T4 | Y 通道 + shave 口径复评，6 个口径一次跑完 | ✅ P5 锚点来源 |
| 11 | 11-02 10:27 | `perceptual-ablation` v1 | 3.0 min | T4 | 感知消融：软链已存在，启动即挂 | ❌ FileExistsError |
| 12 | 11-02 10:32 | `perceptual-ablation` v2 | **3.73 h** | T4 | 感知消融：VGG 感知开 / 关各训一次 + 评估 | ✅ 权重与日志齐全 |
| 13 | 11-02 17:40 | `protocol-matrix` v1 | 41.1 min | T4 | 协议矩阵首跑：两版权重 × 多协议 | ⚠️ 第二版权重缺失，只跑了一组 |
| 14 | 11-02 18:26 | `protocol-matrix` v2 | 54 s | T4 | 协议矩阵：改权重查找路径 | ❌ 仍找不到第二版权重 |
| 15 | 11-02 18:38 | `protocol-matrix` v3 | 48 s | T4 | 协议矩阵：改用 base64 内嵌权重 | ❌ 内嵌模块没生成 |
| 16 | 11-02 19:09 | `protocol-matrix` v4 | 48 s | T4 | 协议矩阵：核对内嵌权重体积 | ❌ 仍失败 |
| 17 | 11-02 19:11 | `protocol-matrix` v5 | 5.3 min | T4 | 协议矩阵：两版权重 × 8 协议全跑 | ✅ P1–P8 首份完整矩阵 |
| 18 | 11-02 19:23 | `protocol-repro-v6` v1 | 2.7 min | T4 | 独立复现：绕开主仓脚本，直接读权重算协议 | ✅ 与主仓脚本逐位一致 |
| 19 | 11-02 22:39 | `seed-matrix` v1 | 4.5 min | T4 | 种子矩阵冒烟：SRCNN ×1，少轮 | ✅ 测速样本 |
| 20 | 11-02 22:50 | `seed-matrix` v2 | 5.9 min | T4 | 种子矩阵冒烟：重跑核对耗时 | ✅ 测速样本 |
| 21 | 11-02 22:58 | `seed-matrix` v3 | 54 s | T4 | 低光分支冒烟：2 轮，lr=0 | ✅ 测速样本 |
| 22 | 11-02 23:10 | `seed-matrix` v4 | **4.53 h** | T4 | A 批：SRCNN ×2，种子 42–46，100 轮 | ✅ n=5，30.686 ± 0.069 |
| 23 | 11-02 23:56 | `protocol-repo` v1 | **3.23 h** | CPU | 对账 v1：P3 / P5 / P6，x4 三协议 | ❌ 锚点配错，判 FAIL |
| 24 | 11-03 09:44 | `seed-matrix` v5 | **5.70 h** | T4 | B 批：SRResNet ×4，种子 42–44，200 轮 | ✅ n=3，26.430 ± 0.027 |
| 25 | 11-03 09:47 | `protocol-repo` v2 | 12 s | CPU | 对账 v2：commit pin 改成短哈希 | ❌ 断言拦截：HEAD 全长哈希 ≠ 短哈希 |
| 26 | 11-03 09:48 | `protocol-repo` v3 | **3.37 h** | CPU | 对账 v3：P8 / P7 双协议 × 两个尺度 | ❌ x2 单协议超 7200 s 被 kill |
| 27 | 11-03 13:19 | `protocol-repo` v4 | 43.8 min | CPU | 对账 v4：x2 只跑 P7，timeout 收到 6600 s | ✅ 四锚点全过 |
| — | 11-03 15:36 | `seed-matrix` v6 | 进行中 | T4 | C 批：SRResNet ×4，种子 45–46，200 轮 | 🔄 约 19:25 完成 |

## 三、两条线并行

GPU 和 CPU 不是串行排队等着的。**A 批训练（23:10 起）与对账 v1（23:56 起）在同一天晚上并行**，
**B 批训练（09:44 起）与对账 v3（09:48 起）在同一个上午并行**——两次重叠合计 6.6 h。
所以 22.87 h 的机器时间实际压在 **16.3 h 的日历跨度**上；首末两次 run 之间隔了 41.9 h，
机器有 55% 的时间在算。

代价是有的：并行时段里 GPU 内核的启动要排队，A 批 v4 的日志显示环境探测用了 16 s、
3.5 GB 的 DIV2K 训练集下载用了 144 s（每次训练都要重下，Kaggle 不缓存）。

## 四、9 次失败，6.82 小时

失败不是意外，是这类工作的主要成本。逐条列在这里，包括被我们自己判定作废的：

| 内核 · 版本 | 烧掉 | 失败原因 | 怎么修的 |
|---|---|---|---|
| `eval-protocol` v2 | 7.2 min | LOL 验证集路径 `/kaggle/working/repo/data/lol/val/low` 不存在 | 改从 HuggingFace parquet 读 |
| `eval-protocol` v3 | 1.1 min | 仓库里的 `train/datasets.py` 遮蔽了 HF 的 `datasets` 包，`load_dataset` 导入失败 | 改名模块、换 import 方式 |
| `perceptual-ablation` v1 | 3.0 min | `os.symlink` 目标已存在，`FileExistsError` | 软链改成幂等（存在就跳过） |
| `protocol-matrix` v2 | 54 s | 第二版权重在工作目录找不到 | 加查找路径 |
| `protocol-matrix` v3 | 48 s | 改用 base64 内嵌，但模块没生成 | 内嵌脚本单独跑 |
| `protocol-matrix` v4 | 48 s | 内嵌权重体积对不上 | 核对 md5 |
| `protocol-repo` v1 | **3.23 h** | 锚点配错：拿 27.47 去比 P3（tv_aa），x2 差 0.112 dB 超容差 0.10 → 判 FAIL | 拆出只换算子的 P7 / P8，锚点重定；容差没放宽 |
| `protocol-repo` v2 | 12 s | commit pin 写了短哈希，`HEAD != pinned` 断言拦截 | 改完整 40 位哈希 |
| `protocol-repo` v3 | **3.37 h** | x2 单协议 CPU 评估超 7200 s 被 kill | 拆成单协议、timeout 收到 6600 s |
| `seed-matrix` v6 | 进行中 | — | — |

其中 `protocol-repo` v1 与 v3 两次就烧掉 **6.6 h**，占全部失败时间的 97%。
它们是 CPU 内核，不花 GPU 配额——这是当时把对账放到 CPU 上的直接收益。

## 五、训练净时长（内核墙钟之外）

内核墙钟含 pip 安装、clone、3.5 GB 数据下载。真正的训练时间由脚本自报：

| 批次 | 模型 × 种子 | 轮数 | 每颗种子 | 每 epoch 中位 | 合计净训练 |
|---|---|---|---|---|---|
| A | SRCNN ×2，种子 42–46 | 100 | 53.2–54.2 min | 31.9–32.4 s | 4.46 h |
| B | SRResNet ×4，种子 42–44 | 200 | 112.3–113.6 min | 33.7–34.0 s | 5.65 h |
| C | SRResNet ×4，种子 45–46 | 200 | 约 113 min | 约 34 s | 3.79 h（进行中） |

A 批跨夜（23:10 → 次日 03:41），B 批一个上午（09:44 → 15:25）。
**1 100 个 epoch 已经跑完**，C 批再补 400 轮，D 批（低光 ×5）再补一轮。

## 六、配额

- Kaggle 免费额度：T4，30 h / 周（周六 00:00 UTC 重置），￥0。
- 本周已用（GPU 部分）：**15.54 h**。
- 剩余可用：**13.8 h**（11-03 16:30 的页面读数）。
- C 批还需约 2.2 h、D 批约 2.5 h → 跑完剩 **约 9 h**，够。

## 七、怎么复算

```bash
python scripts/kernel_ledger.py --out-dir results
```

脚本会遍历 10 个内核的各个版本，拉执行日志与产物清单，重新生成
`results/compute_ledger.csv` 与 `results/kernel_runs.json`。
日志会缓存在 `.cache/kaggle-logs/`——Kaggle 对"正在运行的内核"的历史日志查询不稳定，
缓存让重跑不依赖 API 的临时状态。
