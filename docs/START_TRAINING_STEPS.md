# AutoDL 开训操作手册（逐步执行）

> 配套脚本：`retrain_autodl.sh`　配套数据说明：`RETRAIN_DATA_GUIDE.md`
>
> **每个步骤都标注了「在哪个终端执行」。** 记号说明：
> - 🖥️ **本地电脑终端**（你自己的笔记本/台式机）
> - ☁️ **AutoDL 实例终端**（网页 JupyterLab 的 Terminal，或 SSH 连上去的那个）
> - 🔀 **需要新开一个终端 / 切换到别的终端**
> - ➡️ **接着上一步，不用切换**

---

## 总览：终端切换地图

```
🖥️ 本地机                    ☁️ AutoDL 实例
──────────────────────────────────────────────────────────
T0: 打包源码                  T1: 主线操作（克隆、下数据、开训）
   │                            │
   └── 上传 zip ───────────────►┤
                                ├── T2: tmux 会话内跑训练（T1 里 tmux new 进去）
                                │      ↑ 断线也不影响，随时 attach 回来
                                │
                                └── 训练完 → 打包产物 → 下载回 T0
```

**结论先讲：日常只需要两个终端**——本地机一个、AutoDL 一个。训练跑在 AutoDL 那个终端的 **tmux 会话**里，tmux 是「同一个终端里的一个可分离窗口」，不算新终端。

---

# 第一部分：把代码送到 AutoDL

## 步骤 1 · 🖥️ 本地电脑终端：拿到源码

远端 main **已经更新到 `5d0b0e5`**（用你提供的 PAT 推送成功）。所以有两种拿代码的方式，**任选其一**：

**1-A（最省事）· 直接 clone 远端**
AutoDL 实例终端里 `git clone https://github.com/ElijahZhao/pixelforge-image-restoration-.git` 即可，含全部修复。

**1-B · 用我给你的打包文件**（如果 clone 不便，或想离线拿）
我准备了两个文件，在 `/workspace/dist/`：

| 文件 | 用途 |
|---|---|
| `pixelforge-src-main.tar.gz` | **纯源码**，解压即用（推荐） |
| `pixelforge-main.bundle` | 带完整 git 历史的 bundle（想保留提交记录时用） |

下载到本地电脑：
- 点我给你的文件卡片，把 `pixelforge-src-main.tar.gz` 存到 `~/Downloads/`
- 或用 `pixelforge-main.bundle`（含提交 `5d0b0e5`）

➡️ 这一步在**本地电脑**完成，终端不用切换。

---

## 步骤 2 · 🖥️ 本地电脑终端：上传到 AutoDL

用 AutoDL 官方的上传通道最省事，**不需要命令行**：

1. 打开 AutoDL 控制台 → 你的实例 → **JupyterLab**
2. 左侧文件树进入 `/root/autodl-tmp/`
3. 直接把 `pixelforge-src-main.tar.gz` **拖进文件树**上传

> **为什么放 `/root/autodl-tmp/`？** AutoDL 系统盘只有 30–50 GB，训练数据好几个 GB，
> 会撑爆。`/root/autodl-tmp/` 是数据盘，容量大。代码也一并放这里，脚本里写的就是这个路径。

如果你习惯用命令行（scp），也可以：

```bash
# 🖥️ 本地电脑终端，把 <端口> <主机> 换成 AutoDL 实例 SSH 面板给的
scp ~/Downloads/pixelforge-src-main.tar.gz root@<主机>:/root/autodl-tmp/
```

➡️ 仍在**本地电脑**，不用切换终端。

---

## 步骤 3 · 🔀 切到 ☁️ AutoDL 实例终端：解压代码 + 装依赖

**现在切换终端**：从本地电脑终端 → 切到 AutoDL 的 JupyterLab Terminal（或 SSH 登录的终端）。

```bash
# ☁️ AutoDL 实例终端
cd /root/autodl-tmp
mv pixelforge pixelforge_old 2>/dev/null   # 若已有旧目录，先改名备份
mkdir -p pixelforge
tar xzf pixelforge-src-main.tar.gz -C pixelforge
cd /root/autodl-tmp/pixelforge
```

> ⚠️ **这个包没有顶层目录**（`git archive` 默认行为）。必须先 `mkdir pixelforge`
> 再 `tar xzf ... -C pixelforge` 解进去，**不要直接在 `/root/autodl-tmp` 下解压**，
> 否则文件会散一地、和旧文件混在一起。
> 旧目录改名成 `pixelforge_old` 保留——如果之前下过数据（`pixelforge_old/data/`），
> 稍后可以搬进新目录复用，不用重新下载几个 GB。

装依赖：

```bash
# ☁️ AutoDL 实例终端
pip install -r requirements.txt
```

> AutoDL 的 PyTorch 镜像通常已带 CUDA 版 torch。但本仓库的 `requirements.txt`
> 默认从 PyPI 装 **CPU 版** torch——若镜像自带的 CUDA torch 被换成了 CPU 版，
> 训练会退回 CPU、慢到不可用。**装完必须验证 GPU：**

```bash
# ☁️ AutoDL 实例终端
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

若打印 `cuda: False`，说明 torch 是 CPU 版，立刻换回 CUDA 版（**先卸载再装，否则 pip 认为已满足版本号不会换**）：

```bash
pip uninstall -y torch torchvision
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ 换好 CUDA torch 后，**不要再跑 `pip install -r requirements.txt`**——
> 它会再把 torch 换回 CPU 版。其余依赖（fastapi/gradio 等）已经装好了。

**验证代码是对的**（能对上提交说明就说明是含修复的版本）：

```bash
# ☁️ AutoDL 实例终端
grep -c "auto-resume" retrain_autodl.sh    # 应 >= 2（实测 4）
grep -c "ALLOW_PARTIAL" scripts/download_data.sh   # 应 >= 1（实测 3）
```

两条都该输出非零数字。**若输出 0 或报错，说明你拿到的是旧代码，停在这里告诉我。**

---

# 第二部分：准备数据

## 步骤 4 · ☁️ AutoDL 实例终端：建目录 + 看下载指引

```bash
# ☁️ AutoDL 实例终端，接着上一步
cd /root/autodl-tmp/pixelforge
bash scripts/download_data.sh
```

➡️ 不用切换终端，还是 AutoDL 那个。

---

## 步骤 5 · 🔀 开一个独立终端下数据（可选，但推荐）

下载几个 GB 会占住终端很久。**建议新开一个 AutoDL 终端专门下数据**，让主终端空着。

**切换方式**：JupyterLab 左侧可以再开一个 Terminal；或本地再开一个 SSH 连接。

```bash
# ☁️ AutoDL 实例终端（新的那个）
cd /root/autodl-tmp
mkdir -p zips && cd zips

# DIV2K 超分数据（瑞士服务器，国内可能慢）
wget -c https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget -c https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
```

> `-c` 是断点续传，断了重跑同一条命令接着下。
> **不要下 `DIV2K_train_LR_bicubic.zip`** —— 我们训练时用代码实时造低清图，
> 放官方 LR 版等于降采样两次，训练目标就错了。

LOL 数据集（Google Drive 常年失效，用备用源，拿到 zip 后同样丢进 `/root/autodl-tmp/zips/`）：

```bash
# ☁️ 同上，LOL 视你拿到的链接
# 目标：得到 LOLdataset.zip（内层含 our485/ 与 eval15/）
```

下载太慢的话，用 AutoDL 的**学术资源加速**，或本地下好再用步骤 2 的方式拖上来。

---

## 步骤 6 · ☁️ AutoDL 实例终端：解压到正确结构

回到**主终端**（或任意一个 AutoDL 终端都行）：

```bash
# ☁️ AutoDL 实例终端
cd /root/autodl-tmp/zips

# --- DIV2K ---
unzip -q DIV2K_train_HR.zip
unzip -q DIV2K_valid_HR.zip
cp DIV2K_train_HR/*.png /root/autodl-tmp/pixelforge/data/div2k/train/
cp DIV2K_valid_HR/*.png /root/autodl-tmp/pixelforge/data/div2k/val/

# --- LOL-v1 ---
unzip -q LOLdataset.zip
LOL=/root/autodl-tmp/pixelforge/data/lol
cp LOLdataset/our485/low/*.png  $LOL/train/low/
cp LOLdataset/our485/high/*.png $LOL/train/high/
cp LOLdataset/eval15/low/*.png  $LOL/val/low/
cp LOLdataset/eval15/high/*.png $LOL/val/high/
```

> **LOL 的目录名必须是 `low/` 和 `high/`。** 有些版本叫 `input/`、`target/`，要改名。
> 六个目标目录的**文件名要一一对应**（同名的 low/high 才是同一对被增强的图）。

---

## 步骤 7 · ☁️ AutoDL 实例终端：校验（必须通过）

```bash
# ☁️ AutoDL 实例终端
cd /root/autodl-tmp/pixelforge
bash scripts/download_data.sh --check
```

**期望看到六个 `[有]` 且数字精确匹配**：

```
  [有] DIV2K 训练图: 800 个文件
  [有] DIV2K 验证图: 100 个文件
  [有] LOL 训练 low: 485 个文件
  [有] LOL 训练 high: 485 个文件
  [有] LOL 验证 low: 15 个文件
  [有] LOL 验证 high: 15 个文件
数据就绪。
```

⚠️ **这是我刚修过的地方**：旧版只看目录非空，放 1 张图也会说「数据就绪」。
现在会核对数量——数字对不上会打印 `[少]` 并**拒绝开训**，这是故意的。
少一半图不会报错，只会让结果悄悄变差。

若你**有意只用子集**试流程，显式放行（结果不具备与论文数字的可比性）：

```bash
bash scripts/download_data.sh --check --allow-partial
```

➡️ 仍然在 AutoDL 终端，不用切换。

---

# 第三部分：开训

## 步骤 8 · ☁️ AutoDL 实例终端：建 tmux 会话

**为什么要 tmux**：SSH 一断、网页一关，前台进程就被杀。tmux 让训练跑在与终端
「可分离」的会话里，断线也不掉。

```bash
# ☁️ AutoDL 实例终端
tmux new -s train
```

**执行后你的终端提示符会变成 tmux 的样子**（底部出现绿色状态栏）。
这不是新终端，是**同一个终端里切换进了 tmux 会话**。

> 如果提示 `tmux: command not found`：`apt-get update && apt-get install -y tmux`。

➡️ **不用切换终端**，只是当前终端进了 tmux。

---

## 步骤 9 · ☁️ AutoDL 实例终端（tmux 内）：启动训练

```bash
# ☁️ tmux 会话内
cd /root/autodl-tmp/pixelforge
bash retrain_autodl.sh
```

**跑之前先看它要做什么**（脚本会依次执行）：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| Step 0 | 校验数据 + 跑 27 项单测 | 半分钟 |
| Step 1 | 备份旧产物到 `/root/autodl-tmp/backup_old/` | 几秒 |
| Step 2 | 冒烟测试（2 轮，确认 GPU 跑得通） | 1–2 分钟 |
| Step 3 | **SR ×4 重训，200 轮** | 最长 |
| Step 4 | **SR ×2 重训，200 轮** | 中等 |
| Step 5 | **低光重训，200 轮** | 最短 |
| Step 6 | 导出 3 个 TorchScript `.pt` | 几秒 |
| Step 7 | 自检三个 `.pt` 能前向 | 几秒 |

**Step 2 是安全阀**：若报 `CUDA out of memory`，说明这张卡 batch 16 装不下。
按 `Ctrl-C` 停掉，改小 batch 重跑（见步骤 11）。

---

## 步骤 10 · ☁️ tmux 内：挂起训练（可以关电脑）

训练在跑时，把它挂到后台、放心的走开：

**按 `Ctrl-B`，松开，再按 `D`**

屏幕会提示 `[detached]`，你回到了普通终端。**此时关掉网页、断开 SSH 都不影响训练。**

想回来看进度：

```bash
# ☁️ AutoDL 实例终端（任何时候）
tmux attach -t train
```

再想挂起，还是 `Ctrl-B` 然后 `D`。

> 认不出来自己还在不在 tmux 里？看终端底部有没有那条状态栏。
> 有 = 在 tmux 内；没有 = 已挂起。

---

## 步骤 11 · ☁️ AutoDL 实例终端：显存不够时的退路

如果 Step 2 冒烟测试报了 `CUDA out of memory`：

```bash
# ☁️ tmux 内，先 Ctrl-C 停掉，然后
BATCH=8 bash retrain_autodl.sh       # 退回旧配置
# 或折中
BATCH=12 bash retrain_autodl.sh
```

**只降 batch 就够。** 不要靠减轮数或调 `--save_every` 省显存——那两个跟显存无关。

---

## 步骤 12 · ☁️ AutoDL 实例终端：断点续训（实例被回收后）

**难点：AutoDL 实例可能被回收，训练会中断。**

好消息：**重跑同一条命令即可**，脚本带 `--auto-resume`——有断点就接着跑，
没有就从头开始。首次运行和恢复运行用的是**同一条命令**。

```bash
# ☁️ AutoDL 实例终端
tmux new -s train
cd /root/autodl-tmp/pixelforge
bash retrain_autodl.sh          # 原样重跑，自动续上
```

**已经跑完的阶段可以不重跑**，用开关跳过：

```bash
# 例：SR ×4 已完成，只跑 ×2 和低光
SKIP_SMOKE=1 SKIP_SR4=1 bash retrain_autodl.sh
```

可用开关：`SKIP_SMOKE` / `SKIP_SR4` / `SKIP_SR2` / `SKIP_LOWLIGHT`，设 `1` 即跳过。

> 断点每 10 轮写一次，在 `models/ckpt/`。所以最坏情况是「回到 10 轮前」。
> 断点文件是**原子写入**的，被 kill 也不会留半个损坏文件导致续训崩掉。

---

# 第四部分：训练完成后

## 步骤 13 · ☁️ AutoDL 实例终端：确认产物齐全

```bash
# ☁️ AutoDL 实例终端
cd /root/autodl-tmp/pixelforge

# 三个权重
ls -lh models/*_best.pth

# 三个日志
ls -lh results/train_log_*.csv

# 三个导出的 .pt（Step 7 自检应该已打印它们能前向）
ls -lh serve/models/*.pt
```

脚本跑完最后会打印一段「DONE」和该发回的文件清单。

---

## 步骤 14 · 🔀 ☁️ AutoDL：打包产物

```bash
# ☁️ AutoDL 实例终端
cd /root/autodl-tmp/pixelforge
tar czf /root/autodl-tmp/retrain_out.tar.gz \
  results/train_log_sr_generator_scale4.csv \
  results/train_log_sr_generator_scale2.csv \
  results/train_log_lowlight_srcnn_scale2.csv \
  train_sr_scale4.log train_sr_scale2.log train_lowlight.log \
  serve/models/sr_generator_scale4.pt \
  serve/models/sr_generator_scale2.pt \
  serve/models/lowlight.pt
ls -lh /root/autodl-tmp/retrain_out.tar.gz
```

---

## 步骤 15 · 🔀 切回 🖥️ 本地电脑终端：下载产物

**切换终端**：从 AutoDL 回到本地电脑。

用 **JupyterLab 文件树**下载最省事：进 `/root/autodl-tmp/`，找到
`retrain_out.tar.gz`，右键 → Download。

或命令行：

```bash
# 🖥️ 本地电脑终端
scp root@<主机>:/root/autodl-tmp/retrain_out.tar.gz ~/Downloads/
```

**把 `retrain_out.tar.gz` 发回给我**，我会做两件事：

1. **确认验证曲线是否真的可比** —— 中心裁剪后应表现为**连续波动**，
   而不是旧日志里那种第 101 轮突然冒出来的孤峰（27.39 dB，邻居全在 25.3–26.6）。
2. **新口径 vs 旧口径对照** —— 确认指标改善到底来自模型变好，
   还是来自修复后的尺子量得更准。

---

# 附：常见问题速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `tmux: command not found` | 镜像没装 tmux | `apt-get install -y tmux` |
| Step 0 报「数据集未就绪」 | 数量不足 | 按 `[少]` 提示补齐；确实用子集则加 `--allow-partial` |
| Step 2 报 `CUDA out of memory` | 卡太小 | `BATCH=8 bash retrain_autodl.sh` |
| `--resume: no checkpoint` | 只有在**手动**传 `--resume` 时出现 | 用脚本就好，脚本传的是 `--auto-resume` |
| 续训提示「断点损坏，从头开始」 | 上次被强杀，但原子写入保护了文件 | **正常**，会自动从头跑，不用管 |
| 怀疑跑的还是旧管线 | 拿错代码版本 | `grep -n "auto-resume" retrain_autodl.sh` 应有输出 |
| 磁盘满 | 数据放系统盘了 | 确认数据都在 `/root/autodl-tmp/` 下 |

---

# 附：待你确认的一件事

远端 GitHub 仓库 **已经更新到 `5d0b0e5`**——用你提供的 PAT 推送成功（非强制推送，
远端原本领先的 8 个提交已通过 merge 保留，未丢失）。所以上面的步骤里 clone 远端和用
打包文件两种方式都可，内容一致。

> ⚠️ 提醒：你之前有个待办是**吊销 PAT（个人访问令牌）**。那个令牌已在对话里
> 暴露过、也实际用过一次，请尽快去 GitHub → Settings → Developer settings →
> Personal access tokens → 找到它 → **Revoke**。吊销后不影响已推上去的代码。
