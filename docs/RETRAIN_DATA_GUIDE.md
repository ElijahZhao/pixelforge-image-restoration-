# AutoDL 重训 · 数据准备与执行清单

> 配合 `retrain_autodl.sh` 使用。数据不随仓库分发，需要手动准备一次。

## 零、先把代码放上去

`retrain_autodl.sh` 写死了工作目录 `/root/autodl-tmp/<项目目录>`，**先把仓库放到那里**：

```bash
cd /root/autodl-tmp
git clone <仓库地址> pixelforge     # 目录名用 pixelforge，与脚本一致
cd pixelforge
```

若你的目录名不同，两个选择：改目录名，或改脚本第 19 行的 `cd` 目标。

**确保拿到的是含修复的分支**（否则跑的还是旧管线）：

```bash
git log --oneline -3        # 应能看到 fix: 验证集改为确定性裁剪 等提交
```

## 一、需要哪些数据

| 数据集 | 用途 | 文件 | 数量 | 解压后大小 |
|---|---|---|---|---|
| **DIV2K_train_HR** | SR 训练 | `DIV2K_train_HR.zip` | 800 张 | ~3.3 GB |
| **DIV2K_valid_HR** | SR 验证 | `DIV2K_valid_HR.zip` | 100 张 | ~430 MB |
| **LOL-v1** (`our485`) | 低光训练 | `LOLdataset.zip` | 485 对 | ~90 MB |
| **LOL-v1** (`eval15`) | 低光验证 | 同上 | 15 对 | ~3 MB |

**总计约 4 GB。**

### 三个必须避开的坑

1. **不要下 `DIV2K_train_LR_bicubic.zip`。** 本项目在 `train/datasets.py` 里**实时 bicubic 降采样**造 LR；如果你放的是官方 LR 版，等于降采样了两次，训练目标就错了。只要 **HR** 版。
2. **验证用官方 `valid`，不要从 train 里切。** 否则 PSNR 与文献数字不可比。
3. **LOL-v1 要 `our485`（训练）和 `eval15`（验证）**，不是别的划分。

## 二、下载地址

### DIV2K
- 官方页：https://data.vision.ee.ethz.ch/cvl/DIV2K/
- 直链：
  ```
  https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
  ```
- 服务器在瑞士，国内直连较慢。AutoDL 实例上可先试直连，慢的话用学术加速或本地下载后上传。

### LOL-v1
- 原出处：https://github.com/weichen582/RetinexNet → `LOLdataset.zip`
- **Google Drive 链接常年失效**，备选：
  - Kaggle 搜 `lol-dataset`
  - OpenXLab / 各镜像站
  - 注意认准目录名 `our485/`（含 `low/` `high/`）与 `eval15/`

## 三、放到哪里

**AutoDL 的系统盘通常只有 30–50 GB，大数据必须放 `/root/autodl-tmp/`（大盘）。**

两种上传方式：

**方式 A · sftp / 网盘挂载**（推荐，最省事）
```bash
# 在 AutoDL 实例上，用阿里云盘/AutoDL 自带的文件传输，把 zip 传到
# /root/autodl-tmp/zips/ 下，然后解压。
```

**方式 B · 实例内直接下载**
```bash
cd /root/autodl-tmp
mkdir -p zips && cd zips
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
# LOL 视你拿到的链接而定
```

## 四、解压到正确结构

```bash
cd /root/autodl-tmp/pixelforge
bash scripts/download_data.sh          # 建目录 + 打印指引

# --- DIV2K ---
cd /root/autodl-tmp/zips
unzip -q DIV2K_train_HR.zip
unzip -q DIV2K_valid_HR.zip
# 解压出来是 DIV2K_train_HR/ 与 DIV2K_valid_HR/ 两个文件夹，里面是 0001.png...
cp DIV2K_train_HR/*.png /root/autodl-tmp/pixelforge/data/div2k/train/
cp DIV2K_valid_HR/*.png /root/autodl-tmp/pixelforge/data/div2k/val/

# --- LOL-v1 ---
# 假设解压出 LOLdataset/，内含 our485/ 与 eval15/
cd /root/autodl-tmp/zips && unzip -q LOLdataset.zip
LOL=/root/autodl-tmp/pixelforge/data/lol
cp LOLdataset/our485/low/*.png  $LOL/train/low/
cp LOLdataset/our485/high/*.png $LOL/train/high/
cp LOLdataset/eval15/low/*.png  $LOL/val/low/
cp LOLdataset/eval15/high/*.png $LOL/val/high/
```

> **LOL 的目录名必须严格是 `low/` 和 `high/`** —— `train/datasets.py` 按字面路径查找。
> 若原包是 `input/` 和 `target/`，需要改名。

## 五、校验（必须通过才能开训）

```bash
cd /root/autodl-tmp/pixelforge
bash scripts/download_data.sh --check
```

这个命令会**同时核对目录结构和文件数量**，期望输出：

```
  [有] DIV2K 训练图: 800 个文件
  [有] DIV2K 验证图: 100 个文件
  [有] LOL 训练 low: 485 个文件
  [有] LOL 训练 high: 485 个文件
  [有] LOL 验证 low: 15 个文件
  [有] LOL 验证 high: 15 个文件
数据就绪。
```

**数目对不上就别开训**——少一半的图不会报错，只会让结果悄悄变差。这就是为什么
校验不满足于"目录非空"，而是逐个核对到上面的数字：

- 文件数**不足** → 打印 `[少]` 并以退出码 1 结束，`retrain_autodl.sh` 会在 Step 0 停下。
- 文件数**超出** → 打印 `[多]`，但放行（多余文件不会被读到）。
- 你**确实有意只用子集**（例如先用 100 张跑通流程）→ 需要显式放行：

  ```bash
  bash scripts/download_data.sh --check --allow-partial
  ```

  这种情况下训练能跑，但结果不具备与论文数字的可比性，别拿它下结论。

## 六、开训

```bash
cd /root/autodl-tmp/pixelforge
tmux new -s train
bash retrain_autodl.sh
# Ctrl-B 然后按 D 挂起；断线后用 tmux attach -t train 回来看
```

### 断点续训

实例被回收后，**重跑同一脚本即可**——训练带 `--auto-resume`，有断点就续、没有就开始，
所以首次运行和恢复运行用的是同一条命令。

每 10 轮写一次断点，位于 `models/ckpt/`。想跳过已完成的阶段：

```bash
SKIP_SMOKE=1 SKIP_SR4=1 bash retrain_autodl.sh   # 只跑 x2 和低光
```

### 轮数与 batch

```bash
EPOCHS=200 BATCH=16 bash retrain_autodl.sh   # 默认值
```

- **200 轮**：旧配置在第 50–100 轮就已基本收敛（后 150 轮 SR 只涨 0.36 dB），加轮收益很小。
- **batch 16**：旧配置是 8。**真正的瓶颈不是轮数，而是每轮信息量**——旧配置每轮每张图只裁 1 个 96×96 patch，等于每轮只看整图的 0.3%。

**显存不足怎么办**：冒烟测试（Step 2）会用 batch 16 先跑 2 轮，若报 `CUDA out of memory`，
说明该卡装不下，直接调小：

```bash
BATCH=8 bash retrain_autodl.sh     # 退回旧配置，仍比原来多 x2 与低光两个训练
BATCH=12 bash retrain_autodl.sh    # 折中
```

batch 变小不影响能否跑通，只影响速度与梯度稳定性。**不要靠加 `--save_every` 或减轮数来省显存**，那两个都跟显存无关。

### 预期耗时（参考）

三个训练合计约 200×3 轮的训练量。参考旧日志的单轮耗时（batch 8、不同 GPU 会有差异）：

| 阶段 | 相对耗时 |
|---|---|
| 低光（485 对 / 128px patch） | 最短 |
| SR x2 | 中等 |
| SR x4 | 最长（含感知损失的前向/反向） |

用 `nvidia-smi` 看实际占用；卡住不动先查是不是在验证（验证阶段 GPU 利用率会掉）。
- 显存不足就把 `BATCH` 调回 8 或 12；batch 变小不影响能否跑通，只影响速度与稳定性。

## 七、跑完发回什么

```
results/train_log_sr_generator_scale4.csv
results/train_log_sr_generator_scale2.csv
results/train_log_lowlight_srcnn_scale2.csv
train_sr_scale4.log  train_sr_scale2.log  train_lowlight.log
serve/models/ 下三个 .pt
```

> **低光的文件名里带 `srcnn`，别被它误导。** 低光任务无论传什么 `--model` 都构建
> `LowLightUNet`（见 `train/models.py`），那个 token 只进文件名、不进网络结构。
> 脚本刻意不传 `--model`，好让新产物与已提交的权重 `lowlight_srcnn_scale2_best.pth`
> 和既有日志同名——否则同一份模型会出现 `_srcnn_` 与 `_generator_` 两种叫法，
> 后面对不上账。

拿到日志后我会做两件事：
1. **确认验证曲线是否真的可比**（中心裁剪后应表现为连续波动，而不是旧那种孤峰）
2. **新口径 vs 旧口径对照**，确认指标改善来自模型还是来自修复后的尺子
