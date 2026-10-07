# AutoDL 新实例实操留档（截图证据）

> 目的：留存本人在新租用 AutoDL 实例上执行开训流程的**实操证据**，
> 证明代码版本正确、依赖安装安全（未污染 CUDA torch）、数据准备已启动。
>
> 实操日期：2026-10-07　｜　实例：西北B区 RTX 4090 (24GB) / 22 vCPU / 90GB 内存
> 镜像：PyTorch 2.3.0 / Python 3.12(ubuntu22.04) / CUDA 12.1
> 代码版本：本地 tar 包 `pixelforge-src-main.tar.gz`（基于提交 `72df86c`）

## 截图清单与对应步骤

| 编号 | 文件 | 对应手册步骤 | 证明内容 |
|---|---|---|---|
| 01 | `01_新实例开机_GPU验证.png` | 步骤 4 | `nvidia-smi` 显示 RTX 4090 / 24564MiB；`python -c` 输出 `torch 2.3.0+cu121 \| cuda: True`，**GPU 可用** |
| 02 | `02_解压tar_装依赖.png` | 步骤 5 | `cd /root/autodl-tmp` → `mkdir pixelforge` → `tar xzf pixelforge-src-main.tar.gz -C pixelforge`；开始 `pip install -r requirements.txt` |
| 03–08 | `03…08_pip安装依赖_*.png` | 步骤 5 | pip 逐个下载 numpy/starlette/gradio/huggingface-hub 等依赖；**全程未出现 torch 装卸**（证明确实未被换成 CPU 版） |
| 09 | `09_版本校验_grep输出4和3_步骤6建目录.png` | 步骤 5 → 6 | `grep -c "auto-resume" retrain_autodl.sh` → **4**；`grep -c "ALLOW_PARTIAL" scripts/download_data.sh` → **3**（证明代码为含修复的最新版）；随后执行 `bash scripts/download_data.sh` |
| 10 | `10_数据准备指引输出.png` | 步骤 6 | 脚本打印 DIV2K / LOL 下载指引与目标目录结构 |
| 11 | `11_步骤7_新终端wget_DIV2K_train_HR.png` | 步骤 7 | 新开终端 `cd /root/autodl-tmp && mkdir -p zips && cd zips`，`wget -c DIV2K_train_HR.zip` **正在下载（37%）** |
| 12 | `12_DIV2K下载_404弯路_换validation_release成功.png` | 步骤 7 | 训练集 3.29G 下载成功（35m34s）；验证集按官网原路径 `DIV2K_valid_HR.zip` 报 **404 Not Found**（ETH 已挪文件、官网链接未更新）；改用 `validation_release/` 子目录路径后 **200 OK**，428M 下载成功（5m10s）。**完整记录踩坑与排错过程** |
| 13 | `13_LOL下载完成331M_解压确认顶层目录.png` | 步骤 7 | `lol_dataset.zip` 经 hf-mirror.com 下载完成（331M / 39m12s，走 302→OSS 中转）；`unzip` 后 `ls` 确认四个包齐全：`DIV2K_train_HR.zip`、`DIV2K_valid_HR.zip`、`lol_dataset/`、`lol_dataset.zip`，**全部原始数据就位** |

## 关键佐证点

1. **环境正确**：`torch 2.3.0+cu121` 且 `cuda: True`（截图 01），说明镜像自带的 CUDA torch 完好。
2. **依赖安装安全**：整个 `pip install -r requirements.txt` 过程（截图 02–08）**没有任何 torch 相关行**，
   验证了「从 requirements 移除 torch 声明」这一修复生效——不再出现旧实例中
   `Installing collected packages: torch` 把 `+cu128` 换成 `+cpu` 的致命降级。
3. **代码版本正确**：`grep` 计数 4 / 3 与修复后代码一致（截图 09）。
4. **进度**：三个原始数据包全部下载完毕（截图 11–13）——DIV2K train 3.29G、DIV2K valid 428M、LOL 331M；
   其中验证集 404 的弯路（截图 12）已排错解决，下一步为步骤 8 解压归位 + 步骤 9 `--check` 校验。

## 踩坑记录（排错证据）

- **DIV2K 验证集 404**（截图 12 上半部分）：官网 `data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip` 返回 404。
  排查结论：ETH 将该文件挪至 `validation_release/` 子目录但官网页面链接未更新。改用
  `.../DIV2K/validation_release/DIV2K_valid_HR.zip` 后 200 OK。手册 `START_TRAINING_STEPS.md` 步骤 7 已同步更正。
- **LOL 下载源**：HuggingFace 官方域名在实例上不通，改走 `hf-mirror.com` 镜像（302 跳转至 OSS 中转）成功下载。

## 关联文档

- 操作手册：`docs/START_TRAINING_STEPS.md`
- 数据说明：`docs/RETRAIN_DATA_GUIDE.md`
