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
| 15 | `15_tmux开训_单测全PASS_下载VGG16感知损失权重.png` | 步骤 10–11 | tmux 会话 `train` 已建立（底部绿色状态栏）；`retrain_autodl.sh` 启动，`test_metrics` 5 项 PASS，进入 `test_correctness` 时触发一次性下载 VGG16 权重（528M，感知损失用，三阶段共用缓存） |
| 16 | `16_VGG下载完成_27测全过_冒烟完成_SRx4正式训练中.png` | 步骤 11 | VGG16 权重 528M 下载完成（2h48m）；单测 **27/27 全 PASS**；冒烟 2 epoch 完成（Best PSNR 18.33）；旧产物备份至 `backup_old/`；**SR×4 正式训练进行中**——Epoch 1→4：loss 0.1898→0.0649 单调降，val PSNR 18.12→20.72、SSIM 0.2302→0.4949 单调升，约 21.4s/epoch |
| 17 | `17_SRx4训练至Epoch17_PSNR22.59_连续刷新best.png` | 步骤 11 | SR×4 训练至 Epoch 17：loss 降至 0.0385，val PSNR 升至 22.59、SSIM 0.6718；**每个 epoch 都在刷新 best checkpoint**（曲线健康）；Epoch 10 时已写入断点 `ckpt/sr_generator_scale4_last.pt`（auto-resume 生效） |
| 18 | `18_SRx4完成但仅23.34_发现batch16问题_SRx2接续中.png` | 步骤 11 | SR×4 200 epoch 跑完，但 Best val PSNR 仅 **23.34** / SSIM 0.727，远低于历史同口径 27.39 / 0.819（epoch 101）；对比历史日志定位根因：脚本默认 `BATCH=16`，历史训练用 `batch_size 8`——每 epoch 优化步数减半（50 vs 100）导致欠训练；SR×2 已自动接续（epoch 1）。截图含无害的 cuDNN plan UserWarning |
| 19 | `19_SRx2训练至Epoch30_PSNR26.67_连续刷新best_每10ep存断点.png` | 步骤 11 | SR×2（新训练槽位，历史上从未训过、一直回退 bicubic）训练至 Epoch 30：PSNR 25.13→26.67、SSIM 0.8158→0.8723，loss 0.0365→0.0231，**每 epoch 刷新 best、每 10 epoch 存断点**；决定采用"按槽位择优"策略：SR×4 槽位沿用旧模型（backup_old/），SR×2/低光视新模型表现取舍 |
| 20 | `20_SRx2训练至Epoch45_PSNR27.07_持续刷新best.png` | 步骤 11 | SR×2 训练至 Epoch 45：PSNR 26.42→27.07、SSIM 0.8664→0.8866，loss 0.0256→0.0200，每个 epoch 继续刷新 best、每 10 epoch 存断点（epoch 40 已落盘）；同期建立 `docs/MODEL_SELECTION_TODO.md` 待办清单，固化"按槽位择优"决策：SR×4 弃新复旧，SR×2 跑完与 bicubic ×2 对比后再定 |
| 21 | `21_SRx2训练完毕27.72_低光接续开局正常.png` | 步骤 11 | **SR×2 训练完毕**：200 epoch 跑完，Best val PSNR **27.72 / SSIM 0.8994**（ep195），产出 `models/sr_generator_scale2_best.pth`（史上首个 ×2 权重）；低光阶段自动接续（srcnn 0.50M 参数，无断点从头训练），ep1–12：PSNR 14.35→15.95、SSIM 0.6264→0.7159，开局优于历史（历史 ep1 仅 11.87），约 8.2s/epoch；同期确立硬约束：**不再重训**，三次训练留档形成对照系列 |
| 22 | `22_三阶段全部完成_低光终值17.20_导出自检通过.png` | 步骤 11–13 | **三阶段全部跑完**：低光 200 epoch 完成，Best val PSNR **17.20**（低于历史 18.59 → 按槽位择优弃新复旧）；脚本自动导出 3 个 TorchScript（scale4 5.0M / scale2 3.3M / lowlight 2.3M），自检 [有]×3 全过；打印待发回文件清单。至此本批三阶段全部结束，进入"按槽位择优恢复 + eval_baseline 权威对比 + 打包下载"收尾 |
| 23 | `23_SRx2基线对比_新模型3235胜bicubic3104_采纳.png` | 步骤 14 | **SR×2 槽位定论**：eval_baseline 同口径（div2k/val 100 张，GPU）对比——bicubic 31.04/0.8935 vs 新模型 **32.35/0.9165（+1.31 dB，SSIM +0.0231）**，脚本判定"模型打赢基线"→ **采纳新训 ×2**，serve/models/sr_generator_scale2.pt 保留（训练日志 val 口径 27.72 与 eval 口径不可直接比，以 eval 为准）。三次训练中首个转正的模型，史上首个 ×2 部署权重生效 |
| 24 | `24_复验实为新模型_SRx4_2738胜基线_cp路径待修.png` | 步骤 15 | **cp 恢复失败暴露真实路径**：`backup_old/serve/models/` 不存在（备份脚本把备份放 `/root/autodl-tmp/backup_old/` 且两个 models 目录合并为 `backup_old/models/`）。因此 SR×4 复验实测的仍是**新训模型**：bicubic 26.69/0.7541 vs 新模型 **27.38/0.7771（+0.69 dB）**——新 ×4 在 eval 口径也赢基线（训练日志口径 23.34 的劣势未重现，两口径差异大）；旧模型 eval 口径数字待恢复后补测 |
| 25 | `25_低光新模型eval_1832_增益1055dB.png` | 步骤 15 | **低光新模型 eval 口径成绩**：不处理 7.77/0.1924 vs 新模型 **18.32/0.7457（+10.55 dB）**（LOL val 15 张）；对比历史模型 eval 口径数字待恢复后补测。cudnn 告警为已知无害警告 |
| 26 | `26_backup结构确认_旧x4与低光恢复至部署位_scale2保持新模型.png` | 步骤 16 | **backup_old 真实结构确认**：find 证实旧部署权重在 `/root/autodl-tmp/backup_old/models/`（与推断一致，两 models 目录被 GNU cp 合并）；cp 恢复成功，`ls -l serve/models/` 显示 scale4/lowlight 时间戳 21:36（旧模型）、scale2 保持 21:08（新训模型）——部署位按决策就位 |
| 27 | `27_同口径复核_SRx4旧2747胜新2738_低光新旧口径矛盾.png` | 步骤 16 | **同口径复核定稿**：SR×4 旧模型 **27.47/0.7798** > 新模型 27.38（与历史部署记录 27.47 完全吻合，弃新复旧复核通过）；低光旧模型 eval **18.12/0.7425** vs 新模型 18.32——eval 口径新微高 0.2 dB，但 val 口径旧高 1.4 dB，**口径矛盾**，按"跨口径稳健性"裁决维持旧模型（15 张样本 0.2 dB + SSIM 仅差 0.003 不构成翻盘）。**最终阵容三槽位全部实测打赢各自基线** |
| 14 | `14_数据归位完成_结构校验全过_800_100_485_15.png` | 步骤 8–9 | LOL 四路 `cp` 归位 + DIV2K 解压归位完成；`download_data.sh --check` 输出**六个 `[有]`：800 / 100 / 485 / 485 / 15 / 15**，全部达标，**数据准备阶段完成**，脚本提示可跑 `eval_baseline.py` 评估基线 |

## 关键佐证点

1. **环境正确**：`torch 2.3.0+cu121` 且 `cuda: True`（截图 01），说明镜像自带的 CUDA torch 完好。
2. **依赖安装安全**：整个 `pip install -r requirements.txt` 过程（截图 02–08）**没有任何 torch 相关行**，
   验证了「从 requirements 移除 torch 声明」这一修复生效——不再出现旧实例中
   `Installing collected packages: torch` 把 `+cu128` 换成 `+cpu` 的致命降级。
3. **代码版本正确**：`grep` 计数 4 / 3 与修复后代码一致（截图 09）。
4. **进度**：已进入正式训练（截图 16）——VGG 下载完成、27/27 单测全过、冒烟通过，SR×4（200 epoch）训练中，损失与指标走势正常。

## 踩坑记录（排错证据）

- **DIV2K 验证集 404**（截图 12 上半部分）：官网 `data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip` 返回 404。
  排查结论：ETH 将该文件挪至 `validation_release/` 子目录但官网页面链接未更新。改用
  `.../DIV2K/validation_release/DIV2K_valid_HR.zip` 后 200 OK。手册 `START_TRAINING_STEPS.md` 步骤 7 已同步更正。
- **LOL 下载源**：HuggingFace 官方域名在实例上不通，改走 `hf-mirror.com` 镜像（302 跳转至 OSS 中转）成功下载。

## 关联文档

- 操作手册：`docs/START_TRAINING_STEPS.md`
- 数据说明：`docs/RETRAIN_DATA_GUIDE.md`
