# 部署到 Streamlit Community Cloud（免费 · 推荐）

零费用、无 GPU 额度限制、直接从现有 GitHub 仓库部署。

## 为什么换成 Streamlit

HF Spaces 自 2026 年 7 月起，Gradio 跑计算需付费（PRO）；免费账号仅保留「2 个 ZeroGPU Gradio Space」，且**有每日 GPU 额度限制**。我们的 demo 是 **CPU 就能跑的轻量 TorchScript 推理**，被绑上"5 分钟/天"的 GPU 配额不划算。Streamlit Community Cloud：**公开 app 完全免费、无额度**，直接从 GitHub 部署。

## 文件

```
deploy/streamlit/
├── streamlit_app.py     # 自包含入口（classical 基线 + TorchScript 加载 + Streamlit UI）
├── requirements.txt     # 仅推理依赖
└── models/
    ├── sr_generator_scale4.pt
    └── lowlight.pt
```

## 部署步骤（5 分钟）

1. 浏览器打开 https://share.streamlit.io ，用 **GitHub 账号登录**。
2. 点 **Create app** → **Deploy a public app from GitHub**。
3. 填：
   - **Repository**: `ElijahZhao/pixelforge-image-restoration-`
   - **Branch**: `main`
   - **Main file path**: `deploy/streamlit/streamlit_app.py`
4. （可选）点 **Advanced settings** → Python version 选 3.10 或 3.11。
5. 点 **Deploy!** —— 首次构建约 2–4 分钟（装 torch 稍慢）。
6. 完成后得到公开链接：`https://<你的app名>.streamlit.app`

## 注意事项

- 无访问 12 小时后 app 休眠，访客打开会自动唤醒（冷启动 30–60 秒），属正常。
- 免费档内存约 2.7 GB，本 demo 推理很轻，够用。
- 若 `torch` 安装过慢导致构建超时，可在 `requirements.txt` 里把 `torch` 换成 CPU 版：
  `--extra-index-url https://download.pytorch.org/whl/cpu` + `torch==2.2.2+cpu`。

## 已修复的坑（本部署已含）

低光 U-Net 要求输入边长是 **32 的倍数**，否则解码器拼接崩溃。`streamlit_app.py` 的 `predict_lowlight` 已做自适应补齐（pad → 推理 → 裁回），任意尺寸安全。
