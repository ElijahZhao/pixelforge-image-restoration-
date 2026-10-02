# 部署到 Hugging Face Spaces（方案 B · 最省事）

一个公开链接、全功能 demo（超分 + 低光），无需改 Next.js 静态导出、无需自建服务器。

## 为什么选这条路

`serve/` 后端需要 Python + torch + 权重，**GitHub Pages 托管不了**（纯静态）、**GitHub Actions 也托管不了**（只跑临时 job）。HF Spaces 免费提供带 CPU 的常驻容器，正好跑 Gradio。

## 目录内容

```
deploy/hf_space/
├── app.py              # 自包含入口（classical 基线 + TorchScript 加载 + Gradio UI）
├── requirements.txt    # 仅推理依赖
├── README.md           # Space 元信息（YAML 头：sdk: gradio）
└── models/
    ├── sr_generator_scale4.pt   # 4.8M
    └── lowlight.pt              # 2.3M
```

## 部署步骤（10 分钟）

1. 登录 https://huggingface.co → **New Space**
   - Name: `pixelforge-image-restoration`
   - **SDK: Gradio**，SDK version 默认即可
   - Hardware: **CPU basic (free)** 足够（模型很小）
   - Visibility: **Public**
2. 在该 Space 的 **Files** 页上传 4 个条目：`app.py`、`requirements.txt`、`README.md`、以及 `models/` 整个文件夹。
   - 或用 git：
     ```bash
     git clone https://huggingface.co/spaces/<你的用户名>/pixelforge-image-restoration
     cd pixelforge-image-restoration
     cp -r <本目录>/* .
     git add -A && git commit -m "deploy pixelforge demo" && git push
     ```
3. Space 自动构建（首次约 2–4 分钟，装 torch 稍慢），完成后即得公开链接：
   `https://huggingface.co/spaces/<你的用户名>/pixelforge-image-restoration`

## 已在沙箱验证的事实

- 引擎状态：`SR ×4: ML · Low-light: ML`（两个自训权重都被加载，非基线兜底）。
- 鲁棒性：已修复 U-Net 的「32 整除」约束——见下方说明，100×100、1280×720、63×41 等任意尺寸均不再崩溃。

## ⚠️ 关键修复：U-Net 输入尺寸约束（本部署已含）

低光 U-Net 会多次 2× 下采样，**输入长宽必须是 32 的倍数**，否则解码器拼接时张量尺寸对不上（`Expected size 26 but got size 25`）而崩溃。

`app.py` 中的 `predict_lowlight` 已做**自适应补齐**：输入 padding 到 32 倍数 → 推理 → 裁回原尺寸，任意尺寸都安全。

> **同一问题存在于主仓库 `serve/model_loader.py`**（本地 `uvicorn` 服务同样会在非 32 倍数尺寸上 500）。建议把同样的补齐逻辑同步过去。
