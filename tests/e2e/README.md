# E2E 浏览器测试（PixelForge）

用真实无头 Chromium 跑完整用户流程，验证"前端 → 后端 → 图像返回"全链路，并产出截图作为证据。

## 跑起来

```bash
# 1) 安装浏览器驱动（一次性）
pip install playwright && playwright install chromium
cd web && pnpm install && cd ..

# 2) 运行（会自动起后端 uvicorn :8000 与前端 pnpm dev :3000）
python tests/e2e/e2e.py
```

## 它做了什么

1. 启动 FastAPI 后端（`uvicorn serve.app:app --port 8000`）。
2. 启动 Next.js 前端（`pnpm dev`，默认把 `/api/*` 代理到后端，无需额外配置）。
3. 用 Chromium 走两条流程：
   - **超分辨率 4×**：上传 `assets/sample_scene.png` → 选 4× → 点 Enhance → 结果图 + 拖动对比滑块。
   - **低光增强**：上传 `assets/sample_dark.png` → 点 Enhance → 结果图。
4. 断言结果 `before` / `after` 均为有效 base64 PNG（长度 > 200）。
5. 截图保存到 `tests/e2e/screenshots/`（已 gitignore）。

## 产出截图

| 文件 | 内容 |
|---|---|
| `super_resolution_home.png` | 首页 |
| `super_resolution_result.png` | 超分结果 + 对比滑块 |
| `super_resolution_slider_dragged.png` | 滑块拖动后 |
| `low_light_home.png` / `low_light_result.png` | 低光流程 |

> 当前无训练权重，引擎显示 `Classical baseline`；训练并导出权重后自动变为 `Trained PyTorch model`，本测试无需改动即可验证新模型。

## 退出码

- `0`：全部断言通过。
- `1`：有失败（会在日志列出具体原因）。
