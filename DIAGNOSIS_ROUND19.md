# PixelForge 深度审计 · 第 19 轮：末轮盲区清扫（token 实况 / 数值对账 / 端到端可跑性）

> 审计原则：只读探查，零代码改动。本轮清掉前 18 轮遗留的最后三条盲区，并为收官文档做交叉验证。实读 `.git/config`（脱敏）、三份文档数值、并**真实启动后端**跑端到端与边界测试。

## 盲区 1：`.git/config` 里 token 的当前真实状态 → P0 仍未处理（属实且严重）

**实测（脱敏）：**
```
origin  https://oauth2:ghp_***REDACTED***@ghproxy.net/https://github.com/ElijahZhao/pixelforge-image-restoration-.git
```
- `.git/config` 第 10 行**仍含明文 token**，且 remote 经**第三方镜像 `ghproxy.net`** 转发——token 会随每次 push/fetch 穿过第三方服务器。
- **好消息（必须如实记）**：`git log -p --all` 全历史（17 个 commit）**未发现 `ghp_` token 明文**。即 token **从未进入提交历史**，泄露面局限在本地 `.git/config` + 第三方镜像转发，而非公开仓库内容。

**判定**：PROGRESS 的 P0"轮换已暴露的 token"**真实存在、且当前仍未处理**。风险等级：中—高（不是"仓库里挂着 token"，而是"本地配置+镜像链路暴露"）。处理方式只需 `git remote set-url` 换新 token 并撤销旧的，**无需改写 git 历史**。

## 盲区 2：三份文档指标数值逐位对账 → 完全一致（正面结论）

| 指标 | results/README.md | PROGRESS.md | README.md | 一致? |
|---|---|---|---|---|
| Bicubic 2× PSNR | 33.66 * | — | 33.66 * | ✅ |
| Bicubic 2× SSIM | 0.9299 * | — | 0.9299 * | ✅ |
| SRResNet+感知 4× PSNR | 17.20 | 17.20 | 17.20 | ✅ |
| SRResNet+感知 4× SSIM | 0.217 | 0.217 | 0.217 | ✅ |
| 伽马基线低光 | ~15-17 † | 15–17 | ≈15–17 † | ✅ |
| U-Net 低光 PSNR | 19.26 | 19.26 | 19.26 | ✅ |
| U-Net 低光 SSIM | 0.74-0.78 | ≈0.74–0.78 | 0.74–0.78 | ✅ |
| SRCNN 2× | TBD | （未训） | TBD | ✅ |

**结论：数值层零漂移、逐位一致。** 这是一条应肯定的正面事实——说明作者在"把数字抄进三份文档"这件事上是细心的。**问题不在数值一致性，而在数值本身的来源与解释**（见第 17/18 轮：17.20 因 `0.01` 权重+未归一化而失真；19.26 的"胜出"建立在未自测参考带上）。**"三处一致"保证了没抄错，但保证不了数字是对的。**

## 盲区 3：web 前端在真实后端下的端到端可跑性 → 后端全通（前端未实跑浏览器）

**实测**（真实启动 `uvicorn serve.app:app`）：
```
GET  /api/health  -> {"status":"ok","engines":{"sr_scale2":"classical","sr_scale4":"ml","lowlight":"ml"}}
POST /api/predict task=sr       scale=4 -> HTTP 200 | engine=ml | keys=[task,scale,engine,before,after,note] | after_b64=130268 | 0.1s
POST /api/predict task=lowlight scale=2 -> HTTP 200 | engine=ml | keys=[...同上] | after_b64=3680 | 0.0s
POST /api/predict 坏图       -> HTTP 422 {'error': 'uploaded file is not a valid image'}
POST /api/predict 非法task   -> HTTP 422 {'error': "task must be 'sr' or 'lowlight'"}
```

**结论：后端契约干净、可用**：
- 参数名是 `image`（非 `file`）——**首次实测我用错字段名得到 422，是测试错误而非接口 bug**，已纠正。
- 200 响应结构完整（含 `before`/`after` base64、`engine`、`note`），非法输入正确返回 422 带明确 error——与 PROGRESS 2.2 声称一致。
- **但同时再次印证第 16 轮**：`scale=2` 走的是 SR 的 `classical` 回退（因无 scale2 权重），请求响应里 `engine=ml` 仅指 lowlight 路径；`/api/health` 已诚实暴露 `sr_scale2: classical`。
- **未实跑浏览器**：`web/node_modules` 已存在，但本轮未启动 Next.js 前端做真实浏览器点击（沙箱内 `pnpm dev` + Playwright 成本高）。故"前端 UI 端到端"仍属**部分验证**——后端契约已证可用，前端渲染/滑块行为沿用第 4/16 轮实读结论。

## 本轮结论

> 末轮三条盲区全部落地：**token 风险真实且未处理（但未入历史）**、**三份文档数值逐位一致（正面）**、**后端端到端全通（前端 UI 未实跑）**。至此 19 轮审计的证据边界已清晰——**问题从来不是"哪里跑不起来"，而是"跑起来的结果没有被独立验证"**。

## 硬约束遵守情况

- **零代码改动**：`git diff train/ serve/ web/ scripts/` 无输出；本轮仅新增本报告。
- token 探查全程脱敏（只做模式匹配，未打印明文）。
- 后端实跑后已 `pkill` 清理，无残留进程。

## 下一步

三条盲区已清，转入收官：**《修复方案》** 与 **《工程复盘》**。
