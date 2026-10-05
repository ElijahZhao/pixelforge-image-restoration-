# API 参考

推理服务基于 FastAPI，入口在 `serve/app.py`。启动：

```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

启动后 FastAPI 会自动挂载交互式文档：`/docs`（Swagger UI）与 `/openapi.json`。
本文件是手写版本，只写这个服务实际会返回的东西，不重复框架自动生成的部分。

---

## GET /api/health

返回服务状态，以及每个引擎槽位当前是走 ML 还是经典兜底。

```json
{
  "status": "ok",
  "engines": {
    "sr_scale2": "classical",
    "sr_scale4": "ml",
    "lowlight": "ml"
  }
}
```

`engines` 的取值只有两种：

- `"ml"`：`serve/models/` 下有对应权重，模型已加载。
- `"classical"`：没有权重，会走经典兜底（超分用 bicubic + unsharp，低光用自适应伽马）。

`sr_scale2` 默认为 `"classical"` —— 见下面「关于 2×」一节。

---

## POST /api/predict

上传一张图，返回 before / after 的 base64 PNG。

**请求**（`multipart/form-data`）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `image` | file | 必填 | 待处理的图片 |
| `task` | string | `"sr"` | `"sr"` 或 `"lowlight"`，其他值返回 422 |
| `scale` | int | `2` | `2` 或 `4`；其他值会被静默改成 `2` |

**成功响应**（200）

```json
{
  "task": "sr",
  "scale": 4,
  "engine": "ml",
  "before": "<base64 PNG>",
  "after": "<base64 PNG>",
  "lr": "<base64 PNG, 仅当 task=sr 时存在>",
  "note": "一段说明当前引擎与结果的文字"
}
```

字段说明：

- `before` / `after`：两张 PNG 的裸 base64（**没有** `data:` 前缀，前端自己拼）。
  两者在同一分辨率下，可直接叠放对比。
- `engine`：本次实际用的引擎，`"ml"` 或 `"classical"`。判定在服务端按「有没有权重 + 输入是否为真实暗光照片」做，不看客户端请求。
- `lr`：仅超分任务返回。这是模型**真正看到的低清输入**（原图按 scale 缩下去），三栏前端用它做中栏。两栏前端忽略即可。
- `note`：解释为什么用了当前引擎。回退到经典兜底时会说明原因（例如"这张不是暗光照片，跳过 U-Net"）。

**错误响应**

| 状态码 | 触发条件 | `error` 内容 |
|---|---|---|
| 413 | 上传体积 > `MAX_UPLOAD_BYTES`（默认 10 MB） | `upload too large (>N bytes)` |
| 413 | 像素数 > `MAX_INPUT_PIXELS`（默认 4 MP） | `image too large (WxH = N pixels; limit M)` |
| 413 | 图片头声明的尺寸在解码时触发 PIL 炸弹保护 | `image too large (limit N pixels)` |
| 422 | `task` 不是 `sr` / `lowlight` | `task must be 'sr' or 'lowlight'` |
| 422 | 文件不是有效图片 | `uploaded file is not a valid image` |
| 429 | 触发限流 | `too many requests; slow down`，带 `Retry-After: 2` |

注意像素数是在**解码之前**用文件头 `.size` 校验的：字节数小不代表内存小，一张几十 KB 的 PNG 可以声明 8000×8000，`.convert("RGB")` 会按声明尺寸分配内存。所以先卡尺寸、再解码。

---

## 关于 2×

服务对 `scale=2` **有接口、但没有自训权重**：
`serve/models/` 里只有 `sr_generator_scale4.pt`（4× 的 SRResNet）和 `lowlight.pt`。

所以请求 2× 时，`engine` 是 `"classical"`，走 bicubic + unsharp，不是学出来的。
`/api/health` 会如实把 `sr_scale2` 报成 `"classical"`。

要让 2× 用上模型，先训练并导出：

```bash
python train/train.py --task sr --model srcnn --scale 2 \
  --data_root data --epochs 100 --batch_size 16
```

训练完导出到 `serve/models/`，服务重启后会自动加载（`serve/model_loader.py` 负责发现）。

---

## 环境变量

服务的行为由这几个环境变量控制，都有默认值，不设也能跑。

| 变量 | 默认 | 含义 |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `10485760`（10 MB） | 单次上传的字节上限 |
| `MAX_INPUT_PIXELS` | `4000000`（~4 MP） | 解码后的像素数上限，同时作为 PIL 的解压炸弹阈值 |
| `ALLOWED_ORIGINS` | `*` | CORS 允许来源，逗号分隔。**生产环境要设成具体域名** |
| `RATE_LIMIT_CAPACITY` | `30` | 每个 IP 的突发请求数 |
| `RATE_LIMIT_REFILL_PER_SEC` | `0.5` | 令牌补充速率（0.5/s ≈ 30 次/分） |

限流是**进程内内存**实现的（`serve/app.py` 里的 `_rate_buckets`）。这是单实例 Demo 的取舍：
换成 Redis 限流器会多一个依赖和一份运维负担，在当前规模没必要。
如果以后要跑多实例，这个限流器必须挪到共享存储，否则 N 个实例会各自放行，实际放行量是 N 倍。
