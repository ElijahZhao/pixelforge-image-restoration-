# 运维说明

这份文档写的是这个项目**实际**怎么跑、出问题怎么处理。
它不描述监控告警体系——因为这个项目没有,见下面「已知缺口」一节。

---

## 部署形态

当前有两套可用的部署方式，各自独立，不共享代码：

| 形态 | 入口 | 说明 |
|---|---|---|
| Streamlit 公开 Demo | `deploy/streamlit/streamlit_app.py` | 已上线，自包含：经典兜底 + TorchScript 加载 + UI 都在一个文件里。**不依赖 FastAPI** |
| FastAPI 服务 | `serve/app.py` | 完整 HTTP 服务，含 `/api/health` 与 `/api/predict`。前端 `web/` 与它对接 |
| Docker 镜像 | `Dockerfile` | 推理服务的容器化打包（Python 3.11-slim + 内置权重，非 root，带 `HEALTHCHECK`）。只含后端 |
| HF Spaces（备选） | `deploy/hf_space/app.py` | 备选方案，原因见 `deploy/streamlit/DEPLOY_STREAMLIT.md` |

关键点：**Streamlit 那份是自包含的，不调 FastAPI**。所以线上 Demo 挂了，
`/api/health` 是不是 200 跟它没关系，两者要分开看。

---

## 健康检查

`GET /api/health` 返回 `{"status":"ok","engines":{...}}`，用来确认
（a）服务活着、（b）每个引擎槽位当前是 `ml` 还是 `classical`。

**但要知道：这个 endpoint 目前在部署链路里没有任何自动消费方。**
Streamlit app 不调它，`web/lib/api.ts` 只调 `/api/predict`。
它现在的作用是手动排查：`curl -s localhost:8000/api/health` 一眼看出
模型加载没有。没有探针自动打它——这是缺口，不是设计。

---

## 日志

- **推理服务**：直接走 uvicorn 的 stdout/stderr。没有接日志聚合，没有结构化日志。
  `uvicorn serve.app:app` 会在控制台打印每条请求的访问行。
- **训练**：逐 epoch 指标写在 `results/train_log_*.csv`，随仓库提交，
  不是运行期日志——别把它们当实时日志源。

---

## 环境变量

服务端可调项（都有默认值，详见 `docs/API.md`）：

| 变量 | 默认 | 生产建议 |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | **改成具体域名**。`*` 只适合本地开发 |
| `MAX_UPLOAD_BYTES` | 10 MB | 按需要调 |
| `MAX_INPUT_PIXELS` | 4 MP | 按需要调 |
| `RATE_LIMIT_CAPACITY` | 30 | 按流量调 |
| `RATE_LIMIT_REFILL_PER_SEC` | 0.5 | 按流量调 |

前端连后端靠 `NEXT_PUBLIC_API_URL`（见 `web/.env.local.example`）。

---

## 出问题怎么办

**服务返回 429 太多**：有人（或你自己）触发了限流。等 `Retry-After` 秒，
或调高 `RATE_LIMIT_CAPACITY` / `RATE_LIMIT_REFILL_PER_SEC` 后重启。

**超分结果和预期不符**：先 `curl /api/health` 看 `sr_scale4` 是不是 `ml`。
如果是 `classical`，说明权重没加载上——检查 `serve/models/` 下是否有
`sr_generator_scale4.pt`（或 `deploy/streamlit/models/`，两处部署各自带一份）。

**低光结果没变化/说用了经典兜底**：读响应里的 `note` 字段。
它通常会说明"这张不是暗光照片，跳过了 U-Net"。模型是拿 LOL-v1 真实夜景照片训的，
对截图/合成暗色图会主动跳过（否则会把画面压暗）。

**要回滚**：本项目没有自动化回滚机制。回滚 = 回到上一个正常工作的 commit：

```bash
git log --oneline            # 找到目标 commit
git revert <bad-commit>      # 生成一个反向提交（不改写历史）
```

**不要**对这个仓库做 `git reset --hard` 后强推。远端是公开仓库，历史已有分叉风险。

---

## 已知缺口

以下是**目前没有、但一个完整交付通常会有**的东西。列出来是为了不让人误以为它们存在：

- **没有监控告警**：没有 Prometheus / 哨兵 / 任何探针。服务挂了不会有人被通知。
- **没有依赖漏洞扫描的落地**：CI 里加了 `pip-audit`（见 `.github/workflows/ci.yml`），
  但没有对无修复 CVE 的抑制清单（`.pip-audit-ignore`）；一旦某依赖爆出无补丁 CVE，CI 会直接变红，
  届时需要手动加白名单并说明原因，而不是删掉扫描步骤。
- **容器化只覆盖后端**：`Dockerfile` 打的是推理服务（`serve/`），前端 `web/` 仍是独立构建目标，不在镜像内。
- **没有多实例方案**：限流是进程内的（`serve/app.py`），多实例会失效，详见 `docs/API.md`。
- **没有备份/灾备**：无状态服务，也没有需要备份的持久数据，所以这项目前为空。
