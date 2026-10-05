# 贡献指南（Contributing）

感谢你对 PixelForge 的关注！本文件说明如何搭建开发环境、运行测试与提交改动。

## 开发环境

- Python **3.11**（CI 与 lock 文件均基于 3.11；代码尽量保持向后兼容，但未在更低版本验证）
- Node.js（前端可选，仅在你需要改 `web/` 时）

```bash
# 1) 克隆
git clone https://github.com/ElijahZhao/pixelforge-image-restoration-.git
cd pixelforge-image-restoration-

# 2) 安装依赖（推荐用锁定的版本以保证可复现）
pip install -r requirements.lock.txt

# 3) 前端（可选）
cd web && cp .env.local.example .env.local && pnpm install
```

> 本项目**不是** pip 可安装包：`train/` 作为脚本目录使用，`serve/` 以命名空间包方式导入，
> 因此 `pyproject.toml` 只承载 pytest 配置，请勿添加 `[build-system]` / `[project]`。

## 运行测试

```bash
# 全套（训练单元测试 + API 冒烟），应输出 28 passed
python -m pytest

# 仅训练侧（不依赖 pytest 的旧入口，仍保留）
python -m train.tests.run_tests
```

- 测试**全离线、CPU 可跑**，无需 GPU、无需浏览器、无需联网下载权重。
- 需要联网权重（VGG）的测试在离线环境下会自动 `SKIP`，不计为失败。

## 代码风格

- 遵循 PEP 8；提交前请确保无静态告警：
  ```bash
  python -m pyflakes .
  ```
- 前端（`web/`）提交前运行类型检查：
  ```bash
  cd web && pnpm exec tsc --noEmit
  ```

## 提交 Pull Request

1. 从 `main` 切出特性分支：`git checkout -b fix/your-topic`。
2. 保持改动**聚焦**——一个 PR 只做一件事，便于审查。
3. 若修改了行为，请同步更新 `README.md` / `CHANGELOG.md`（在 `Unreleased` 段追加）。
4. 确保 `python -m pytest` 全绿后再提交 PR。
5. PR 描述中说明：**做了什么、为什么、如何验证**。

## 报告问题

- 通过 GitHub Issues 提交，请附：复现步骤、期望结果、实际结果、环境（OS / Python / 是否 GPU）。
- **请勿在 Issue / PR 中粘贴任何令牌、密钥或私密数据。**

## 许可

提交贡献即表示你同意以本项目的 [MIT License](LICENSE) 授权你的贡献。
