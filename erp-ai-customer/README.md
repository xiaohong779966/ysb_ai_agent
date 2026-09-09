# ERP 智能客服系统

本项目用于建设基于企业知识库的 ERP 软件售后 AI 智能客服系统。当前已完成基础环境和 Excel 知识库解析上传，并开始第三阶段 Embedding 与向量检索模块建设。

## 当前完成范围

已完成：

- Python 3.12 与 FastAPI 后端骨架
- 基于 Pydantic Settings 的环境配置
- 文本和 JSON 标准输出日志
- `X-Request-ID` 请求追踪与访问日志
- 根接口、健康检查、Swagger 和 OpenAPI
- 单后端服务 Dockerfile 与 Docker Compose
- pytest、Ruff 和 Docker 文件静态测试
- Excel 知识库解析、数据校验和上传预览接口
- 与厂商无关的 Embedding Provider 协议
- Embedding 输入、向量维度和数值有效性校验
- 本地 Embedding 模型的环境配置

第三阶段当前只完成 Embedding 抽象层与配置模块。应用启动时不会下载或加载模型，尚未实现 Sentence Transformers 适配器、Chroma 持久化、向量导入、相似度检索、LangChain RAG 或大模型问答。

## 项目结构

```text
erp-ai-customer/
├── backend/
│   ├── app/
│   │   ├── api/              # 根路由和版本化 API
│   │   ├── config/           # 配置与日志初始化
│   │   ├── database/         # 后续业务数据库模块
│   │   ├── middleware/       # 请求追踪中间件
│   │   ├── rag/              # Embedding、向量检索与后续 RAG 模块
│   │   ├── schemas/          # API 响应模型
│   │   └── main.py           # FastAPI 应用工厂
│   ├── tests/                # 单元和接口测试
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/                 # 后续 Vue 或 Web 聊天组件
├── knowledge/                # ERP Excel 知识库样本
├── .env.example
├── docker-compose.yml
├── pytest.ini
├── pyproject.toml
└── README.md
```

## Windows 本地开发

### 1 环境要求

- Windows 10 或 Windows 11
- Python 3.12
- PowerShell
- Docker Desktop（仅 Docker 运行方式需要）

### 2 创建环境并安装依赖

```powershell
cd D:\ysb_ai_agent\erp-ai-customer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-dev.txt
Copy-Item .env.example .env
```

如果 PowerShell 阻止激活脚本，可以不激活虚拟环境，直接使用 `.\.venv\Scripts\python.exe` 执行后续命令。

### 3 启动 FastAPI

```powershell
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后可访问：

| 功能 | 地址 |
| --- | --- |
| 服务信息 | `http://127.0.0.1:8000/` |
| 健康检查 | `http://127.0.0.1:8000/api/v1/health` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |
| OpenAPI JSON | `http://127.0.0.1:8000/openapi.json` |

所有 HTTP 响应都包含 `X-Request-ID`。客户端提供该请求头时服务会透传，否则自动生成 UUID。

## 配置说明

应用从操作系统环境变量和项目根目录 `.env` 读取配置。不要提交包含本地配置或密钥的 `.env` 文件。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | `ERP AI Customer Service` | 服务名称 |
| `APP_VERSION` | `0.1.0` | 服务版本 |
| `APP_ENV` | `local` | `local/development/testing/staging/production` |
| `DEBUG` | `false` | FastAPI 调试模式 |
| `API_V1_PREFIX` | `/api/v1` | 第一版 API 前缀 |
| `HOST` | `0.0.0.0` | 本地监听地址 |
| `PORT` | `8000` | 本地或宿主机映射端口 |
| `LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR/CRITICAL` |
| `LOG_FORMAT` | `text` | `text` 或 `json`；Compose 容器默认覆盖为 `json` |
| `KNOWLEDGE_UPLOAD_MAX_MB` | `10` | Excel 上传大小限制，允许 1–100 MB |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Embedding 实现标识；当前只允许本地 Sentence Transformers |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 后续本地适配器使用的中文 Embedding 模型 |
| `EMBEDDING_DEVICE` | `auto` | `auto/cpu/cuda`，由后续本地适配器解析 |
| `EMBEDDING_BATCH_SIZE` | `32` | 批量向量化大小，允许 1–512 |
| `EMBEDDING_NORMALIZE` | `true` | 是否要求后续适配器输出归一化向量 |

配置值不符合约束时，应用在启动阶段直接返回明确的 Pydantic 校验错误。

## 测试与代码检查

```powershell
python -m pytest -q
python -m ruff check backend
```

也可以运行单个模块的测试：

```powershell
python -m pytest backend\tests\test_settings.py -q
python -m pytest backend\tests\test_logging.py backend\tests\test_request_context.py -q
python -m pytest backend\tests\test_api.py -q
python -m pytest backend\tests\test_docker_files.py -q
python -m pytest backend\tests\test_excel_parser.py backend\tests\test_knowledge_upload.py -q
python -m pytest backend\tests\test_embedding_service.py backend\tests\test_embedding_settings.py -q
```

## Docker 运行

首次运行前创建本地环境文件：

```powershell
Copy-Item .env.example .env
```

验证并启动后端容器：

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f backend
```

验证服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

停止环境：

```powershell
docker compose down
```

Compose 当前只启动 FastAPI 后端。Chroma 将在第三阶段后续模块接入，PostgreSQL 将在聊天记录阶段接入。

> 当前项目创建环境未安装或未启用 Docker CLI，因此 Dockerfile 和 Compose 已通过自动化静态结构测试，但尚未在本机完成镜像构建与容器启动验收。请在安装 Docker Desktop 后执行上述命令完成运行验证。

## 后续开发阶段

1. Excel 知识库解析和数据校验（已完成）
2. Embedding 与 Chroma 向量检索（进行中：已完成抽象层与配置）
3. LangChain RAG 和 AI 问答接口
4. 聊天记录与 PostgreSQL
5. Vue 或简单 Web 聊天窗口
6. Linux Docker 部署验证
7. 权限、工单和人工客服等企业功能

