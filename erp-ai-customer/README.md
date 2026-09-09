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
- Sentence Transformers 本地 Provider、延迟模型加载和 CPU/CUDA 自动选择
- 与厂商无关的 Vector Store 协议和类型化异常
- Chroma 持久化配置、延迟客户端初始化和 CRUD 适配器
- Docker 中用于 Chroma 数据持久化的 `chroma-data` named volume
- Excel 解析结果到 Embedding、Chroma upsert 的知识库索引编排服务
- 原子校验、确定性记录 ID、来源 Metadata 和分类异常处理
- 使用离线确定性向量的真实 Chroma 持久化集成测试
- 查询向量化、Embedding 兼容过滤和 ERP FAQ 结果校验的相似度检索服务
- 检索数量限制、分类异常和真实 Chroma 最近邻检索集成测试

第三阶段已完成 Embedding 抽象层、Sentence Transformers 本地 Provider、Chroma Vector Store、Excel 知识库向量索引服务和相似度检索服务。索引服务为 FAQ 生成稳定 ID 和来源 Metadata，并在工作簿存在校验问题时整批拒绝。检索服务将用户问题向量化，只查询文档类型、Embedding Provider、模型和维度均兼容的记录，并把存储结果转换为经过校验的 ERP FAQ 命中对象。FastAPI 当前尚未组装这些服务，因此应用启动不会下载模型或打开 Chroma。尚未实现上传接口持久化、HTTP 检索接口、LangChain RAG 或大模型问答。

## 项目结构

```text
erp-ai-customer/
├── backend/
│   ├── app/
│   │   ├── api/              # 根路由和版本化 API
│   │   ├── config/           # 配置与日志初始化
│   │   ├── database/         # 后续业务数据库模块
│   │   ├── middleware/       # 请求追踪中间件
│   │   ├── rag/              # Embedding、Vector Store、索引与检索编排模块
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
| `EMBEDDING_DEVICE` | `auto` | `auto/cpu/cuda`；`auto` 优先使用可用 CUDA，否则使用 CPU |
| `EMBEDDING_BATCH_SIZE` | `32` | 批量向量化大小，允许 1–512 |
| `EMBEDDING_NORMALIZE` | `true` | 是否输出单位长度归一化向量 |
| `VECTOR_STORE_PROVIDER` | `chroma` | 向量存储实现标识；当前只允许 Chroma |
| `CHROMA_PERSIST_DIRECTORY` | `data/chroma` | Chroma 持久化目录；相对路径基于项目根目录解析 |
| `CHROMA_COLLECTION_NAME` | `erp_faq` | ERP FAQ Collection 名称，启动时执行格式校验并规范化为小写 |
| `CHROMA_DISTANCE_METRIC` | `cosine` | Collection 距离度量，可选 `cosine/l2/ip` |
| `CHROMA_QUERY_LIMIT` | `5` | 默认相似度查询条数，允许 1–100 |

配置值不符合约束时，应用在启动阶段直接返回明确的 Pydantic 校验错误。

### 本地 Embedding 模型说明

`sentence-transformers` 已声明为生产依赖。Provider 支持文档与查询分别编码、批量大小配置、向量归一化，以及 `auto/cpu/cuda` 设备选择。模型采用延迟加载；只有后续模块创建 `EmbeddingService` 并首次使用 Provider 时，才会从本地缓存读取模型，缓存不存在时再尝试下载 `EMBEDDING_MODEL`。

Embedding 单元测试和知识索引测试使用替身模型，不联网、不下载 `BAAI/bge-small-zh-v1.5`。当前模块只验证 Embedding 编排边界；真实模型下载与推理仍留到独立集成验收，不影响索引服务单元测试。

### Chroma Vector Store 说明

`chromadb` 已声明为生产依赖。当前适配器接收上游生成的向量，支持批量 `upsert`、向量查询、按 ID 删除和计数，并显式禁用 Chroma 的内置 Embedding Function，确保 Embedding 与向量存储职责分离。持久化客户端和 Collection 都采用延迟初始化，Collection 使用配置的距离度量。

当前开发机的 `.venv` 已安装 `chromadb 1.5.9`，并已安装其 Windows 原生扩展所需的 Microsoft Visual C++ x64 运行库。适配器单元测试仍使用 Fake Client 和 Fake Collection；知识索引与检索集成测试使用离线确定性三维向量和真实 `PersistentClient`，验证索引写入、持久化读取、Embedding 兼容过滤及最近邻 FAQ 返回，不启动独立 Chroma Server，也不下载真实 Embedding 模型。

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
python -m pytest backend\tests\test_sentence_transformers_provider.py -q
python -m pytest backend\tests\test_vector_store_settings.py backend\tests\test_chroma_store.py -q
python -m pytest backend\tests\test_knowledge_index_service.py -q
python -m pytest backend\tests\test_knowledge_index_chroma_integration.py -q
python -m pytest backend\tests\test_knowledge_retrieval_service.py -q
python -m pytest backend\tests\test_knowledge_retrieval_chroma_integration.py -q
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

Compose 当前仍只启动 FastAPI 后端，不启动独立 Chroma Server。后端使用 `chroma-data` named volume 挂载 `/app/data/chroma`，为后续在容器内使用 Chroma Persistent Client 保留数据。PostgreSQL 将在聊天记录阶段接入。

> 当前项目创建环境未安装或未启用 Docker CLI，因此 Dockerfile 和 Compose 已通过自动化静态结构测试，但尚未在本机完成镜像构建与容器启动验收。请在安装 Docker Desktop 后执行上述命令完成运行验证。

## 后续开发阶段

1. Excel 知识库解析和数据校验（已完成）
2. Embedding 与 Chroma 向量检索（核心服务已完成：Embedding Provider、Chroma 持久化、Excel 向量索引和相似度检索）
3. LangChain RAG 和 AI 问答接口
4. 聊天记录与 PostgreSQL
5. Vue 或简单 Web 聊天窗口
6. Linux Docker 部署验证
7. 权限、工单和人工客服等企业功能
