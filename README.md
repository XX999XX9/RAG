# RAG 学习辅助问答系统

一个基于 Retrieval-Augmented Generation (RAG) 技术的智能学习辅助系统，支持文献管理、混合检索、分源检索等功能。采用 FastAPI + 前端界面的前后端分离架构。

---

## 📋 目录

- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [模块说明](#模块说明)
- [API 接口](#api-接口)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [扩展开发](#扩展开发)
- [安全说明](#安全说明)

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 📄 **多格式文件支持** | txt、pdf、docx、md 文件上传 |
| 📖 **智能章节分割** | 自动识别章节标题，按章节分割文本 |
| 📚 **文献标题识别** | 自动识别书名/论文名，记录文献列表 |
| 🔍 **分源检索** | 支持指定文献检索与全局检索混合使用 |
| 🔗 **邻近 Chunk 扩展** | 检索结果自动补充物理相邻文本块，保证上下文完整 |
| 🔄 **本地重排序** | 支持 Ollama 本地模型重排序 |
| 💬 **会话历史** | 支持多轮对话，流式响应 |
| 📊 **章节元数据** | 保存章节信息，便于追溯 |
| 🌐 **前后端分离** | FastAPI 后端 + 简约前端界面 |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端界面 (HTML/CSS/JS)                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              聊天界面 / 文件上传 / 消息展示              │    │
│  └──────────────────────────┬──────────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────────┘
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 后端服务                            │
│  ┌─────────────────┐     ┌─────────────────┐                  │
│  │   API Routes    │     │    Services     │                  │
│  │  (REST/WebSocket)│     │  (RAG/Knowledge)│                  │
│  └────────┬────────┘     └────────┬────────┘                  │
└──────────┼────────────────────────┼───────────────────────────┘
           │                        │
           └──────────┬─────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        核心服务层                              │
│  ┌─────────────────┐     ┌─────────────────┐                  │
│  │  knowledge_base │     │      rag        │                  │
│  │  (知识库管理)    │     │   (RAG核心)     │                  │
│  └────────┬────────┘     └────────┬────────┘                  │
│           │                        │                           │
│           ├──► chapter_splitter   │                           │
│           │                        ▼                           │
│           │              ┌─────────────────┐                   │
│           │              │  vector_stores │                   │
│           │              │  (向量存储)      │                   │
│           │              └────────┬────────┘                   │
│           │                       │                           │
│           │                       └──► ollama_reranker        │
│           ▼                       ▼                           │
│  ┌─────────────────────────────────────────┐                   │
│  │           Chroma DB (向量数据库)         │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 模块说明

### 1. 前端模块

#### `frontend/index.html` - 前端界面
- **功能**：提供黑白简约风格的聊天界面
- **特性**：
  - 文件上传（点击/拖拽）
  - 实时消息展示
  - WebSocket 流式响应
  - 响应式设计
- **技术栈**：HTML5、CSS3、JavaScript（ES6+）

### 2. 后端模块

#### `backend/main.py` - FastAPI 服务
- **功能**：提供 REST API 和 WebSocket 接口
- **核心接口**：
  - `POST /api/chat` - 对话问答
  - `WebSocket /api/chat/ws` - 流式对话
  - `POST /api/upload` - 文件上传
  - `GET /api/files` - 获取文件列表
  - `DELETE /api/files/{filename}` - 删除文件

### 3. 核心业务模块

#### `rag.py` - RAG 核心服务
- **功能**：构建完整的 RAG 执行链
- **核心组件**：
  - 嵌入模型（DashScope Embeddings）
  - 向量检索器（VectorStoreService）
  - 聊天模型（ChatOpenAI + DeepSeek）
  - 提示词模板（含分源检索逻辑）
- **核心函数**：
  - `detect_reference_intent()`: 检测用户参考意图，匹配文献标题
  - `_merge_neighbor_chunks()`: 合并物理上连续的邻近 chunk

#### `knowledge_base.py` - 知识库管理服务
- **功能**：文档解析、分割、向量化、入库
- **核心流程**：
  1. 文件解析（支持 txt/pdf/docx/md）
  2. MD5 去重校验
  3. 章节分割（调用 `chapter_splitter.py`）
  4. 文献标题识别（调用 ChatOpenAI 模型）
  5. 向量化存储（调用 Chroma），每个 chunk 记录 `chunk_index`

### 4. 数据服务模块

#### `vector_stores.py` - 向量存储服务
- **功能**：管理向量数据库，提供检索能力
- **核心方法**：
  - `hybrid_retrieve()`: 混合检索（关键词+语义）
  - `retrieve_by_title()`: 基于文献标题的元数据检索
  - `hybrid_retrieve_with_reference()`: 分源检索（指定文献+全局）
  - `_expand_with_neighbors()`: 邻近 Chunk 扩展（自动补充物理相邻文本块）
  - 支持 Ollama 重排序

#### `chapter_splitter.py` - 章节分割器
- **功能**：按章节标题分割文本
- **支持的标题格式**：
  - `第一章 标题`、`第一篇 标题`（中文数字）
  - `1. 标题`、`1、标题`（数字编号）
  - `一、标题`、`一. 标题`（中文数字编号）
  - `A. 标题`（字母编号）
  - `【标题】`、`《标题》`（特殊符号）

#### `ollama_reranker.py` - Ollama 重排序器
- **功能**：使用本地 Ollama 模型对检索结果重排序
- **模型支持**：`B-A-M-N/qwen3-reranker-0.6b-fp16:latest`

---

## 🔌 API 接口

### 基础信息

| 属性 | 值 |
|------|------|
| 基础 URL | `http://localhost:8000` |
| API 文档 | `http://localhost:8000/docs` |
| 用户聊天界面 | `http://localhost:8000/chat` |
| 管理员控制台 | `http://localhost:8000/admin` |

### 页面说明

| 页面 | 说明 |
|------|------|
| `/chat` 用户页 | 仅对话问答；文件管理已移除，由管理员在 `/admin` 维护 |
| `/admin` 管理员页 | ① **会话记录**：查看所有用户会话的完整聊天记录，每条 AI 回复末尾以可点击序号展示本次回复的检索参考资料，点击后查看原文与元数据；② **知识库文档**：按来源浏览向量库实存数据，分页查看每个分块的内容与元数据；③ **知识库管理**：上传文件；删除采用回收站机制——移入回收站的文件在检索与模型读取全流程中不可见，可随时恢复或彻底删除 |

### 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 对话问答（HTTP） |
| `/api/chat/ws` | WebSocket | 流式对话 |
| `/api/upload` | POST | 上传文件到知识库（管理员） |
| `/api/files` | GET | 获取已上传文件列表（管理员） |
| `/api/files/{filename}` | DELETE | 删除文件（管理员，软删除：移入回收站） |
| `/api/literature` | GET | 获取文献列表 |
| `/api/admin/documents` | GET | 知识库文档列表（按来源聚合，含分块数量，管理员；`?include_deleted=true` 含回收站分块统计） |
| `/api/admin/documents/{filename}/chunks` | GET | 分页浏览文档分块与元数据（管理员，`?page=1&page_size=20&include_deleted=false`） |
| `/api/admin/sessions` | GET | 用户会话列表（含消息数、最后活跃时间，管理员） |
| `/api/admin/sessions/{session_id}` | GET | 指定会话的完整聊天记录（管理员） |
| `/api/admin/sessions/{session_id}/retrievals` | GET | 指定会话的检索审计记录：每次模型回复引用的分块原文+元数据，按回复顺序排列（管理员） |
| `/api/admin/recycle` | GET | 回收站文件列表（管理员） |
| `/api/admin/recycle/{filename}/restore` | POST | 从回收站恢复文件（管理员） |
| `/api/admin/recycle/{filename}` | DELETE | 彻底删除回收站文件（管理员，物理删除不可恢复） |
| `/health` | GET | 健康检查 |

### 接口详情

#### POST /api/chat

请求体（`session_id` 可选，传入后可续接多轮对话；不传则服务端新建会话并在响应中返回）：
```json
{
    "message": "什么是认知心理学？",
    "session_id": "chat-1756627200000-abc123"
}
```

响应：
```json
{
    "success": true,
    "response": "认知心理学是研究人类认知过程的科学...",
    "session_id": "chat-1756627200000-abc123"
}
```

#### POST /api/upload

请求：`multipart/form-data`
- `file`: 文件（支持 txt/pdf/docx/md）

响应：
```json
{
    "success": true,
    "message": "[成功]内容已经成功载入向量库",
    "filename": "心理学导论.txt"
}
```

---

## 🚀 快速开始

### 1. 环境配置

创建 `.env` 文件：

```env
# 阿里云百炼API
BAILIAN_API_KEY=your_bailian_api_key
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 阿里云DashScope API
DASHSCOPE_API_KEY=your_dashscope_api_key

# 可选：API Key 鉴权（设置后所有 /api/* 接口需要携带 X-API-Key 请求头）
# RAG_API_KEY=your_service_api_key

# 可选：管理员密钥（设置后 /api/admin/* 与管理员页面接口需要携带；
# 未设置时复用 RAG_API_KEY，两者都未设置时管理员接口在本地开放）
# ADMIN_API_KEY=your_admin_api_key

# 可选：每 IP 每分钟对聊天/上传接口的限流阈值（默认 30）
# RATE_LIMIT_PER_MINUTE=30
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

**启动 FastAPI 服务**（端口固定为 8000，由 `config_data.py` 的 `SERVER_PORT` 决定）：
```bash
python backend/main.py
# 或：uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**访问地址**：
- API 文档：http://localhost:8000/docs
- 聊天界面：http://localhost:8000/chat

> 注意：前端页面的 JS 依赖（marked / DOMPurify）由后端 `/static` 提供，请通过 `http://localhost:8000/chat` 访问，不要直接以文件方式打开 index.html。

---

## ⚙️ 配置说明

### 核心配置项（config_data.py）

```python
# 服务端口
SERVER_PORT = 8000

# 文件解析
SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'md']
MAX_FILE_SIZE_MB = 200

# 章节分割（核心配置）
USE_CHAPTER_SPLITTER = True  # 启用章节分割
CHAPTER_OVERLAP_LINES = 2     # 章节重叠行数
CHAPTER_MIN_CONTENT_LENGTH = 50  # 最小章节长度

# 检索权重
KEYWORD_WEIGHT = 0.3  # 关键词权重
SEMANTIC_WEIGHT = 0.7  # 语义权重

# 邻近 Chunk 扩展
USE_LLM_CHUNK_EXPANSION = True  # 启用 LLM 驱动的邻近 chunk 扩展

# 会话历史
MAX_HISTORY_MESSAGES = 20  # 送入模型的最大历史消息条数

# 回收站（软删除）
RECYCLE_RETENTION_DAYS = 0  # 回收站文件保留天数，超过后自动彻底删除；0 表示不启用自动清理

# 文献识别（注意：不包含"根据""按照"等高频词，避免误触发分源检索）
REFERENCE_KEYWORDS = ['参考', '依据', '引用', '出自', '来源', '参见']

# 模型配置
chat_model_name = 'deepseek-v4-pro'  # 基座模型（通过百炼平台接入）
embedding_model_name = 'text-embedding-v4'  # 嵌入模型

# Ollama重排序
USE_OLLAMA_RERANKER = True  # 启用本地重排序
OLLAMA_RERANKER_MODEL = 'bbjson/bge-reranker-base:latest'
```

### 启用 Ollama 重排序

1. 启动 Ollama 服务：
```bash
ollama serve
```

2. 拉取模型：
```bash
ollama pull bbjson/bge-reranker-base:latest
```

3. 修改配置：
```python
USE_OLLAMA_RERANKER = True
```

---

## 📝 使用示例

### 示例1：通过 API 上传文档

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@心理学导论.txt"
```

### 示例2：通过 API 提问

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是认知心理学？"}'
```

### 示例3：使用 Python SDK

```python
from rag import RagService
import config_data as config

rag = RagService()

# 提问
response = rag.chain.stream(
    {'input': '什么是认知心理学？'},
    config.session_config
)

# 流式输出
for chunk in response:
    print(chunk, end='', flush=True)
```

---

## 🔧 扩展开发

### 添加新的文件类型支持

在 `knowledge_base.py` 的 `upload_by_file` 函数中添加对应的读取分支：

```python
elif file_ext == 'new_type':
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
```

同时在 `config_data.py` 的 `SUPPORTED_FILE_TYPES` 中登记新扩展名。

### 自定义章节标题模式

在 `config_data.py` 中添加：

```python
CHAPTER_HEADING_PATTERNS = [
    # 已有模式...
    r'^自定义模式.*$',  # 添加新模式
]
```

### 添加新的 API 接口

在 `backend/main.py` 中添加：

```python
@app.get("/api/custom")
async def custom_endpoint():
    # 自定义逻辑
    return {"message": "Custom endpoint"}
```

---

## 📁 项目结构

```
RAG/
├── backend/
│   └── main.py             # FastAPI 后端服务（含 /api/admin 管理员接口）
├── frontend/
│   ├── index.html          # 用户页（仅对话）
│   ├── admin.html          # 管理员控制台（会话审计 / 文档浏览 / 知识库管理）
│   ├── marked.min.js       # Markdown 渲染库（本地化）
│   └── purify.min.js       # DOMPurify 消毒库（本地化，防 XSS）
├── rag.py                  # RAG核心服务（含检索审计记录落盘）
├── knowledge_base.py       # 知识库管理
├── vector_stores.py        # 向量存储服务（含文档浏览方法）
├── file_history_store.py   # 会话历史存储
├── chapter_splitter.py     # 章节分割器
├── ollama_reranker.py      # Ollama重排序器
├── config_data.py          # 配置文件
├── requirements.txt        # Python依赖
├── 修复报告.md             # 问题修复报告
├── literature_list.json    # 文献列表
├── chroma_db/              # 向量数据库
├── chat_history/           # 会话历史
├── retrieval_history/      # 检索审计记录（管理员页面数据源）
└── .env                    # 环境变量
```

---

## 📌 关键说明

1. **数据持久化**：向量数据存储在 `./chroma_db/`，会话历史存储在 `./chat_history/`，文献列表存储在 `./literature_list.json`
2. **去重机制**：使用 MD5 校验避免重复入库
3. **回收站机制（软删除）**：删除文件时向量数据物理保留，仅在元数据打 `deleted='true'` 不可用标签，检索（语义/关键词/指定文献/邻近 chunk 扩展）与模型读取全流程通过存活过滤器自动屏蔽；可从回收站一键恢复（写回 MD5 与文献记录），或彻底删除（物理清除）。设置 `RECYCLE_RETENTION_DAYS > 0` 后服务每小时自动清理超期文件
4. **章节信息**：章节标题、编号等元数据会保存到向量库
5. **流式响应**：通过 WebSocket 实现打字机效果
6. **模型分配**：
   - 知识入库：云端 DashScope Embeddings
   - 检索：云端 DashScope
   - 重排序：可选本地 Ollama
   - 聊天：云端 ChatOpenAI (DeepSeek V4 Pro)
7. **邻近 Chunk 扩展**：检索时自动补充排名靠前结果的物理相邻文本块，连续 chunk 会被合并为完整文档，确保模型获得完整的上下文
8. **分源检索**：当用户使用"参考""依据"等词汇时，一半检索从指定文献中获取，另一半从全局检索，回复时分开展示
9. **多轮对话**：前端为每个对话维护 `sessionId` 并随消息发送，服务端按会话文件持久化历史（截断保留最近 `MAX_HISTORY_MESSAGES` 条）

---

## 🔒 安全说明

- **CORS**：默认仅允许 `http://localhost:8000` / `http://127.0.0.1:8000`
- **API Key（可选）**：在 `.env` 中设置 `RAG_API_KEY` 后，所有 `/api/*` 接口要求 `X-API-Key` 请求头；网页端可通过 `http://localhost:8000/chat?key=xxx` 首次注入并自动缓存
- **限流**：`/api/chat`、`/api/upload` 默认每 IP 每分钟 30 次（`RATE_LIMIT_PER_MINUTE` 可调）
- **XSS 防护**：AI 回复经 DOMPurify 消毒后渲染；文件名等动态内容统一 HTML 转义
- **会话安全**：`session_id` 仅允许字母/数字/下划线/连字符（≤64 位），防止路径遍历
- **公网部署建议**：前置 nginx 启用 TLS，并务必设置 `RAG_API_KEY`

> 历史遗留问题的完整修复记录见 [修复报告.md](修复报告.md)。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License
