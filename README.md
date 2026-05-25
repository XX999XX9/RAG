# RAG 心理学问答系统

一个基于 Retrieval-Augmented Generation (RAG) 技术的心理学问答系统，支持按章节分割文本、混合检索、本地重排序等功能。采用 FastAPI + 前端界面的前后端分离架构。

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

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 📄 **多格式文件支持** | txt、pdf、docx、md 文件上传 |
| 📖 **智能章节分割** | 自动识别章节标题，按章节分割文本 |
| 🔍 **混合检索** | 关键词检索 + 语义检索 |
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
  - 聊天模型（ChatTongyi）
  - 提示词模板

#### `knowledge_base.py` - 知识库管理服务
- **功能**：文档解析、分割、向量化、入库
- **核心流程**：
  1. 文件解析（支持 txt/pdf/docx/md）
  2. MD5 去重校验
  3. 章节分割（调用 `chapter_splitter.py`）
  4. 向量化存储（调用 Chroma）

### 4. 数据服务模块

#### `vector_stores.py` - 向量存储服务
- **功能**：管理向量数据库，提供检索能力
- **核心方法**：
  - `hybrid_retrieve()`: 混合检索（关键词+语义）
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
| 基础 URL | `http://localhost:8989` |
| API 文档 | `http://localhost:8989/docs` |
| 聊天界面 | `http://localhost:8989/chat` |

### 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 对话问答（HTTP） |
| `/api/chat/ws` | WebSocket | 流式对话 |
| `/api/upload` | POST | 上传文件到知识库 |
| `/api/files` | GET | 获取已上传文件列表 |
| `/api/files/{filename}` | DELETE | 删除指定文件 |
| `/health` | GET | 健康检查 |

### 接口详情

#### POST /api/chat

请求体：
```json
{
    "message": "什么是认知心理学？",
    "history": []
}
```

响应：
```json
{
    "success": true,
    "response": "认知心理学是研究人类认知过程的科学..."
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
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

**启动 FastAPI 服务**：
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8989 --reload
```

**访问地址**：
- API 文档：http://localhost:8989/docs
- 聊天界面：http://localhost:8989/chat

---

## ⚙️ 配置说明

### 核心配置项（config_data.py）

```python
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

# Ollama重排序
USE_OLLAMA_RERANKER = False  # 启用本地重排序
OLLAMA_RERANKER_MODEL = 'B-A-M-N/qwen3-reranker-0.6b-fp16:latest'
```

### 启用 Ollama 重排序

1. 启动 Ollama 服务：
```bash
ollama serve
```

2. 拉取模型：
```bash
ollama pull B-A-M-N/qwen3-reranker-0.6b-fp16:latest
```

3. 修改配置：
```python
USE_OLLAMA_RERANKER = True
```

---

## 📝 使用示例

### 示例1：通过 API 上传文档

```bash
curl -X POST http://localhost:8989/api/upload \
  -F "file=@心理学导论.txt"
```

### 示例2：通过 API 提问

```bash
curl -X POST http://localhost:8989/api/chat \
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

在 `knowledge_base.py` 的 `parse_file` 函数中添加：

```python
def parse_file(file_obj, file_type: str) -> str:
    if file_type == 'new_type':
        # 添加新文件类型的解析逻辑
        pass
```

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
│   └── main.py             # FastAPI 后端服务
├── frontend/
│   └── index.html          # 前端界面（黑白简约风格）
├── rag.py                  # RAG核心服务
├── knowledge_base.py       # 知识库管理
├── vector_stores.py        # 向量存储服务
├── chapter_splitter.py     # 章节分割器
├── ollama_reranker.py      # Ollama重排序器
├── file_history_store.py   # 会话历史存储
├── config_data.py          # 配置文件
├── requirements.txt        # Python依赖
├── chroma_db/              # 向量数据库
├── chat_history/           # 会话历史
└── .env                    # 环境变量
```

---

## 📌 关键说明

1. **数据持久化**：向量数据存储在 `./chroma_db/`，会话历史存储在 `./chat_history/`
2. **去重机制**：使用 MD5 校验避免重复入库
3. **章节信息**：章节标题、编号等元数据会保存到向量库
4. **流式响应**：通过 WebSocket 实现打字机效果
5. **模型分配**：
   - 知识入库：云端 DashScope Embeddings
   - 检索：云端 DashScope
   - 重排序：可选本地 Ollama
   - 聊天：云端 ChatTongyi

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License
