from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import os
import sys
import json
import uuid
import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 RAG 服务（上级目录）
RAG_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(RAG_DIR))
from rag import RagService, RETRIEVAL_HISTORY_DIR
from knowledge_base import KnowledgeBaseService, load_literature_list
from file_history_store import (
    is_valid_session_id, HISTORY_BASE_DIR, FileChatMessageHistory
)
import config_data as config

# 大文件上传配置（从配置文件读取）
MAX_FILE_SIZE = config.MAX_FILE_SIZE_MB * 1024 * 1024  # 转换为字节

# 可选 API Key 鉴权：在 .env 中设置 RAG_API_KEY 后，所有 /api/* 接口均要求携带
API_KEY = os.getenv('RAG_API_KEY')
# 管理员鉴权：可单独设置 ADMIN_API_KEY；未设置时复用 RAG_API_KEY（两者都未设置时管理员接口在本地开放）
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY') or API_KEY

# 简单限流配置：每 IP 每分钟对昂贵接口（聊天/上传）的最大请求数
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))


class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """中间件：限制上传文件大小"""
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_FILE_SIZE:
                    return JSONResponse(status_code=413, content={"detail": "请求体过大"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "无效的 content-length 请求头"})
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """中间件：可选的 API Key 鉴权（设置 RAG_API_KEY 环境变量后启用）"""
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api"):
            required_key = API_KEY
            # 管理员接口使用独立（或复用的）密钥
            if path.startswith("/api/admin") and ADMIN_API_KEY:
                required_key = ADMIN_API_KEY
            if required_key:
                key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
                if key != required_key:
                    return JSONResponse(status_code=401, content={"detail": "无效的 API Key"})
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """中间件：对昂贵接口做简单的每 IP 滑动窗口限流"""
    def __init__(self, app):
        super().__init__(app)
        self._history = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith(("/api/chat", "/api/upload")):
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            window = self._history[client_ip]
            while window and window[0] < now - 60:
                window.popleft()
            if len(window) >= RATE_LIMIT_PER_MINUTE:
                return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后再试"})
            window.append(now)
        return await call_next(request)


# 初始化服务
rag_service = None
knowledge_service = None


def init_services():
    global rag_service, knowledge_service
    # 初始化失败直接抛出异常（fail-fast），避免服务带病运行
    rag_service = RagService()
    knowledge_service = KnowledgeBaseService()
    print("Services initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_services()
    # 回收站定期自动清理（RECYCLE_RETENTION_DAYS > 0 时启用，每小时检查一次）
    if config.RECYCLE_RETENTION_DAYS > 0:
        stop_event = threading.Event()

        def _recycle_cleaner():
            while not stop_event.wait(3600):
                try:
                    knowledge_service.clean_expired_recycle()
                except Exception:
                    logger.exception("回收站定期清理执行失败")

        threading.Thread(target=_recycle_cleaner, daemon=True, name='recycle-cleaner').start()
        logger.info(f"回收站自动清理已启用: 保留 {config.RECYCLE_RETENTION_DAYS} 天")
    yield


# 初始化应用
app = FastAPI(title="RAG Learning Assistant API", version="1.0", lifespan=lifespan)

# 添加中间件（注意：Starlette 按添加的逆序执行，以下三层都会生效）
app.add_middleware(LimitUploadSizeMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

# 配置 CORS：仅允许本机部署的常见来源（前后端同源部署时基本不依赖 CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型定义
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # 由服务端管理会话历史，客户端无需（也不应）传对话内容

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str = None


def _resolve_session_id(client_session_id) -> str:
    """使用客户端传入的 session_id（需通过安全校验），否则生成新会话"""
    if client_session_id and is_valid_session_id(client_session_id):
        return str(client_session_id)
    return str(uuid.uuid4())


def _collect_stream(input_data, stream_config) -> str:
    """在线程中执行同步流式调用并收集完整结果，避免阻塞事件循环"""
    chunks = []
    for chunk in rag_service.chain.stream(input_data, stream_config):
        chunks.append(str(chunk))
    return "".join(chunks)


async def _stream_to_websocket(websocket: WebSocket, input_data: dict, stream_config: dict):
    """在线程中消费同步流式生成器，并通过 asyncio 队列把块转发给 WebSocket"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for chunk in rag_service.chain.stream(input_data, stream_config):
                loop.call_soon_threadsafe(queue.put_nowait, str(chunk))
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        await websocket.send_json({"type": "chunk", "content": item})

# API 路由

@app.get("/")
async def root():
    return {"message": "RAG Psychology QA API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """对话接口"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    session_id = _resolve_session_id(request.session_id)
    # session_id 随链路传递，用于服务端保存检索审计记录
    input_data = {'input': request.message, 'session_id': session_id}

    try:
        result = await asyncio.to_thread(
            _collect_stream, input_data, {'configurable': {'session_id': session_id}}
        )
        return {"success": True, "response": result, "session_id": session_id}
    except Exception:
        logger.exception("对话处理失败")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

@app.websocket("/api/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket 对话接口"""
    # WebSocket 无法携带自定义请求头，API Key 通过查询参数传递
    if API_KEY and websocket.query_params.get("api_key") != API_KEY:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    session_id = None
    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                await websocket.send_json({"type": "error", "content": "无效的消息格式"})
                continue

            if not isinstance(message_data, dict):
                await websocket.send_json({"type": "error", "content": "无效的消息格式"})
                continue

            message = str(message_data.get("message", "") or "").strip()
            if not message:
                continue

            if not rag_service:
                await websocket.send_json({"type": "error", "content": "RAG service not initialized"})
                continue

            # 首条消息确定会话：优先使用客户端传入且合法的 session_id
            if session_id is None:
                session_id = _resolve_session_id(message_data.get("session_id"))

            try:
                # session_id 随链路传递，用于服务端保存检索审计记录
                input_data = {'input': message, 'session_id': session_id}
                stream_config = {'configurable': {'session_id': session_id}}
                await _stream_to_websocket(websocket, input_data, stream_config)
                await websocket.send_json({"type": "done"})
            except Exception:
                logger.exception("WebSocket 对话处理失败")
                await websocket.send_json({"type": "error", "content": "服务内部错误，请稍后重试"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到知识库"""
    temp_path = None
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")

        filename = os.path.basename(file.filename or "")
        if not filename:
            raise HTTPException(status_code=400, detail="无效的文件名")

        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        if file_ext not in config.SUPPORTED_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(config.SUPPORTED_FILE_TYPES)}")

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}', dir='/tmp') as temp_file:
            temp_path = temp_file.name

        file_size = 0
        chunk_size = 65536
        chunk_count = 0
        max_chunks = (MAX_FILE_SIZE // chunk_size) + 100

        with open(temp_path, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                file_size += len(chunk)
                chunk_count += 1

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail=f"文件大小超过限制（最大{config.MAX_FILE_SIZE_MB}MB），当前文件大小：{file_size / 1024 / 1024:.2f}MB")

                if chunk_count > max_chunks:
                    raise HTTPException(status_code=400, detail="文件上传异常，可能是恶意文件")

        logger.info(f"文件接收完成: {filename}，大小: {file_size / 1024 / 1024:.2f}MB，开始处理...")

        result = await asyncio.to_thread(knowledge_service.upload_by_file, temp_path, filename)

        if result.startswith('[成功]'):
            logger.info(f"文件处理成功: {filename}")
            return {"success": True, "message": result, "filename": filename, "size": file_size}
        elif result.startswith('[跳过]'):
            logger.info(f"文件跳过: {filename}, 原因: {result}")
            return {"success": False, "message": result, "filename": filename, "size": file_size}
        else:
            logger.error(f"文件处理失败: {filename}, 原因: {result}")
            raise HTTPException(status_code=500, detail=result)
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="文件上传超时，请检查文件大小或网络状况")
    except Exception:
        logger.exception(f"文件上传失败: {getattr(file, 'filename', '')}")
        raise HTTPException(status_code=500, detail="文件上传失败，请稍后重试")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_e:
                logger.warning(f"清理临时文件失败: {cleanup_e}")

@app.get("/api/files")
async def list_files():
    """获取已上传文件列表（数据源：向量库存活分块聚合，排除回收站文件；md5.txt 仅用于上传查重）"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        files = await asyncio.to_thread(knowledge_service.list_files)
        return {"success": True, "files": files}
    except HTTPException:
        raise
    except Exception:
        logger.exception("获取文件列表失败")
        raise HTTPException(status_code=500, detail="获取文件列表失败")

@app.get("/api/literature")
async def list_literature():
    """获取文献列表"""
    try:
        literature_list = load_literature_list()
        return {"success": True, "literature": literature_list}
    except Exception:
        logger.exception("获取文献列表失败")
        raise HTTPException(status_code=500, detail="获取文献列表失败")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除知识库中的文件（软删除：移入回收站，打不可用标签，检索全流程屏蔽）"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")

        result = await asyncio.to_thread(knowledge_service.soft_delete_file, filename)

        if result.startswith('[成功]'):
            return {"success": True, "message": result}
        elif result.startswith('[提示]'):
            return {"success": False, "message": result}
        else:
            raise HTTPException(status_code=500, detail=result)
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"删除文件失败: {filename}")
        raise HTTPException(status_code=500, detail="删除文件失败，请稍后重试")

@app.get("/api/admin/recycle")
async def admin_list_recycle():
    """回收站文件列表"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        items = await asyncio.to_thread(knowledge_service.list_recycle_bin)
        return {"success": True, "items": items}
    except HTTPException:
        raise
    except Exception:
        logger.exception("获取回收站列表失败")
        raise HTTPException(status_code=500, detail="获取回收站列表失败")

@app.post("/api/admin/recycle/{filename}/restore")
async def admin_restore_file(filename: str):
    """从回收站恢复文件（去除不可用标签，恢复检索可见性）"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        result = await asyncio.to_thread(knowledge_service.restore_file, filename)
        if result.startswith('[成功]'):
            return {"success": True, "message": result}
        return {"success": False, "message": result}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"恢复回收站文件失败: {filename}")
        raise HTTPException(status_code=500, detail="恢复失败，请稍后重试")

@app.delete("/api/admin/recycle/{filename}")
async def admin_purge_file(filename: str):
    """彻底删除回收站文件（物理删除向量分块与全部记录，不可恢复）"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=503, detail="知识库服务未初始化")
        result = await asyncio.to_thread(knowledge_service.purge_file, filename)
        if result.startswith('[成功]') or result.startswith('[提示]'):
            return {"success": True, "message": result}
        raise HTTPException(status_code=500, detail=result)
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"彻底删除回收站文件失败: {filename}")
        raise HTTPException(status_code=500, detail="彻底删除失败，请稍后重试")

# ==================== 管理员接口 ====================

@app.get("/api/admin/documents")
async def admin_list_documents(include_deleted: bool = False):
    """知识库文档列表（按来源聚合，含分块数量；include_deleted=true 时同时返回回收站分块统计）"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG 服务未初始化")
        documents = await asyncio.to_thread(
            rag_service.vector_service.list_documents, include_deleted
        )
        return {"success": True, "documents": documents}
    except HTTPException:
        raise
    except Exception:
        logger.exception("获取知识库文档列表失败")
        raise HTTPException(status_code=500, detail="获取知识库文档列表失败")


@app.get("/api/admin/documents/{filename}/chunks")
async def admin_get_document_chunks(filename: str, page: int = 1, page_size: int = 20,
                                     include_deleted: bool = False):
    """分页浏览指定文档的向量分块（内容 + 元数据；include_deleted=true 时包含回收站分块）"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG 服务未初始化")
        if not filename:
            raise HTTPException(status_code=400, detail="无效的文件名")
        result = await asyncio.to_thread(
            rag_service.vector_service.get_document_chunks, filename, page, page_size,
            include_deleted
        )
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"浏览文档分块失败: {filename}")
        raise HTTPException(status_code=500, detail="浏览文档分块失败")


@app.get("/api/admin/sessions")
async def admin_list_sessions():
    """用户会话列表（含消息数、最后活跃时间、预览）"""
    try:
        sessions = []
        if HISTORY_BASE_DIR.exists():
            for p in HISTORY_BASE_DIR.iterdir():
                if not p.is_file():
                    continue
                try:
                    stat = p.stat()
                    history = FileChatMessageHistory(p.name, str(HISTORY_BASE_DIR))
                    messages = history.messages
                    first_user = next(
                        (m.content for m in messages if getattr(m, 'type', '') == 'human'), ''
                    )
                    sessions.append({
                        'session_id': p.name,
                        'messages': len(messages),
                        'preview': str(first_user)[:60],
                        'last_active': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    })
                except Exception as parse_e:
                    logger.warning(f"解析会话文件失败: {p.name}, {parse_e}")
        sessions.sort(key=lambda s: s['last_active'], reverse=True)
        return {"success": True, "sessions": sessions}
    except Exception:
        logger.exception("获取会话列表失败")
        raise HTTPException(status_code=500, detail="获取会话列表失败")


@app.get("/api/admin/sessions/{session_id}")
async def admin_get_session(session_id: str):
    """指定会话的完整聊天记录"""
    try:
        if not is_valid_session_id(session_id):
            raise HTTPException(status_code=400, detail="非法的 session_id")
        history = FileChatMessageHistory(session_id, str(HISTORY_BASE_DIR))
        messages = [
            {'role': getattr(m, 'type', 'unknown'), 'content': m.content}
            for m in history.messages
        ]
        return {"success": True, "session_id": session_id, "messages": messages}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"获取会话记录失败: {session_id}")
        raise HTTPException(status_code=500, detail="获取会话记录失败")


@app.get("/api/admin/sessions/{session_id}/retrievals")
async def admin_get_retrievals(session_id: str):
    """
    指定会话的检索审计记录（JSONL 按追加顺序，第 N 条对应第 N 次模型回复）
    """
    try:
        if not is_valid_session_id(session_id):
            raise HTTPException(status_code=400, detail="非法的 session_id")
        retrievals = []
        record_file = RETRIEVAL_HISTORY_DIR / f'{session_id}.jsonl'
        if record_file.exists():
            with open(record_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        retrievals.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"检索记录行解析失败: {session_id}")
        return {"success": True, "session_id": session_id, "retrievals": retrievals}
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"获取检索记录失败: {session_id}")
        raise HTTPException(status_code=500, detail="获取检索记录失败")


# ==================== 静态资源与页面 ====================

FRONTEND_DIR = RAG_DIR / 'frontend'

# 静态文件服务（前端）
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    with open(FRONTEND_DIR / 'index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open(FRONTEND_DIR / 'admin.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, timeout_keep_alive=120)
