from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import sys
import json
import uuid
import asyncio
import logging
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
from rag import RagService
from knowledge_base import KnowledgeBaseService, load_literature_list
import config_data as config

# 大文件上传配置（从配置文件读取）
MAX_FILE_SIZE = config.MAX_FILE_SIZE_MB * 1024 * 1024  # 转换为字节

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    """中间件：限制上传文件大小"""
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="请求体过大")
        return await call_next(request)

# 初始化应用
app = FastAPI(title="RAG Learning Assistant API", version="1.0")

# 添加文件大小限制中间件
app.add_middleware(LimitUploadSizeMiddleware)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
rag_service = None
knowledge_service = None

def init_services():
    global rag_service, knowledge_service
    try:
        rag_service = RagService()
        knowledge_service = KnowledgeBaseService()
        print("Services initialized successfully")
    except Exception as e:
        print(f"Failed to initialize services: {str(e)}")

# 启动时初始化服务
init_services()

# 模型定义
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list = []
    session_id: str = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str = None

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
    try:
        if not rag_service:
            raise HTTPException(status_code=500, detail="RAG service not initialized")
        
        # 构建输入
        input_data = {'input': request.message}
        
        # 使用客户端传入的 session_id 或生成新的
        session_id = request.session_id or str(uuid.uuid4())
        response = rag_service.chain.stream(input_data, {'configurable': {'session_id': session_id}})
        
        # 收集响应
        result = ""
        for chunk in response:
            result += str(chunk)
        
        return {"success": True, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket 对话接口"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")
            
            if not rag_service:
                await websocket.send_json({"type": "error", "content": "RAG service not initialized"})
                continue
            
            try:
                input_data = {'input': message}
                response = rag_service.chain.stream(input_data, {'configurable': {'session_id': session_id}})
                
                for chunk in response:
                    await websocket.send_json({"type": "chunk", "content": str(chunk)})
                
                await websocket.send_json({"type": "done"})
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到知识库"""
    temp_path = None
    try:
        if not knowledge_service:
            raise HTTPException(status_code=500, detail="知识库服务未初始化")
        
        filename = os.path.basename(file.filename)
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
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_e:
                logger.warning(f"清理临时文件失败: {cleanup_e}")

@app.get("/api/files")
async def list_files():
    """获取已上传文件列表"""
    try:
        history_file = RAG_DIR / 'md5.txt'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                files = []
                for line in lines:
                    line = line.strip()
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            files.append(parts[1])
                        else:
                            files.append(parts[0])
                    elif line:
                        files.append(line)
            return {"success": True, "files": files}
        return {"success": True, "files": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/literature")
async def list_literature():
    """获取文献列表"""
    try:
        literature_list = load_literature_list()
        return {"success": True, "literature": literature_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除知识库中的文件"""
    try:
        if not knowledge_service:
            raise HTTPException(status_code=500, detail="知识库服务未初始化")
        
        result = knowledge_service.delete_file(filename)
        
        if result.startswith('[成功]'):
            return {"success": True, "message": result}
        elif result.startswith('[提示]'):
            return {"success": False, "message": result}
        else:
            return {"success": False, "message": result}
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

FRONTEND_DIR = RAG_DIR / 'frontend'

# 静态文件服务（前端）
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    with open(FRONTEND_DIR / 'index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, timeout_keep_alive=120)
