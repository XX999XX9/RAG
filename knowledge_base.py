#离线流程：知识库更新服务
import logging
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
import sys
import hashlib
import json
from pathlib import Path
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import fitz
from docx import Document
import jieba

# MD5 文件操作锁（线程安全）
md5_lock = threading.Lock()

# 导入章节分割器
if config.USE_CHAPTER_SPLITTER:
    from chapter_splitter import ChapterSplitter

# RAG 目录的绝对路径
RAG_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(RAG_DIR))

# 使用绝对路径加载 RAG 目录下的 .env 文件
env_path = RAG_DIR / '.env'
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv('BAILIAN_API_KEY')
base_url = os.getenv('BAILIAN_BASE_URL')

# 使用绝对路径的 md5 文件和向量库目录
MD5_FILE_PATH = RAG_DIR / 'md5.txt'
PERSIST_DIRECTORY = RAG_DIR / 'chroma_db'
LITERATURE_LIST_PATH = RAG_DIR / 'literature_list.json'

literature_lock = threading.Lock()


def identify_literature_title(filename: str, text_sample: str) -> str:
    """
    使用模型识别文件的真实文献标题（书名、论文名等）
    """
    try:
        chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=api_key,
            base_url=base_url
        )
        prompt = f"""请从以下文件名和文本内容中，识别出该文件的真实文献标题（书名、论文名等）。
要求：
1. 只返回标题本身，不要任何解释或额外文字
2. 如果是书籍，返回书名（去掉作者、出版社、网站来源等附加信息）
3. 如果是论文，返回论文标题
4. 如果无法识别，返回原始文件名（去掉扩展名）

文件名：{filename}

文本内容前500字：
{text_sample[:500]}

文献标题："""
        response = chat_model.invoke(prompt)
        title = response.content.strip().strip('"').strip('《》').strip()
        logger.info(f"模型识别文献标题: {filename} -> {title}")
        return title
    except Exception as e:
        logger.error(f"模型识别文献标题失败: {str(e)}，使用文件名作为标题")
        return filename.rsplit('.', 1)[0] if '.' in filename else filename


def load_literature_list() -> list:
    """
    加载文献列表
    """
    with literature_lock:
        try:
            if LITERATURE_LIST_PATH.exists():
                with open(LITERATURE_LIST_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"加载文献列表失败: {str(e)}")
            return []


def save_literature_item(filename: str, title: str):
    """
    保存一条文献记录到文献列表
    """
    with literature_lock:
        try:
            literature_list = []
            if LITERATURE_LIST_PATH.exists():
                with open(LITERATURE_LIST_PATH, 'r', encoding='utf-8') as f:
                    literature_list = json.load(f)
            
            for item in literature_list:
                if item.get('filename') == filename:
                    item['title'] = title
                    break
            else:
                literature_list.append({
                    'filename': filename,
                    'title': title,
                    'add_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            with open(LITERATURE_LIST_PATH, 'w', encoding='utf-8') as f:
                json.dump(literature_list, f, ensure_ascii=False, indent=2)
            logger.info(f"文献记录保存成功: {filename} -> {title}")
        except Exception as e:
            logger.error(f"文献记录保存失败: {str(e)}")


def remove_literature_item(filename: str):
    """
    从文献列表中删除一条记录
    """
    with literature_lock:
        try:
            if not LITERATURE_LIST_PATH.exists():
                return
            with open(LITERATURE_LIST_PATH, 'r', encoding='utf-8') as f:
                literature_list = json.load(f)
            literature_list = [item for item in literature_list if item.get('filename') != filename]
            with open(LITERATURE_LIST_PATH, 'w', encoding='utf-8') as f:
                json.dump(literature_list, f, ensure_ascii=False, indent=2)
            logger.info(f"文献记录删除成功: {filename}")
        except Exception as e:
            logger.error(f"文献记录删除失败: {str(e)}")

#文件解析函数
def parse_file(file_obj, file_type: str) -> str:
    """
    解析不同类型文件为纯文本
    file_obj: streamlit上传的文件对象（BytesIO）
    file_type: 文件类型（txt/pdf/docx/md）
    """
    if file_type == 'txt':
        return file_obj.getvalue().decode('utf-8')
    elif file_type == 'pdf':
        doc = fitz.open(stream=file_obj.getvalue(), filetype="pdf")
        text = ''.join(page.get_text() or '' for page in doc)
        doc.close()
        if not text.strip():
            raise RuntimeError("PDF内容为空，无法提取文字。可能原因：1)扫描件PDF 2)图片型PDF 3)特殊编码PDF")
        return text
    elif file_type == 'docx':
        doc = Document(file_obj)
        lines = []
        for para in doc.paragraphs:
            if config.DOCX_IGNORE_HEADER_FOOTER and (para.text.strip() == "" or para.style.name in ['Header', 'Footer']):
                continue
            lines.append(para.text)
        return '\n'.join(lines)
    elif file_type == 'md':
        return file_obj.getvalue().decode('utf-8')
    else:
        raise ValueError(f"不支持的文件类型：{file_type}")


def check_md5(md5_str:str):
    """
    检查传入的md5字符串是否已经被处理过了
        return False(md5未处理过)   True（已经处理过，已有记录）
    """
    with md5_lock:
        try:
            if not MD5_FILE_PATH.exists():
                MD5_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                MD5_FILE_PATH.write_text('', encoding='utf-8')
                logger.info(f"MD5文件不存在，创建新文件: {MD5_FILE_PATH}")
                return False
            for line in MD5_FILE_PATH.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                stored_md5 = line.split('|')[0] if '|' in line else line
                if stored_md5 == md5_str:
                    return True
            return False
        except Exception as e:
            logger.error(f"MD5校验失败: {str(e)}")
            return False

def save_md5(md5_str:str, filename:str):
    """
    将传入的md5字符串和文件名，记录到文件内保存
    """
    with md5_lock:
        try:
            with open(MD5_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{md5_str}|{filename}\n")
            logger.info(f"MD5保存成功：{md5_str} - {filename}")
        except Exception as e:
            logger.error(f"MD5保存失败: {str(e)}")

def get_string_md5(input_str:str,encoding='utf-8'):
    """
    获取传入字符串的md5值
    """
    #将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding = encoding)

    #创建md5对象
    md5_obj = hashlib.md5() #得到md5对象
    md5_obj.update(str_bytes)   #更新内容（传入即将要转换的字符串数组）
    md5_hex = md5_obj.hexdigest()   #得到md5的十六进制字符串

    return md5_hex


def extract_keywords(text: str) -> str:
    """
    从文本中提取关键词，返回逗号分隔的字符串
    ChromaDB要求metadata值为字符串或数字，不能是列表
    """
    words = jieba.cut(text)
    keywords = []
    for word in words:
        if word not in config.KEYWORD_STOP_WORDS and len(word) >= config.KEYWORD_MIN_LENGTH:
            keywords.append(word)
    result = list(set(keywords))[:10]
    return ','.join(result) if result else ''


def _split_text_by_max_length(text: str, max_length: int = 8191) -> list:
    """
    将文本分割成不超过最大长度的片段
    用于满足嵌入模型的输入长度限制(最大8192字符)
    
    Args:
        text: 待分割的文本
        max_length: 最大长度限制（默认8191，略小于DashScope的8192限制）
    
    Returns:
        分割后的文本片段列表
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    separators = ['\n\n', '\n', '。', '！', '？', '；', '：', '. ', '! ', '? ', '; ', ': ']
    
    for separator in separators:
        parts = text.split(separator)
        current_chunk = ''
        
        for part in parts:
            if len(current_chunk) + len(part) + len(separator) <= max_length:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if all(len(chunk) <= max_length for chunk in chunks):
            logger.debug(f"文本分割完成，使用分隔符'{separator}'，原长度 {len(text)}，分割为 {len(chunks)} 段")
            return chunks
        
        chunks = []
    
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    logger.debug(f"文本分割完成，使用强制分割，原长度 {len(text)}，分割为 {len(chunks)} 段")
    return chunks


class KnowledgeBaseService(object):
    def __init__(self):
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        try:
            embedding = DashScopeEmbeddings(model=config.embedding_model_name)
            logger.info(f"使用云端 DashScope 嵌入模型: {config.embedding_model_name}")
            
            self.chroma = Chroma(
                collection_name=config.collection_name,
                embedding_function=embedding,
                persist_directory=str(PERSIST_DIRECTORY)
            )
            logger.info("嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            raise RuntimeError(f"嵌入模型初始化失败: {str(e)}")

        if config.USE_CHAPTER_SPLITTER:
            # 使用章节分割器（学习资料专用）
            self.chapter_splitter = ChapterSplitter(
                heading_patterns=config.CHAPTER_HEADING_PATTERNS,
                overlap_lines=config.CHAPTER_OVERLAP_LINES,
                min_content_length=config.CHAPTER_MIN_CONTENT_LENGTH
            )
            self.spliter = None  # 禁用普通文本分割器
            logger.info("使用章节分割器模式")
        else:
            # 使用传统的字符分割器
            self.chapter_splitter = None
            self.spliter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size, #分割行的文本段最大长度
                chunk_overlap=config.chunk_overlap, #连续文本段之间的字符重叠数
                separators=config.separators,   #自然段落划分的符号
                length_function=len,    #使用python自带的len函数做长度统计的依据
            )
            logger.info("使用传统字符分割器模式")

    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化，存入向量化数据库中"""
        if not data or not data.strip():
            logger.info(f"文件解析后内容为空，无需入库: {filename}")
            return '[失败]文件解析后内容为空，无需入库'

        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return '[跳过]内容已经存在知识库中'

        literature_title = identify_literature_title(filename, data)
        save_literature_item(filename, literature_title)
        logger.info(f"文献标题识别完成: {filename} -> {literature_title}")

        if self.chapter_splitter:
            docs_with_metadata = self.chapter_splitter.split_with_metadata(data, filename)
            
            knowledge_chunks = []
            metadatas = []
            global_chunk_index = 0
            for doc in docs_with_metadata:
                chapter_content = doc['content']
                max_len = min(config.CHAPTER_MAX_CONTENT_LENGTH, 8191)
                sub_chunks = _split_text_by_max_length(chapter_content, max_length=max_len)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    chunk_metadata = {
                        'source': doc['metadata']['source'],
                        'title': literature_title,
                        'chapter': doc['metadata']['chapter'],
                        'chapter_number': doc['metadata']['chapter_number'],
                        'total_chapters': doc['metadata']['total_chapters'],
                        'chunk_index': global_chunk_index,
                        'sub_chunk_index': i + 1 if len(sub_chunks) > 1 else None,
                        'total_sub_chunks': len(sub_chunks) if len(sub_chunks) > 1 else None,
                        'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'operator': 'administrator',
                        'keywords': extract_keywords(sub_chunk)
                    }
                    knowledge_chunks.append(sub_chunk)
                    metadatas.append(chunk_metadata)
                    global_chunk_index += 1
            
            logger.info(f"章节分割完成: {filename}，共识别 {len(docs_with_metadata)} 个章节，分割为 {len(knowledge_chunks)} 个 chunks")
        else:
            if len(data) > config.max_split_char_number:
                knowledge_chunks:list[str] = self.spliter.split_text(data)
            else:
                knowledge_chunks = [data]

            metadatas = []
            for idx, chunk in enumerate(knowledge_chunks):
                chunk_metadata = {
                    'source': filename,
                    'title': literature_title,
                    'chapter': None,
                    'chapter_number': None,
                    'total_chapters': None,
                    'chunk_index': idx,
                    'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'operator': 'administrator',
                    'keywords': extract_keywords(chunk)
                }
                metadatas.append(chunk_metadata)

        try:
            self.chroma.add_texts(
                knowledge_chunks,
                metadatas=metadatas

            )
            logger.info(f"向量入库成功: {filename}，分割为 {len(knowledge_chunks)} 个 chunks")

            save_md5(md5_hex, filename)

            return '[成功]内容已经成功载入向量库'
        except Exception as e:
            logger.error(f"向量入库失败: {str(e)}")
            return f'[失败]内容入库失败: {str(e)}'

    def upload_by_file(self, file_path: str, filename: str):
        """
        从文件路径读取文件并上传到知识库
        file_path: 文件的完整路径
        filename: 原始文件名
        """
        try:
            # 获取文件扩展名
            file_ext = filename.split('.')[-1].lower()
            
            # 根据文件类型读取内容
            if file_ext == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            elif file_ext == 'pdf':
                doc = fitz.open(file_path)
                data = ''.join(page.get_text() or '' for page in doc)
                # 检测是否为扫描件/图片PDF
                has_images = any(page.get_images(full=True) for page in doc)
                doc.close()
                if not data.strip():
                    if has_images:
                        return '[失败]PDF内容为空，该文件可能是扫描件或图片PDF，不支持OCR文字识别，请上传包含可提取文字的PDF'
                    return '[失败]PDF内容为空，无法提取文字。可能原因：1)扫描件PDF 2)图片型PDF 3)特殊编码PDF'
            elif file_ext == 'docx':
                doc = Document(file_path)
                lines = []
                for para in doc.paragraphs:
                    if config.DOCX_IGNORE_HEADER_FOOTER and (para.text.strip() == "" or para.style.name in ['Header', 'Footer']):
                        continue
                    lines.append(para.text)
                data = '\n'.join(lines)
            elif file_ext == 'md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
            else:
                logger.error(f"不支持的文件类型: {file_ext}")
                return f'[失败]不支持的文件类型: {file_ext}'
            
            logger.info(f"文件读取成功: {filename}，内容长度: {len(data)} 字符")
            
            # 校验文件内容是否为空
            if not data or not data.strip():
                logger.info(f"文件内容为空，无需入库: {filename}")
                return '[失败]文件内容为空，无法入库'
            
            # 调用 upload_by_str 进行入库
            return self.upload_by_str(data, filename)
            
        except Exception as e:
            logger.error(f"文件读取失败: {str(e)}")
            return f'[失败]文件读取失败: {str(e)}'

    def delete_file(self, filename: str):
        """
        从知识库中删除指定文件的所有向量数据
        """
        try:
            results = self.chroma.get(
                where={"source": filename},
                include=[]
            )
            
            ids_to_delete = results['ids']
            
            if ids_to_delete:
                self.chroma.delete(ids_to_delete)
                logger.info(f"成功删除 {len(ids_to_delete)} 个向量，来源: {filename}")
                
                with md5_lock:
                    if MD5_FILE_PATH.exists():
                        with open(MD5_FILE_PATH, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        new_lines = [line for line in lines if filename not in line]
                        
                        with open(MD5_FILE_PATH, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                
                remove_literature_item(filename)
                
                return f'[成功]已从知识库中删除文件: {filename}'
            else:
                logger.info(f"未找到文件 {filename} 的向量数据")
                return f'[提示]未找到文件 {filename} 的数据'
        except Exception as e:
            logger.error(f"删除文件失败: {str(e)}")
            return f'[失败]删除文件失败: {str(e)}'


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str('周杰伦',filename='testfile')
    print(r)
