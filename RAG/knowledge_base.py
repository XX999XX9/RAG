#离线流程：知识库更新服务
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import PyPDF2
from docx import Document
import jieba

#加载模型参数
load_dotenv(dotenv_path='.env',override=True)
api_key = os.getenv('BAILIAN_API_KEY')
base_url = os.getenv('BAILIAN_BASE_URL')

#文件解析函数
def parse_file(file_obj, file_type: str) -> str:
    """
    解析不同类型文件为纯文本
    file_obj: streamlit上传的文件对象（BytesIO）
    file_type: 文件类型（txt/pdf/docx/md）
    """
    try:
        if file_type == 'txt':
            try:
                text = file_obj.getvalue().decode('utf-8')
            except Exception as e:
                logger.error(f"TXT文件解析失败: {str(e)}")
                raise RuntimeError(f"TXT文件解析失败: {str(e)}")
        elif file_type == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(file_obj)
                text = ""
                start = config.PDF_PARSE_START_PAGE
                end = config.PDF_PARSE_END_PAGE or len(pdf_reader.pages)
                for page_num in range(start, end):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            except Exception as e:
                logger.error(f"PDF文件解析失败: {str(e)}")
                raise RuntimeError(f"PDF文件解析失败: {str(e)}")
        elif file_type == 'docx':
            try:
                doc = Document(file_obj)
                text = ""
                for para in doc.paragraphs:
                    if config.DOCX_IGNORE_HEADER_FOOTER and (para.text.strip() == "" or para.style.name in ['Header', 'Footer']):
                        continue
                    text += para.text + "\n"
            except Exception as e:
                logger.error(f"DOCX文件解析失败: {str(e)}")
                raise RuntimeError(f"DOCX文件解析失败: {str(e)}")
        elif file_type == 'md':
            try:
                text = file_obj.getvalue().decode('utf-8')
            except Exception as e:
                logger.error(f"MD文件解析失败: {str(e)}")
                raise RuntimeError(f"MD文件解析失败: {str(e)}")
        else:
            raise ValueError(f"不支持的文件类型：{file_type}")
        
        logger.info(f"文件解析成功，类型: {file_type}，内容长度: {len(text)} 字符")
        return text
    except Exception as e:
        logger.error(f"文件解析失败: {str(e)}")
        raise


def check_md5(md5_str:str):
    """
    检查传入的md5字符串是否已经被处理过了
        return False(md5未处理过)   True（已经处理过，已有记录）
    """
    try:
        if not os.path.exists(config.md5_path):
            #if 进入表示文件不存在，说明没有处理过这个md5文件
            open(config.md5_path,'w',encoding='utf-8').close()
            logger.info(f"MD5文件不存在，创建新文件: {config.md5_path}")
            return False
        else:
            for line in open(config.md5_path,'r',encoding='utf-8').readlines():
                line = line.strip()  #处理字符串前后的空格和回车
                if line == md5_str:
                    logger.info(f"MD5校验通过：{md5_str} 已存在")
                    return True #已经处理过

            logger.info(f"MD5校验通过：{md5_str} 不存在")
            return False
    except Exception as e:
        logger.error(f"MD5校验失败: {str(e)}")
        return False

def save_md5(md5_str:str):
    """
    将传入的md5字符串，记录到文件内保存
    """
    try:
        with open(config.md5_path,'a',encoding='utf-8') as f:
            f.write(md5_str+'\n')
        logger.info(f"MD5保存成功：{md5_str}")
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


def extract_keywords(text: str) -> list:
    """
    从文本中提取关键词
    """
    # 使用结巴分词
    words = jieba.cut(text)
    
    # 过滤停用词和短词
    keywords = []
    for word in words:
        if word not in config.KEYWORD_STOP_WORDS and len(word) >= config.KEYWORD_MIN_LENGTH:
            keywords.append(word)
    
    # 去重并返回前10个关键词
    return list(set(keywords))[:10]


class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        # 向量存储实例Chroma对象
        try:
            self.chroma = Chroma(
                collection_name=config.collection_name, #数据库的表名
                embedding_function = DashScopeEmbeddings(model=config.embedding_model_name),#不需要传入api_key参数，可以自动读取
                persist_directory=config.persist_directory  #数据库本地存储文件夹
            )
            logger.info("嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            raise RuntimeError(f"嵌入模型初始化失败: {str(e)}")

        #文本分割器的对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, #分割行的文本段最大长度
            chunk_overlap=config.chunk_overlap, #连续文本段之间的字符重叠数
            separators=config.separators,   #自然段落划分的符号
            length_function=len,    #使用python自带的len函数做长度统计的依据
        )

    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化，存入向量化数据库中"""
        # 校验解析后的文本是否为空：虽然理论上不会为空，但是为了保险起见，还是校验一下
        #再次校验原因：防御性编程（该模块可能被其他地方调用，增加鲁棒性）；职责分离，作为知识库服务层应该对传入数据进行校验；提供监控日志等
        if not data or not data.strip():
            logger.info(f"文件解析后内容为空，无需入库: {filename}")
            return '文件解析后内容为空，无需入库'

        #先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return '[跳过]内容已经存在知识库中'

        if len(data) > config.max_split_char_number:
            knowledge_chunks:list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 为每个文本块提取关键词并生成metadata
        metadatas = []
        for chunk in knowledge_chunks:
            chunk_metadata = {
                'source': filename,
                'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'operator': 'administrator',
                'keywords': extract_keywords(chunk)
            }
            metadatas.append(chunk_metadata)

        #内容加载到向量库中
        try:
            self.chroma.add_texts(
                #iterable -> list\tuple
                knowledge_chunks,
                metadatas=metadatas

            )
            logger.info(f"向量入库成功: {filename}，分割为 {len(knowledge_chunks)} 个 chunks")

            #跑完文件上传至向量数据库的流程，记录该文件已经保存过了
            save_md5(md5_hex)

            return '[成功]内容已经成功载入向量库'
        except Exception as e:
            logger.error(f"向量入库失败: {str(e)}")
            return f'[失败]内容入库失败: {str(e)}'

if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str('周杰伦',filename='testfile')
    print(r)
