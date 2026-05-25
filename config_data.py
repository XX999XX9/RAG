#配置文件

# 服务端口配置
SERVER_PORT = 8989  # 后端服务端口
SERVER_HOST = "0.0.0.0"  # 服务绑定地址

# 文件解析配置
SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'md']  # 支持的文件类型
MAX_FILE_SIZE_MB = 200  # 单文件大小限制（MB）
PDF_PARSE_START_PAGE = 0  # PDF起始解析页码
PDF_PARSE_END_PAGE = None  # PDF结束解析页码（None表示全部）
DOCX_IGNORE_HEADER_FOOTER = True  # Word解析是否忽略页眉页脚

md5_path = './md5.txt'

#Chroma
collection_name = 'RAG'#向量数据的集合名，类似关系型数据的的表名
persist_directory = './chroma_db'#向量数据的本地存储目录

#spliter
chunk_size = 1000
chunk_overlap = 100
separators = [
    # 核心段落分隔
    "\n\n", "\n", "\r\n\r\n", "\r\n",  # 换行（兼容Windows/Linux换行符）
    # 中文句末符号
    "。", "！", "？", "…", "；", "：",
    # 英文句末符号（带空格避免误分割缩写）
    ". ", "! ", "? ", "; ", ": ", "... ",
    # 特殊场景分隔符（如文档中的列表/标题）
    "、", "，", ", ", "—", "——", "|", "||"
]#ai生成

max_split_char_number = 1000#文本分割的阈值

# 章节分割配置（学习资料专用）
USE_CHAPTER_SPLITTER = True  # 是否使用章节分割模式
CHAPTER_HEADING_PATTERNS = [
    r'^第[零一二三四五六七八九十百千万]+[章节编篇部卷].*$',  # 中文数字章节标题（如 第一章、第二篇）
    r'^\d+\.\d+\.\d+[\.\uff0e、]?[^\n]+$',  # 三级数字编号（如 1.1.1 标题）- 更具体的先匹配
    r'^\d+\.\d+[\.\uff0e、]?[^\n]+$',  # 二级数字编号（如 1.1 标题、1.1. 标题）
    r'^\d+[\.\uff0e、][^\n]+$',  # 一级数字编号（如 1. 标题、1、标题）
    r'^[一二三四五六七八九十]+[\uff0e、\.][^\n]+$',  # 中文数字编号标题（如 一、标题）
    r'^[（\(][一二三四五六七八九十]+[）\)][^\n]+$',  # 带括号中文数字（如 （一）标题）
    r'^[（\(][一二三四五六七八九十]+[）\)][一二三四五六七八九十]+[\uff0e、\.][^\n]+$',  # 带括号中文数字二级（如 （一）一、标题）
    r'^[A-Za-z][\uff0e、\.]?[^\n]+$',  # 字母编号标题（如 A 标题、A. 标题）
    r'^【.+】.*$',  # 方括号标题
    r'^《.+》.*$',  # 书名号标题
]
CHAPTER_OVERLAP_LINES = 2  # 章节之间的重叠行数
CHAPTER_MIN_CONTENT_LENGTH = 50  # 章节最小内容长度（字符）
CHAPTER_MAX_CONTENT_LENGTH = 1000  # 章节最大内容长度（超过则按文本块切分）

# 关键词检索配置
KEYWORD_STOP_WORDS = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']  # 中文停用词
KEYWORD_MIN_LENGTH = 2#关键词最小长度
KEYWORD_WEIGHT = 0.3  # 关键词检索权重
SEMANTIC_WEIGHT = 0.7  # 语义检索权重
RETRIEVE_MERGE_STRATEGY = 'weighted'  # 结果合并策略：weighted（加权）或 hybrid（混合）

# 知识库管理配置
DOC_UNIQUE_ID_PREFIX = 'doc_'  # 文档唯一ID前缀（如doc_xxx_paragraph_yyy）

# 文献识别配置
LITERATURE_LIST_PATH = './literature_list.json'  # 文献列表存储路径
REFERENCE_KEYWORDS = ['参考', '依据', '引用', '根据', '出自', '来源', '按照', '参见']  # 触发分源检索的关键词

#向量库中返回的检索结果数量
retrieve_top_k = 10

#嵌入模型名称
embedding_model_name = 'text-embedding-v4'
chat_model_name = 'deepseek-v4-pro'

# Ollama 模型配置（本地部署的模型）
OLLAMA_HOST = 'http://localhost:11434'  # Ollama 服务地址
OLLAMA_RERANKER_MODEL = 'B-A-M-N/qwen3-reranker-0.6b-fp16:latest'  # 用于重排序的模型

# 独立的模型使用开关
USE_OLLAMA_RERANKER = False  # 是否使用 Ollama 重排序（检索和重排）
# 知识入库始终使用云端嵌入模型（DashScope）
# 聊天模型始终使用云端模型（ChatTongyi）

#session id配置
session_config={
        'configurable':{
            'session_id':'user_001'
    }
}