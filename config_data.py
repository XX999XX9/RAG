#配置文件

# 文件解析配置
SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'md']  # 支持的文件类型
MAX_FILE_SIZE_MB = 10  # 单文件大小限制（MB）
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

# 关键词检索配置
KEYWORD_STOP_WORDS = ['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']  # 中文停用词
KEYWORD_MIN_LENGTH = 2#关键词最小长度
KEYWORD_WEIGHT = 0.3  # 关键词检索权重
SEMANTIC_WEIGHT = 0.7  # 语义检索权重
RETRIEVE_MERGE_STRATEGY = 'weighted'  # 结果合并策略：weighted（加权）或 hybrid（混合）

# 知识库管理配置
DOC_UNIQUE_ID_PREFIX = 'doc_'  # 文档唯一ID前缀（如doc_xxx_paragraph_yyy）

#向量库中返回的检索结果数量
retrieve_top_k = 2

#嵌入模型名称
embedding_model_name = 'text-embedding-v4'
chat_model_name = 'qwen3-max'

#session id配置
session_config={
        'configurable':{
            'session_id':'user_001'
    }
}