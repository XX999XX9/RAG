#rag核心服务
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import jieba
from dotenv import load_dotenv
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import  ChatTongyi
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from langchain_core.output_parsers import StrOutputParser
import config_data as config

load_dotenv('.env',override=True)
api_key = os.getenv('BAILIAN_API_KEY')
base_url = os.getenv('BAILIAN_BASE_URL')

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


class RagService(object):
    def __init__(self):
        try:
            self.vector_service = VectorStoreService(
                embedding=DashScopeEmbeddings(model=config.embedding_model_name)
            )
            logger.info("嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            raise RuntimeError(f"嵌入模型初始化失败: {str(e)}")

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system', """
                你是求职信息智能问答助手，仅基于提供的参考资料回答关于求职信息的问题：
                1. 参考资料：{context}
                2. 回答要求：简洁专业，仅围绕求职相关内容，不编造信息；
                3. 若参考资料无相关信息，回复"未在数据库中查询到相关内容，以下是根据互联网公开内容获取到的信息"，并输出后续内容。
                """),
                ('system','并且我提供用户的对话历史记录，对话历史记录如下：'),
                MessagesPlaceholder('history'),
                ('user','请回答用户提问：{input}')
            ]
        )

        try:
            self.chat_model = ChatTongyi(model=config.chat_model_name,api_key=api_key)
            logger.info("ChatTongyi模型初始化成功")
        except Exception as e:
            logger.error(f"ChatTongyi模型初始化失败: {str(e)}")
            raise RuntimeError(f"ChatTongyi模型初始化失败: {str(e)}")

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""

        def format_document(docs):
            if not docs:
                logger.info("检索到的参考资料数量: 0")
                return '无相关参考资料'
            formatted_str = ""
            for i, doc in enumerate(docs):
                # 稍微调整格式，强调内容
                source = doc.metadata.get('source', '未知')#如果有source就返回来源，没有就返回未知
                # 尝试获取匹配类型（如果有的话）
                match_type = doc.metadata.get('match_type', 'semantic')
                formatted_str += f"[参考资料{i + 1}]({match_type}):\n内容: {doc.page_content}\n来源: {source}\n\n"
            logger.info(f"检索到的参考资料数量: {len(docs)}")
            return formatted_str

        #可加入chain中self.prompt_template后面打印提示词进行调试
        def print_prompt(prompt):
            print('=' * 60)
            print(prompt.to_string())
            print('=' * 60)
            return prompt

        # 从输入的字典{'input':...,'history':...}中获取input的键值
        def format_for_retriever(value: dict):
            query = value['input']
            logger.info(f"检索开始: {query}")
            # 提取关键词
            keywords = extract_keywords(query)
            logger.info(f"提取的关键词: {keywords}")
            return {'query': query, 'keywords': keywords}

        # 执行混合检索
        def hybrid_retrieve(value):
            query = value['query']
            keywords = value['keywords']
            results = self.vector_service.hybrid_retrieve(query, keywords)
            return results

        # 我们前面get_input的操作链上内容从首个字典传出时格式为{'input':{'input':...,''history':[]},'context':...}
        # 但是实际上需要的是{'input':...,'history':[],'context':...},所以进行转换再传递到后面
        def format_for_prompt_template(value: dict):
            new_value = {'input': value['input']['input'], 'history': value['input']['history'],
                         'context': value['context']}
            logger.info(f"检索结束: {value['input']['input']}")
            return new_value

        def api_call_start(value):
            logger.info(f"API调用开始")
            return value

        def api_call_end(value):
            logger.info(f"API调用结束")
            return value

        chain = ({'input': RunnablePassthrough(), 'context': RunnableLambda(format_for_retriever) |
                RunnableLambda(hybrid_retrieve) | format_document} | RunnableLambda(format_for_prompt_template) |
                RunnableLambda(api_call_start) |
                self.prompt_template | self.chat_model | 
                RunnableLambda(api_call_end) |
                StrOutputParser()
        )
        '''
        示例：
        用户输入：XX岗位的求职要求是什么
        步骤 1：调用链时的输入
        input_data = {"input": "XX岗位的求职要求是什么", "history": []}  # 用户提问+历史记录
        session_config = {"configurable": {"session_id": "user_001"}}
        步骤 2：RunnablePassthrough 透传 + 检索子链执行
        # 链的第一步：生成 {'input': ..., 'context': ...} 字典
        {
            'input': RunnablePassthrough(),  # 透传原始input_data → {'input': "XX岗位的求职要求是什么", 'history': []}
            'context': RunnableLambda(format_for_retriever) | RunnableLambda(hybrid_retrieve) | format_document
        }
        format_for_retriever 接收 input_data → 只提取 input_data['input']（“XX岗位的求职要求是什么”），提取关键词 → 输出 {'query': "XX岗位的求职要求是什么", 'keywords': ["XX职位","求职要求"]}
        然后执行混合检索，返回检索到的参考资料，并进行格式化输出
        步骤 3：第一步输出结果
        {
        'input': {'input': "XX岗位的求职要求是什么", 'history': []},  # 透传的原始输入
        'context': "[参考资料1](semantic):\n内容: xxx\n来源: xxx\n\n"  # 检索结果格式化后
        }
        步骤 4：format_for_prompt_template 修正格式
        {
        'input': "XX岗位的求职要求是什么",
        'history': [],
        'context': "[参考资料1](semantic):\n内容: xxx\n来源: xxx\n\n"
        }
        步骤5：传入提示词模板->调用模型->返回结果->解析为字符串
        提示词模板部分MessagesPlaceholder('history')会被RunnableWithMessageHistory()自动填充历史会话记录
        '''

        # 带入历史记录增强的chain,但是输入需要从str形式变成字典形式
        conversation_chain = RunnableWithMessageHistory(#导入历史对话记录以及自动填充历史会话记录
            chain,
            get_history,
            input_messages_key='input',
            history_messages_key='history'
        )

        return conversation_chain

# if __name__ == '__main__':
#     res = RagService().chain.stream({'input':'langchain可以用于干什么'},session_config)
# for chunk in res:
#     print(chunk,end='',flush=True)
