#长期会话记忆存储服务
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from typing import Sequence
from dotenv import load_dotenv
import os
import json
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import  BaseChatMessageHistory


load_dotenv('.env',override=True)
api_key = os.getenv('BAILIAN_API_KEY')
base_url = os.getenv('BAILIAN_BASE_URL')

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id#会话id
        self.storage_path = storage_path#不同会话id的存储文件所在的文件夹路径
        #完整的文件路径
        self.file_path = os.path.join(self.storage_path,self.session_id)

        #确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:#三大功能之一，把消息添加到会话历史文件中
        #message是Sequance序列：类似list,tuple
        all_messages = list(self.messages)#已有的消息列表
        all_messages.extend(messages)
        '''
        将数据写入本地文件中
        类对象写入文件->一堆二进制
        为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        官方message_to_dict:将单个消息对象（BaseMessage类实例）->字典
        new_message = []
        for message in all_messages:
            d = message_to_dict(message)
            new_messages.append(d)
        '''

        new_messages = [message_to_dict(message) for message in all_messages]
        #将数据写入文件
        try:
            with open(self.file_path,'w',encoding='utf-8') as f:
                json.dump(new_messages,f)
            logger.info(f"会话历史写入成功: session_id={self.session_id}, 消息数量={len(new_messages)}")
        except Exception as e:
            logger.error(f"会话历史写入失败: {str(e)}")


    @property #装饰器将messages方法编程成员属性用
    def messages(self) -> list[BaseMessage]:#三大功能之二，读取历史会话文件
        #当前文件内：list[字典]
        try:
            with open(self.file_path,'r',encoding='utf-8') as f:
                messages_data = json.load(f)
                messages = messages_from_dict(messages_data)
                logger.info(f"会话历史读取成功: session_id={self.session_id}, 消息数量={len(messages)}")
                return messages
        except FileNotFoundError:
            logger.info(f"会话历史文件不存在: {self.file_path}")
            return []
        except Exception as e:
            logger.error(f"会话历史读取失败: {str(e)}")
            return []

    def clear(self) -> None:#三大功能之三，清空历史会话文件
        try:
            with open(self.file_path,'w',encoding='utf-8') as f:
                json.dump([],f)#向文件写入一个空列表实现清空功能
            logger.info(f"会话历史清空成功: session_id={self.session_id}")
        except Exception as e:
            logger.error(f"会话历史清空失败: {str(e)}")


#通过会话id获取InMemoryChatMessageHistory
def get_history(session_id):
    logger.info(f"获取会话历史: session_id={session_id}")
    return FileChatMessageHistory(session_id,'./chat_history')
