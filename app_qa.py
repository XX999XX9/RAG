#项目主程序(streamlit),启动对话WEB页面
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import streamlit as st
from rag import RagService
import config_data as config

#标题
st.title('求职信息智能问答模型')
st.divider() #分隔符

#用session_state保存历史对话和RagService对象
if 'message' not in st.session_state:
    st.session_state['message'] = [{'role':'assistant','content':'你好，有什么可以帮助你'}]
if 'rag' not in st.session_state:
    try:
        st.session_state['rag'] = RagService()
        logger.info("RagService初始化成功")
    except Exception as e:
        logger.error(f"RagService初始化失败: {str(e)}")
        st.error("问答服务初始化失败，请检查配置")
#把会话历史写入页面聊天框
for message in st.session_state['message']:
    st.chat_message(message['role']).write(message['content'])
#在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    #在页面输入用户的提问
    st.chat_message('user').write(prompt)
    st.session_state['message'].append({'role':'user','content':prompt})
    
    logger.info(f"用户输入接收: {prompt}")
    
    ai_res_list = []
    with st.spinner("AI思考中..."):
        try:
            logger.info("AI回答生成开始")
            res_stream = st.session_state['rag'].chain.stream({'input': prompt}, config.session_config)

            def capture(generator,cache_list):
                for chunk in generator:
                    cache_list.append(chunk)
                    yield chunk

            st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
            st.session_state['message'].append({'role': 'assistant', 'content':''.join(ai_res_list)})
            
            logger.info(f"AI回答生成结束: {''.join(ai_res_list)[:100]}...")  # 只记录前100个字符
        except Exception as e:
            logger.error(f"AI服务调用失败: {str(e)}")
            error_message = "AI 服务调用失败，请检查 API 密钥或网络"
            st.chat_message("assistant").write(error_message)
            st.session_state['message'].append({'role': 'assistant', 'content': error_message})
