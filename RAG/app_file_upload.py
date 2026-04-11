#离线流程：知识库更新主程序(streamlit)
'''
基于Streamlit完成WEB网页上传服务
Streamlit:当web页面元素发生变化，则代码重新执行一遍
'''
import time
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import streamlit as st
from knowledge_base import KnowledgeBaseService, parse_file
import config_data as config

#添加网页标题
st.title('知识库更新服务')

#文件上传服务：file_uploader
uploader_file = st.file_uploader(
    '请上传文件（支持 txt/pdf/docx/md 格式）',
    type=config.SUPPORTED_FILE_TYPES,
    accept_multiple_files=False  #是否仅支持一个文件上传
)

#由于streamlit的页面每次刷新或者元素发生变化，所有代码都会重新执行一次，但是保存在st.session_state()这个字典里的不会，可以用来记录信息
if 'service' not in st.session_state:
    st.session_state['service'] = KnowledgeBaseService()


def get_file_extension(filename: str) -> str:
    """获取文件后缀名（不含点）"""
    return os.path.splitext(filename)[1][1:].lower()


def validate_file(file_obj) -> tuple[bool, str]:
    """
    校验文件是否合法
    返回: (是否通过, 错误信息)
    """
    # 校验文件后缀是否在支持列表内
    file_ext = get_file_extension(file_obj.name)
    if file_ext not in config.SUPPORTED_FILE_TYPES:
        return False, f"不支持的文件类型：{file_ext}，请上传 {', '.join(config.SUPPORTED_FILE_TYPES)} 格式的文件"

    # 校验文件大小（限制单文件大小）
    max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_obj.size > max_size_bytes:
        return False, f"文件大小超过 {config.MAX_FILE_SIZE_MB}MB"

    return True, ""


if uploader_file is not None:
    #提取文件的信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024 #KB

    st.subheader(f'文件名：{file_name}')#二级标题
    st.write(f'文件格式{file_type}|文件大小{file_size:.2f} KB')#写文字

    logger.info(f"文件上传开始: {file_name}, 大小: {file_size:.2f} KB")

    # 执行文件校验
    is_valid, error_msg = validate_file(uploader_file)
    if not is_valid:
        logger.warning(f"文件校验失败: {error_msg}")
        st.error(error_msg)
    else:
        logger.info(f"文件校验通过: {file_name}")
        # 获取文件后缀并调用 parse_file 解析文件
        file_ext = get_file_extension(file_name)

        try:
            # 调用 parse_file 函数解析不同类型文件
            text = parse_file(uploader_file, file_ext)

            # 校验文件内容是否为空
            if not text or not text.strip():
                logger.warning(f"文件内容为空: {file_name}")
                st.error("文件内容为空")
            else:
                logger.info(f"文件解析成功: {file_name}, 内容长度: {len(text)} 字符")
                with st.spinner('载入知识库中...'):  #在spinner内的代码执行过程中，会有一个转圈动画
                    time.sleep(1)
                    result = st.session_state['service'].upload_by_str(text, file_name)
                    logger.info(f"文件入库结果: {result}")
                    st.write(result)
        except Exception as e:
            logger.error(f"文件解析失败: {str(e)}")
            st.error(f"文件解析失败：{str(e)}")
    
    logger.info(f"文件上传流程结束: {file_name}")
    logger.info(f"文件上传流程结束: {file_name}")
