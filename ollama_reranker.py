# Ollama 重排序器服务
import logging
from typing import List, Tuple
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaReranker:
    """
    使用 Ollama 本地模型进行文档重排序
    """
    
    def __init__(self, model_name: str = "B-A-M-N/qwen3-reranker-0.6b-fp16:latest"):
        """
        初始化 Ollama 重排序器
        
        Args:
            model_name: Ollama 重排序模型名称
        """
        try:
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model=model_name)
            logger.info(f"Ollama 重排序模型初始化成功: {model_name}")
        except Exception as e:
            logger.error(f"Ollama 重排序模型初始化失败: {str(e)}")
            raise RuntimeError(f"Ollama 重排序模型初始化失败: {str(e)}")
        
        self.model_name = model_name
    
    def compute_similarity(self, query_embedding: List[float], doc_embedding: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            query_embedding: 查询向量
            doc_embedding: 文档向量
        
        Returns:
            余弦相似度分数
        """
        import numpy as np
        
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)
        
        dot_product = np.dot(query_vec, doc_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return float(dot_product / (query_norm * doc_norm))
    
    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            docs: 原始检索结果文档列表
            top_k: 返回的文档数量
        
        Returns:
            重排序后的文档列表
        """
        if not docs:
            logger.info("没有文档需要重排序")
            return docs
        
        try:
            # 获取查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            logger.info(f"查询嵌入向量生成成功，长度: {len(query_embedding)}")
            
            # 获取所有文档的嵌入向量
            doc_contents = [doc.page_content for doc in docs]
            doc_embeddings = self.embeddings.embed_documents(doc_contents)
            logger.info(f"文档嵌入向量生成成功，数量: {len(doc_embeddings)}")
            
            # 计算相似度并排序
            scored_docs = []
            for i, (doc, embedding) in enumerate(zip(docs, doc_embeddings)):
                similarity = self.compute_similarity(query_embedding, embedding)
                scored_docs.append((doc, similarity))
                logger.debug(f"文档 {i} 相似度: {similarity}")
            
            # 按相似度降序排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前 top_k 个文档
            result_docs = [doc for doc, score in scored_docs[:top_k]]
            logger.info(f"重排序完成，返回 {len(result_docs)} 个文档")
            
            return result_docs
        
        except Exception as e:
            logger.error(f"文档重排序失败: {str(e)}")
            # 如果重排序失败，返回原始文档列表
            return docs[:top_k]


# 测试代码
if __name__ == "__main__":
    # 测试重排序器
    reranker = OllamaReranker()
    
    # 创建测试文档
    test_docs = [
        Document(page_content="Python 是一种高级编程语言，非常适合数据科学和机器学习。", metadata={"source": "test1.txt"}),
        Document(page_content="Java 是一种面向对象的编程语言，广泛应用于企业级开发。", metadata={"source": "test2.txt"}),
        Document(page_content="机器学习是人工智能的一个分支，使用算法让计算机从数据中学习。", metadata={"source": "test3.txt"}),
    ]
    
    # 测试查询
    query = "Python 在数据科学中的应用"
    reranked_docs = reranker.rerank(query, test_docs, top_k=2)
    
    print("重排序结果:")
    for i, doc in enumerate(reranked_docs):
        print(f"{i+1}. {doc.page_content}")
        print(f"   来源: {doc.metadata.get('source', '未知')}")
        print()
