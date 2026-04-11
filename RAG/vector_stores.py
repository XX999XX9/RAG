#向量存储服务(模型去向量库中检索)
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        """
        embedding:嵌入模型的传入
        """
        self.embedding = embedding
        try:
            logger.info(f"向量检索器初始化: collection_name={config.collection_name}, persist_directory={config.persist_directory}")
            self.vector_store = Chroma(
                collection_name=config.collection_name,
                embedding_function =self.embedding,
                persist_directory=config.persist_directory
            )
            logger.info("向量库初始化成功")
        except Exception as e:
            logger.error(f"向量库初始化失败: {str(e)}")
            raise RuntimeError(f"向量库初始化失败: {str(e)}")

    def get_retriever(self):
        """返回纯向量检索器，项目中没有被使用，作为后续开发备用"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={'k':config.retrieve_top_k})
            logger.info(f"获取向量检索器: top_k={config.retrieve_top_k}")
            return retriever
        except Exception as e:
            logger.error(f"获取向量检索器失败: {str(e)}")
            raise RuntimeError(f"获取向量检索器失败: {str(e)}")

    def hybrid_retrieve(self, query: str, keywords: list, k: int = None):
        """
        混合检索：结合关键词检索和语义检索
        """
        if k is None:
            k = config.retrieve_top_k
        
        try:
            # 语义检索
            semantic_results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"语义检索结果数量: {len(semantic_results)}")
            
            # 关键词检索
            keyword_results = []
            if keywords:
                # 构建关键词查询
                for keyword in keywords:
                    # 搜索包含关键词的文档
                    results = self.vector_store.similarity_search(keyword, k=k)
                    keyword_results.extend(results)#将检索结果迭代（逐个）添加到keyword_results
                # 去重
                keyword_results = self._deduplicate_docs(keyword_results)
                logger.info(f"关键词检索结果数量: {len(keyword_results)}")
            
            # 合并结果
            merged_results = self._merge_results(semantic_results, keyword_results, k)
            logger.info(f"合并后检索结果数量: {len(merged_results)}")
            
            return merged_results
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise RuntimeError(f"混合检索失败: {str(e)}")

    def _deduplicate_docs(self, docs):
        """
        去重文档
        """
        seen = set()
        unique_docs = []
        for doc in docs:
            # 用“文本前100字符+来源文件”做唯一标识，避免重复
            doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def _merge_results(self, semantic_results, keyword_results, k):
        """
        合并检索结果
        """
        # 为结果添加分数
        results_with_scores = {}
        
        # 处理语义检索结果
        for i, doc in enumerate(semantic_results):
            doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}"
            # 语义检索分数：位置越靠前分数越高
            score = (k - i) / k * config.SEMANTIC_WEIGHT
            if doc_id in results_with_scores:#这里其实没有必要去判断，之所以这样写是为了和下面处理关键词检索结果的代码保持一致，降低代码复杂度
                results_with_scores[doc_id]['score'] += score
            else:
                results_with_scores[doc_id] = {
                    'doc': doc,
                    'score': score,
                    'type': 'semantic'
                }
        
        # 处理关键词检索结果
        for i, doc in enumerate(keyword_results):
            doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}"
            # 关键词检索分数：位置越靠前分数越高
            score = (k - i) / k * config.KEYWORD_WEIGHT
            if doc_id in results_with_scores:#这里需要进行if判断因为doc_id可能在语义检索结果中有了得分，所以进行判断并累加得分以及更新type
                results_with_scores[doc_id]['score'] += score
                results_with_scores[doc_id]['type'] = 'hybrid'
            else:
                results_with_scores[doc_id] = {
                    'doc': doc,
                    'score': score,
                    'type': 'keyword'
                }
        
        # 按分数排序并返回前k个结果
        sorted_results = sorted(results_with_scores.values(), key=lambda x: x['score'], reverse=True)[:k]
        return [item['doc'] for item in sorted_results]
