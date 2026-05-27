#向量存储服务(模型去向量库中检索)
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from langchain_chroma import Chroma
import config_data as config

# RAG 目录的绝对路径
RAG_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = RAG_DIR / 'chroma_db'

# 导入 Ollama 重排序器
if config.USE_OLLAMA_RERANKER:
    try:
        from ollama_reranker import OllamaReranker
        logger.info("Ollama 重排序器导入成功")
    except Exception as e:
        logger.warning(f"Ollama 重排序器导入失败: {str(e)}")

class VectorStoreService(object):
    def __init__(self,embedding):
        """
        embedding:嵌入模型的传入
        """
        self.embedding = embedding
        try:
            logger.info(f"向量检索器初始化: collection_name={config.collection_name}, persist_directory={PERSIST_DIRECTORY}")
            PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            self.vector_store = Chroma(
                collection_name=config.collection_name,
                embedding_function =self.embedding,
                persist_directory=str(PERSIST_DIRECTORY)
            )
            logger.info("向量库初始化成功")
        except Exception as e:
            logger.error(f"向量库初始化失败: {str(e)}")
            raise RuntimeError(f"向量库初始化失败: {str(e)}")
        
        # 初始化 Ollama 重排序器（如果启用）
        self.reranker = None
        if config.USE_OLLAMA_RERANKER:
            try:
                self.reranker = OllamaReranker(model_name=config.OLLAMA_RERANKER_MODEL)
                logger.info(f"Ollama 重排序器初始化成功: {config.OLLAMA_RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Ollama 重排序器初始化失败，将跳过重排序: {str(e)}")

    def _expand_with_neighbors(self, docs, top_n=2):
        """
        为检索结果中排名靠前的文档补充物理上相邻的 chunk
        top_n: 对前 top_n 个结果进行邻近扩展
        """
        if not docs:
            return docs
        
        expanded = []
        seen_ids = set()
        
        for i, doc in enumerate(docs):
            doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}-{doc.metadata.get('chunk_index', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                expanded.append(doc)
            
            if i >= top_n:
                continue
            
            source = doc.metadata.get('source', '')
            chunk_index = doc.metadata.get('chunk_index', None)
            title = doc.metadata.get('title', '')
            
            if chunk_index is None or not source:
                continue
            
            for offset in [-1, 1]:
                neighbor_index = chunk_index + offset
                if neighbor_index < 0:
                    continue
                
                neighbor_filter = {
                    '$and': [
                        {'source': source},
                        {'chunk_index': neighbor_index}
                    ]
                }
                
                try:
                    neighbor_results = self.vector_store.similarity_search(
                        '', k=1, filter=neighbor_filter
                    )
                    if neighbor_results:
                        neighbor = neighbor_results[0]
                        neighbor_id = f"{neighbor.page_content[:100]}-{neighbor.metadata.get('source', '')}-{neighbor.metadata.get('chunk_index', '')}"
                        if neighbor_id not in seen_ids:
                            seen_ids.add(neighbor_id)
                            neighbor.metadata['is_neighbor'] = True
                            neighbor.metadata['neighbor_of'] = chunk_index
                            expanded.append(neighbor)
                            logger.debug(f"补充邻近 chunk: source={source}, chunk_index={neighbor_index}")
                except Exception as e:
                    logger.warning(f"查询邻近 chunk 失败: source={source}, chunk_index={neighbor_index}, error={str(e)}")
        
        logger.info(f"邻近扩展完成: 原始 {len(docs)} 个 -> 扩展后 {len(expanded)} 个")
        return expanded

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
            semantic_results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"语义检索结果数量: {len(semantic_results)}")
            
            keyword_results = []
            if keywords:
                for keyword in keywords:
                    results = self.vector_store.similarity_search(keyword, k=k)
                    keyword_results.extend(results)
                keyword_results = self._deduplicate_docs(keyword_results)
                logger.info(f"关键词检索结果数量: {len(keyword_results)}")
            
            merged_results = self._merge_results(semantic_results, keyword_results, k)
            logger.info(f"合并后检索结果数量: {len(merged_results)}")
            
            if self.reranker and merged_results:
                merged_results = self.reranker.rerank(query, merged_results, top_k=k)
                logger.info(f"Ollama 重排序完成")
            
            merged_results = self._expand_with_neighbors(merged_results, top_n=config.NEIGHBOR_EXPAND_TOP_N)
            
            return merged_results
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise RuntimeError(f"混合检索失败: {str(e)}")

    def retrieve_by_title(self, query: str, title: str, k: int = None):
        """
        基于文献标题的过滤检索：只在指定文献的向量数据中检索
        """
        if k is None:
            k = config.retrieve_top_k
        
        try:
            filtered_results = self.vector_store.similarity_search(
                query,
                k=k,
                filter={"title": title}
            )
            logger.info(f"按文献标题过滤检索: title={title}, 结果数量: {len(filtered_results)}")
            return filtered_results
        except Exception as e:
            logger.error(f"按文献标题过滤检索失败: {str(e)}")
            return []

    def hybrid_retrieve_with_reference(self, query: str, keywords: list, reference_title: str = None, k: int = None):
        """
        分源混合检索：当用户提到参考/依据时，一半从指定文献检索，一半从全局检索
        reference_title: 用户指定的文献标题
        """
        if k is None:
            k = config.retrieve_top_k
        
        try:
            half_k = max(k // 2, 1)
            
            reference_results = []
            if reference_title:
                reference_results = self.retrieve_by_title(query, reference_title, k=half_k)
                logger.info(f"指定文献检索结果数量: {len(reference_results)}")
            
            global_results = self.hybrid_retrieve(query, keywords, k=max(k - len(reference_results), half_k))
            logger.info(f"全局检索结果数量: {len(global_results)}")
            
            merged = []
            seen = set()
            for doc in reference_results:
                doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    doc.metadata['match_type'] = 'reference'
                    merged.append(doc)
            
            for doc in global_results:
                doc_id = f"{doc.page_content[:100]}-{doc.metadata.get('source', '')}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    doc.metadata['match_type'] = 'global'
                    merged.append(doc)
            
            logger.info(f"分源检索合并结果数量: {len(merged)} (参考: {len(reference_results)}, 全局: {len(global_results)})")
            
            merged = self._expand_with_neighbors(merged, top_n=config.NEIGHBOR_EXPAND_TOP_N)
            
            return merged
        except Exception as e:
            logger.error(f"分源检索失败: {str(e)}")
            return self.hybrid_retrieve(query, keywords, k=k)

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
