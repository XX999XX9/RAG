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
from langchain_core.documents import Document
import config_data as config

# RAG 目录的绝对路径
RAG_DIR = Path(__file__).parent.resolve()
PERSIST_DIRECTORY = RAG_DIR / 'chroma_db'

# 存活过滤条件：软删除（回收站）中的 chunk 在检索全流程中不可见
# 检索前会通过 _ensure_alive_tag() 给所有 chunk 补齐 deleted 字段，因此 $ne 判定安全
ALIVE_FILTER = {'deleted': {'$ne': 'true'}}


def merge_alive_filter(base_filter: dict = None) -> dict:
    """将存活过滤条件与已有 filter 合并（用于所有检索路径，排除回收站数据）"""
    if not base_filter:
        return dict(ALIVE_FILTER)
    return {'$and': [base_filter, dict(ALIVE_FILTER)]}

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

        # 存量数据迁移：给缺失 deleted 字段的 chunk 补上（幂等，保证存活过滤对全部数据生效）
        self._ensure_alive_tag()

        # 初始化 Ollama 重排序器（如果启用）
        self.reranker = None
        if config.USE_OLLAMA_RERANKER:
            try:
                self.reranker = OllamaReranker(
                    model_name=config.OLLAMA_RERANKER_MODEL,
                    base_url=config.OLLAMA_HOST,
                )
                logger.info(f"Ollama 重排序器初始化成功: {config.OLLAMA_RERANKER_MODEL}")
            except Exception as e:
                logger.warning(f"Ollama 重排序器初始化失败，将跳过重排序: {str(e)}")

    def _ensure_alive_tag(self):
        """
        给所有缺失 deleted 元数据字段的 chunk 补上 deleted=''（存活）。
        软删除依赖 $ne 过滤，缺失字段的文档行为在部分 Chroma 版本中不确定，补齐后语义确定。
        """
        try:
            collection = self.vector_store._collection
            batch = 1000
            offset = 0
            patched_ids, patched_metas = [], []
            while True:
                res = collection.get(include=['metadatas'], limit=batch, offset=offset)
                ids = res.get('ids') or []
                metas = res.get('metadatas') or []
                if not ids:
                    break
                for cid, meta in zip(ids, metas):
                    meta = meta or {}
                    if 'deleted' not in meta:
                        patched_ids.append(cid)
                        patched_metas.append({**meta, 'deleted': ''})
                if len(ids) < batch:
                    break
                offset += batch
            if patched_ids:
                collection.update(ids=patched_ids, metadatas=patched_metas)
                logger.info(f"存活标签迁移完成: 为 {len(patched_ids)} 个存量 chunk 补充 deleted 字段")
        except Exception as e:
            logger.warning(f"存活标签迁移失败（不影响启动）: {str(e)}")

    def fetch_chunk_by_index(self, source: str, chunk_index: int):
        """
        根据 source 和 chunk_index 精确获取某个 chunk
        用于 LLM 驱动的邻近 chunk 扩展（按元数据直查，无需 embedding 调用）
        """
        try:
            results = self.vector_store._collection.get(
                where=merge_alive_filter({'$and': [
                    {'source': {'$eq': source}},
                    {'chunk_index': {'$eq': chunk_index}},
                ]}),
                include=['documents', 'metadatas'],
            )
            docs = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(results.get('documents') or [], results.get('metadatas') or [])
            ]
            if docs:
                logger.info(f"获取邻近 chunk 成功: source={source}, chunk_index={chunk_index}")
            else:
                logger.info(f"邻近 chunk 不存在: source={source}, chunk_index={chunk_index}")
            return docs
        except Exception as e:
            logger.warning(f"获取邻近 chunk 失败: source={source}, chunk_index={chunk_index}, error={str(e)}")
            return []

    def get_retriever(self):
        """返回纯向量检索器，项目中没有被使用，作为后续开发备用"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={'k':config.retrieve_top_k})
            logger.info(f"获取向量检索器: top_k={config.retrieve_top_k}")
            return retriever
        except Exception as e:
            logger.error(f"获取向量检索器失败: {str(e)}")
            raise RuntimeError(f"获取向量检索器失败: {str(e)}")

    def list_documents(self, include_deleted: bool = False) -> list:
        """
        列出知识库中所有文档（按 source 聚合，含存活/回收站分块数量），供管理员页面浏览
        include_deleted=False 时仅返回含存活分块的文档（正常视图）
        include_deleted=True 时返回全部文档（回收站视图，含 deleted_chunks 统计）
        """
        try:
            collection = self.vector_store._collection
            sources = {}
            batch = 1000
            offset = 0
            # 分页扫描全部元数据，避免一次性加载过大
            while True:
                res = collection.get(include=['metadatas'], limit=batch, offset=offset)
                metas = res.get('metadatas') or []
                if not metas:
                    break
                for meta in metas:
                    meta = meta or {}
                    src = meta.get('source', '未知')
                    info = sources.setdefault(src, {
                        'source': src,
                        'title': meta.get('title', ''),
                        'chunks': 0,
                        'deleted_chunks': 0,
                    })
                    if meta.get('deleted') == 'true':
                        info['deleted_chunks'] += 1
                    else:
                        info['chunks'] += 1
                    if not info['title'] and meta.get('title'):
                        info['title'] = meta.get('title')
                if len(metas) < batch:
                    break
                offset += batch
            if not include_deleted:
                # 正常视图：只保留有存活分块的文档
                result = [v for v in sources.values() if v['chunks'] > 0]
            else:
                result = list(sources.values())
            result.sort(key=lambda x: x['source'])
            logger.info(f"知识库文档列表: {len(result)} 个来源（include_deleted={include_deleted}）")
            return result
        except Exception as e:
            logger.error(f"获取知识库文档列表失败: {str(e)}")
            raise RuntimeError(f"获取知识库文档列表失败: {str(e)}")

    def get_document_chunks(self, source: str, page: int = 1, page_size: int = 20,
                            include_deleted: bool = False) -> dict:
        """
        分页浏览指定文档的所有 chunk（内容 + 元数据），供管理员页面浏览
        include_deleted=False 时只返回存活分块；True 时返回全部（含回收站分块）
        """
        try:
            collection = self.vector_store._collection
            base_where = {'source': source}
            if not include_deleted:
                where = merge_alive_filter(base_where)
                try:
                    total = collection.count(where=where)
                except TypeError:
                    total = len(collection.get(where=where, include=[])['ids'])
            else:
                where = base_where
                try:
                    total = collection.count(where=where)
                except TypeError:
                    total = len(collection.get(where=where, include=[])['ids'])

            page = max(page, 1)
            page_size = max(min(page_size, 100), 1)
            offset = (page - 1) * page_size

            res = collection.get(
                where=where,
                include=['documents', 'metadatas'],
                limit=page_size,
                offset=offset,
            )
            items = [
                {'content': doc, 'metadata': meta or {}}
                for doc, meta in zip(res.get('documents') or [], res.get('metadatas') or [])
            ]
            logger.info(f"浏览文档分块: source={source}, 第{page}页, 返回 {len(items)} 条（共 {total} 条）")
            return {'source': source, 'total': total, 'page': page,
                    'page_size': page_size, 'items': items}
        except Exception as e:
            logger.error(f"浏览文档分块失败: source={source}, {str(e)}")
            raise RuntimeError(f"浏览文档分块失败: {str(e)}")

    def hybrid_retrieve(self, query: str, keywords: list, k: int = None):
        """
        混合检索：结合关键词检索和语义检索
        """
        if k is None:
            k = config.retrieve_top_k
        
        try:
            semantic_results = self.vector_store.similarity_search(
                query, k=k, filter=merge_alive_filter()
            )
            logger.info(f"语义检索结果数量: {len(semantic_results)}")

            keyword_results = []
            if keywords:
                for keyword in keywords:
                    results = self.vector_store.similarity_search(
                        keyword, k=k, filter=merge_alive_filter()
                    )
                    keyword_results.extend(results)
                keyword_results = self._deduplicate_docs(keyword_results)
                logger.info(f"关键词检索结果数量: {len(keyword_results)}")
            
            merged_results = self._merge_results(semantic_results, keyword_results, k)
            logger.info(f"合并后检索结果数量: {len(merged_results)}")
            
            if self.reranker and merged_results:
                merged_results = self.reranker.rerank(query, merged_results, top_k=config.rerank_top_k)
                logger.info(f"Ollama 重排序完成，返回 {len(merged_results)} 个文档")
            
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
                filter=merge_alive_filter({"title": title})
            )
            logger.info(f"按文献标题过滤检索: title={title}, 结果数量: {len(filtered_results)}")

            # 对指定文献的检索结果也进行重排序
            if self.reranker and filtered_results:
                filtered_results = self.reranker.rerank(query, filtered_results, top_k=config.rerank_top_k)
                logger.info(f"指定文献检索重排序完成，返回 {len(filtered_results)} 个文档")

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

            # 对合并后的结果进行最终重排序
            if self.reranker and merged:
                merged = self.reranker.rerank(query, merged, top_k=config.rerank_top_k)
                logger.info(f"分源检索最终重排序完成，返回 {len(merged)} 个文档")

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
            # 关键词检索分数：位置越靠前分数越高（下限为 0，避免去重合并后出现负分）
            score = max(k - i, 0) / k * config.KEYWORD_WEIGHT
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
