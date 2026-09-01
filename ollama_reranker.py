# Ollama 重排序器服务
import logging
from typing import List, Literal, Optional

import httpx
import numpy as np
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RerankMode = Literal["rerank_api", "embed", "generate"]


class OllamaReranker:
    """使用 Ollama 本地 rerank 模型对检索结果重排序。"""

    def __init__(
        self,
        model_name: str = "bbjson/bge-reranker-base:latest",
        base_url: str = "http://localhost:11434",
        instruction: str = "Please judge relevance.",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.instruction = instruction
        self._embeddings = None

        self._verify_model_exists()
        self.mode = self._detect_mode()
        logger.info(f"Ollama 重排序模型初始化成功: {model_name} (mode={self.mode})")

    def _verify_model_exists(self) -> None:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            installed = {model["name"] for model in response.json().get("models", [])}
        except Exception as exc:
            raise RuntimeError(
                f"无法连接 Ollama 服务 ({self.base_url})，请先运行 `ollama serve`: {exc}"
            ) from exc

        if self.model_name not in installed:
            available = ", ".join(sorted(installed)) or "（无已安装模型）"
            raise RuntimeError(
                f"Ollama 模型 '{self.model_name}' 未找到。"
                f"请执行 `ollama pull {self.model_name}`，"
                f"或在 config_data.py 中把 OLLAMA_RERANKER_MODEL 改成已安装的模型。"
                f"当前已安装: {available}"
            )

    def _detect_mode(self) -> RerankMode:
        try:
            response = httpx.post(
                f"{self.base_url}/api/rerank",
                json={
                    "model": self.model_name,
                    "query": "test",
                    "documents": ["test document"],
                },
                timeout=30.0,
            )
            if response.status_code == 200:
                return "rerank_api"
        except Exception:
            pass

        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": "test"},
                timeout=30.0,
            )
            if response.status_code == 200:
                from langchain_ollama import OllamaEmbeddings

                self._embeddings = OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.base_url,
                )
                logger.warning(
                    "当前 Ollama 版本不支持 /api/rerank，已回退到 embed 模式。"
                    "该模式仅适用于支持 embedding 的模型，重排质量可能不如专用 rerank API。"
                )
                return "embed"
        except Exception:
            pass

        return "generate"

    @staticmethod
    def compute_similarity(query_embedding: List[float], doc_embedding: List[float]) -> float:
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))

    def _build_qwen_prompt(self, query: str, document: str) -> str:
        return (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "\n"
            f"<|im_start|>user\n<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n<Document>: {document}\n"
            "<|im_start|>assistant\n"
        )

    def _score_with_generate(self, query: str, document: str) -> float:
        prompt = self._build_qwen_prompt(query, document)
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 1},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip().lower()
        if text.startswith("yes"):
            return 1.0
        if text.startswith("no"):
            return 0.0
        return 0.5

    def _rerank_with_api(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        response = httpx.post(
            f"{self.base_url}/api/rerank",
            json={
                "model": self.model_name,
                "query": query,
                "documents": [doc.page_content for doc in docs],
                "top_k": top_k,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        ranked_docs = []
        for item in results[:top_k]:
            index = item.get("index")
            if index is None or index >= len(docs):
                continue
            doc = docs[index]
            score = item.get("relevance_score", item.get("score"))
            if score is not None:
                doc.metadata["rerank_score"] = score
            ranked_docs.append(doc)
        return ranked_docs or docs[:top_k]

    def _rerank_with_embed(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        query_embedding = self._embeddings.embed_query(query)
        doc_embeddings = self._embeddings.embed_documents([doc.page_content for doc in docs])
        scored_docs = []
        for doc, embedding in zip(docs, doc_embeddings):
            score = self.compute_similarity(query_embedding, embedding)
            doc.metadata["rerank_score"] = score
            scored_docs.append((doc, score))
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def _rerank_with_generate(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        scored_docs = []
        for doc in docs:
            score = self._score_with_generate(query, doc.page_content)
            doc.metadata["rerank_score"] = score
            scored_docs.append((doc, score))
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        if not docs:
            logger.info("没有文档需要重排序")
            return docs

        try:
            if self.mode == "rerank_api":
                result = self._rerank_with_api(query, docs, top_k)
            elif self.mode == "embed":
                result = self._rerank_with_embed(query, docs, top_k)
            else:
                result = self._rerank_with_generate(query, docs, top_k)

            logger.info(f"重排序完成 (mode={self.mode})，返回 {len(result)} 个文档")
            return result
        except Exception as exc:
            logger.error(f"文档重排序失败: {exc}")
            return docs[:top_k]


if __name__ == "__main__":
    import config_data as config

    reranker = OllamaReranker(
        model_name=config.OLLAMA_RERANKER_MODEL,
        base_url=config.OLLAMA_HOST,
    )

    test_docs = [
        Document(
            page_content="Python 是一种高级编程语言，非常适合数据科学和机器学习。",
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="Java 是一种面向对象的编程语言，广泛应用于企业级开发。",
            metadata={"source": "test2.txt"},
        ),
        Document(
            page_content="机器学习是人工智能的一个分支，使用算法让计算机从数据中学习。",
            metadata={"source": "test3.txt"},
        ),
    ]

    reranked_docs = reranker.rerank("Python 在数据科学中的应用", test_docs, top_k=2)
    print("重排序结果:")
    for index, doc in enumerate(reranked_docs, start=1):
        score = doc.metadata.get("rerank_score")
        score_text = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
        print(f"{index}. {doc.page_content}{score_text}")
        print(f"   来源: {doc.metadata.get('source', '未知')}")
