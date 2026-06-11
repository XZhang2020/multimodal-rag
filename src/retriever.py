import pickle
import os
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorStore


def _tokenize(text: str):
    """中文用 jieba 切，英文/数字保留原 token；过滤空白"""
    return [t for t in jieba.lcut(text) if t.strip()]


class Retriever:
    def __init__(self, vector_store: VectorStore = None):
        # Bug7：允许外部注入，避免 ingest 内重复实例化
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.reranker = CrossEncoder(
            settings.rerank_model,
            trust_remote_code=True,
            device="cpu"
        )
        self.bm25_index = None
        self.all_child_docs = None

        # 启动自动加载索引 + 重建BM25实例
        index_path = "bm25_index.pkl"
        if os.path.exists(index_path):
            self.load_bm25_index(index_path)
            if self.all_child_docs:
                tokenized_docs = [_tokenize(doc["content"]) for doc in self.all_child_docs]
                self.bm25_index = BM25Okapi(tokenized_docs)

    def build_bm25_index(self, child_docs):
        """基于子块构建BM25"""
        self.all_child_docs = child_docs
        tokenized_docs = [_tokenize(doc["content"]) for doc in child_docs]
        self.bm25_index = BM25Okapi(tokenized_docs)

    def save_bm25_index(self, path="bm25_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.all_child_docs, f)

    def load_bm25_index(self, path="bm25_index.pkl"):
        with open(path, "rb") as f:
            self.all_child_docs = pickle.load(f)
        # 加载文档后必须同时重建 BM25 实例，否则 bm25_search 会报错
        if self.all_child_docs:
            tokenized_docs = [_tokenize(doc["content"]) for doc in self.all_child_docs]
            self.bm25_index = BM25Okapi(tokenized_docs)

    def bm25_search(self, query, top_k=20):
        if self.bm25_index is None:
            raise ValueError("请先构建/加载BM25索引")
        tokenized_query = _tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {
                "id": self.all_child_docs[i]["id"],
                "parent_id": self.all_child_docs[i]["metadata"]["parent_id"],
                "content": self.all_child_docs[i]["content"],
                "metadata": self.all_child_docs[i]["metadata"],
                "score": float(scores[i])
            }
            for i in top_indices
        ]

    def hybrid_search(self, query, top_k=20):
        bm25_res = self.bm25_search(query, top_k=top_k)
        vec_res = self.vector_store.search(query, top_k=top_k)
        # 去重
        merge = {r["id"]: r for r in bm25_res}
        for r in vec_res:
            if r["id"] not in merge:
                merge[r["id"]] = r
        return list(merge.values())

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.reranker.predict(pairs)
        for idx, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[idx])
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]

    def retrieve(self, query):
        # 1. 混合检索得到子块
        hybrid_res = self.hybrid_search(query, top_k=settings.hybrid_retrieval_top_k)
        if not hybrid_res:
            return []
        # 2. 精排子块
        reranked_child = self.rerank(query, hybrid_res, settings.rerank_top_k)
        # 3. 提取所有关联父块ID
        parent_ids = list(set(item["parent_id"] for item in reranked_child))
        # 4. 拉取完整父块内容
        parent_docs = self.vector_store.get_parent_documents(parent_ids)
        # 5. 按"同父块下子块的最高分"给父块排序（Bug5：取 max，而不是 last-write-wins）
        score_map = {}
        for item in reranked_child:
            pid = item["parent_id"]
            score_map[pid] = max(score_map.get(pid, float("-inf")), item["rerank_score"])
        parent_docs.sort(key=lambda x: score_map.get(x["id"], float("-inf")), reverse=True)
        return parent_docs
