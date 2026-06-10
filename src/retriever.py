import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.reranker = CrossEncoder(
            settings.rerank_model,
            trust_remote_code=True,
            device="cpu"
        )
        self.bm25_index = None
        self.all_documents = None
        # 新增：业务字符串ID -> Qdrant数字ID 映射
        self.id_mapping = {}

    def build_bm25_index(self, documents):
        self.all_documents = documents
        # 构建映射表
        for idx, doc in enumerate(documents):
            self.id_mapping[doc["id"]] = idx
        tokenized_docs = [doc["content"].split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

    def save_bm25_index(self, path="bm25_index.pkl"):
        # 连同映射表一起保存
        with open(path, "wb") as f:
            pickle.dump((self.bm25_index, self.all_documents, self.id_mapping), f)

    def load_bm25_index(self, path="bm25_index.pkl"):
        with open(path, "rb") as f:
            self.bm25_index, self.all_documents, self.id_mapping = pickle.load(f)

    def bm25_search(self, query, top_k=20):
        if self.bm25_index is None:
            raise ValueError("请先 build 或 load BM25 索引")
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            {
                "id": self.all_documents[i]["id"],
                "content": self.all_documents[i]["content"],
                "metadata": self.all_documents[i]["metadata"],
                "score": scores[i]
            }
            for i in top_indices
        ]

    def hybrid_search(self, query, top_k=20):
        bm25_results = self.bm25_search(query, top_k=top_k)
        vector_results = self.vector_store.search(query, top_k=top_k)
        all_results = {r["id"]: r for r in bm25_results}
        for r in vector_results:
            if r["id"] not in all_results:
                all_results[r["id"]] = r
        return list(all_results.values())

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.reranker.predict(pairs)
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]

    def retrieve(self, query):
        hybrid_results = self.hybrid_search(query, top_k=settings.hybrid_retrieval_top_k)
        if not hybrid_results:
            return []
        reranked = self.rerank(query, hybrid_results, settings.rerank_top_k)
        # 取出父级业务ID
        parent_biz_ids = list(set(d["metadata"]["parent_id"] for d in reranked))
        # 转为 Qdrant 合法数字ID
        parent_store_ids = [self.id_mapping[bid] for bid in parent_biz_ids if bid in self.id_mapping]
        # 用数字ID查询
        parent_docs = self.vector_store.get_parent_documents(parent_store_ids)

        score_map = {d["id"]: d["rerank_score"] for d in reranked}
        parent_docs.sort(key=lambda x: score_map.get(x["id"], 0), reverse=True)
        return parent_docs