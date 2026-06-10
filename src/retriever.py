from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.config import settings
from src.vector_store import VectorStore

class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.reranker = CrossEncoder(
            settings.rerank_model,
            trust_remote_code=True
        )
        self.bm25_index = None
        self.all_documents = None
    
    def build_bm25_index(self, documents):
        """构建BM25索引用于关键词检索"""
        self.all_documents = documents
        tokenized_docs = [doc["content"].split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def bm25_search(self, query, top_k=20):
        """BM25关键词检索"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_bm25_index first.")
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取Top K结果
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
        """混合检索：BM25 + 向量检索"""
        # 分别进行BM25和向量检索
        bm25_results = self.bm25_search(query, top_k=top_k)
        vector_results = self.vector_store.search(query, top_k=top_k)
        
        # 合并结果并去重
        all_results = {}
        
        for result in bm25_results:
            all_results[result["id"]] = result
        
        for result in vector_results:
            if result["id"] not in all_results:
                all_results[result["id"]] = result
        
        return list(all_results.values())
    
    def rerank(self, query, documents, top_k=5):
        """精排：使用交叉编码器重新打分"""
        if not documents:
            return []
        
        # 准备精排输入
        pairs = [(query, doc["content"]) for doc in documents]
        
        # 精排打分
        scores = self.reranker.predict(pairs)
        
        # 将分数添加到文档中
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
        
        # 按精排分数排序
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return documents[:top_k]
    
    def retrieve(self, query):
        """完整的检索流程"""
        # 1. 混合检索召回Top 20
        hybrid_results = self.hybrid_search(query, top_k=settings.hybrid_retrieval_top_k)
        
        if not hybrid_results:
            return []
        
        # 2. 精排得到Top 5
        reranked_results = self.rerank(query, hybrid_results, top_k=settings.rerank_top_k)
        
        # 3. 获取对应的父块内容
        parent_ids = list(set([doc["metadata"]["parent_id"] for doc in reranked_results]))
        parent_docs = self.vector_store.get_parent_documents(parent_ids)
        
        # 按精排分数排序父块
        parent_id_to_score = {
            doc["metadata"]["parent_id"]: doc["rerank_score"]
            for doc in reranked_results
        }
        
        parent_docs.sort(
            key=lambda x: parent_id_to_score.get(x["id"], 0),
            reverse=True
        )
        
        return parent_docs
    
    def save_bm25_index(self, path="bm25_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.bm25_index, self.all_documents), f)

    def load_bm25_index(self, path="bm25_index.pkl"):
        with open(path, "rb") as f:
            self.bm25_index, self.all_documents = pickle.load(f)