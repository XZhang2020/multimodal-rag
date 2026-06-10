from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from src.config import settings
from src.embedder import Embedder

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.embedder = Embedder()
        self.collection_name = settings.qdrant_collection_name
        
        # 创建集合（如果不存在）
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, documents):
        """添加文档到向量数据库"""
        points = []
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedder.embed(texts)
        
        # 用循环下标作为Qdrant合法数字ID，原始id存入payload
        for idx, doc in enumerate(documents):
            points.append(
                PointStruct(
                    id=idx,          # 合法数字ID
                    vector=embeddings[idx],
                    payload={
                        "origin_id": doc["id"],  # 保留原始字符串ID
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    }
                )
            )
        
        # 批量插入
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
    
    def search(self, query, top_k=20):
        """向量检索"""
        query_vector = self.embedder.embed(query)[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

        return [
            {
                "id": hit.payload["origin_id"],
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            }
            for hit in results.points
        ]
    
    def get_parent_documents(self, parent_ids):
        """根据父块ID获取完整的父块内容"""
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=parent_ids
        )
        
        return [
            {
                "id": hit.payload["origin_id"],
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"]
            }
            for hit in results
        ]