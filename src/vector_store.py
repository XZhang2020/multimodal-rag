import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from src.config import settings
from src.embedder import Embedder

# 单次 scroll 拉取的安全上限，避免 Qdrant 默认 10 条限制
_SCROLL_PAGE_SIZE = 1024


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.embedder = Embedder()
        self.collection_name = settings.qdrant_collection_name

        # 只确保 collection 存在，不动已有数据
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def reset_collection(self):
        """显式清空并重建 collection，仅在 ingest 时调用"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE
            )
        )

    # 存入父块（无向量，仅存原文）
    def add_parent_documents(self, parent_docs):
        points = []
        for doc in parent_docs:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0.0] * settings.embedding_dim,  # 占位空向量
                    payload={
                        "doc_type": "parent",
                        "origin_id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    }
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    # 存入子块（带向量，用于检索）
    def add_child_documents(self, child_docs):
        if not child_docs:
            return
        texts = [doc["content"] for doc in child_docs]
        embeddings = self.embedder.embed(texts)
        points = []
        for idx, doc in enumerate(child_docs):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx],
                    payload={
                        "doc_type": "child",
                        "origin_id": doc["id"],
                        "parent_origin_id": doc["metadata"]["parent_id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    }
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

    # 向量检索（仅查子块，服务端过滤 doc_type）
    def search(self, query, top_k=20):
        query_vector = self.embedder.embed(query)[0]
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value="child")
                    )
                ]
            )
        )
        return [
            {
                "id": hit.payload["origin_id"],
                "parent_id": hit.payload["parent_origin_id"],
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            }
            for hit in results.points
        ]

    # 根据父块原始ID批量取父块内容（服务端过滤 + 分页）
    def get_parent_documents(self, parent_origin_ids):
        if not parent_origin_ids:
            return []

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value="parent")
                ),
                FieldCondition(
                    key="origin_id",
                    match=MatchAny(any=list(parent_origin_ids))
                )
            ]
        )

        collected = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=_SCROLL_PAGE_SIZE,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(points)
            if next_offset is None:
                break

        return [
            {
                "id": p.payload["origin_id"],
                "content": p.payload["content"],
                "metadata": p.payload["metadata"]
            }
            for p in collected
        ]
