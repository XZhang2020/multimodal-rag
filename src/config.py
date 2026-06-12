from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = ""
    # LLM配置
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    
    # Embedding配置
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    
    # 精排配置
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 5
    
    # 分块配置
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 400
    child_chunk_size: int = 400
    child_chunk_overlap: int = 80
    
    # 检索配置
    hybrid_retrieval_top_k: int = 20

    # 多轮对话配置
    conversation_history_turns: int = 5  # 滑动窗口保留最近 N 轮（user+assistant 各算一条）
    rewrite_history_turns: int = 2       # query 改写时只看最近 N 轮，避免远距离上下文干扰指代消解

    # Qdrant配置
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "multimodal_rag"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        extra = "ignore"

settings = Settings()