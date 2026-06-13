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
    
    # ===== 多模态解析配置 =====
    # 图片：是否调用 VLM 生成语义描述（无可用 VLM 时自动降级为仅 OCR/占位）
    enable_vlm_caption: bool = True
    # VLM 独立端点：默认指向本地 ollama qwen3-vl（免费、不受文本 API 多模态限制）
    vlm_model: str = "qwen3-vl:4b"
    vlm_base_url: str = "http://localhost:11434/v1"
    vlm_api_key: str = "ollama"             # ollama 不校验，占位即可
    vlm_no_think: bool = True               # qwen3 思考模式会吃光 token，关掉
    # qwen3-vl 思考通道约耗 600-700 token，必须给足预算否则正文为空（实测 512 全空）
    vlm_max_tokens: int = 1536
    vlm_timeout: int = 120                  # 单张图超时（秒），超时降级
    # VLM 失败的短期负缓存：临时故障（本地模型偶发 OOM）TTL 内跳过重试，过期后允许重试
    vlm_fail_ttl: int = 21600               # 6 小时
    # 表格：是否用 LLM 生成 NL 摘要进向量库（无 key 时降级为表头/首行拼接）
    enable_table_summary: bool = True
    # 扫描页判定：PyMuPDF 抽出的文本字符数低于该阈值，判为扫描页 → 走 PaddleOCR
    ocr_text_coverage_threshold: int = 20
    paddle_ocr_lang: str = "ch"             # 中文 SOTA，刻意替换 docling 默认 EasyOCR
    # 解析缓存：按内容哈希缓存 OCR/VLM 结果，避免重复 ingest 重复付费
    enable_parse_cache: bool = True

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