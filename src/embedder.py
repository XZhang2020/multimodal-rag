from sentence_transformers import SentenceTransformer
from src.config import settings
import numpy as np

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            settings.embedding_model,
            trust_remote_code=True
        )
        self.dimension = settings.embedding_dim
    
    def embed(self, texts):
        """生成文本向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # 转换为float32列表，方便存储
        return embeddings.astype(np.float32).tolist()