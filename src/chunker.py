from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class SmartChunker:
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.parent_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            is_separator_regex=False
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            is_separator_regex=False
        )
    
    def split_documents(self, documents):
        """
        实现父子分块策略
        返回：(parent_docs, child_docs)
        """
        parent_docs = []
        child_docs = []
        
        for doc in documents:
            # 先生成父块
            parent_chunks = self.parent_splitter.split_text(doc["content"])
            
            for parent_idx, parent_content in enumerate(parent_chunks):
                parent_id = f"{doc['metadata']['source']}_{doc['metadata']['page']}_{parent_idx}"
                
                parent_doc = {
                    "id": parent_id,
                    "content": parent_content,
                    "metadata": {**doc["metadata"], "parent_id": parent_id}
                }
                parent_docs.append(parent_doc)
                
                # 为每个父块生成子块
                child_chunks = self.child_splitter.split_text(parent_content)
                
                for child_idx, child_content in enumerate(child_chunks):
                    child_doc = {
                        "id": f"{parent_id}_{child_idx}",
                        "content": child_content,
                        "metadata": {
                            **doc["metadata"],
                            "parent_id": parent_id,
                            "child_idx": child_idx
                        }
                    }
                    child_docs.append(child_doc)
        
        return parent_docs, child_docs
