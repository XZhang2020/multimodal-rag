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

        多模态适配：metadata.atomic=True 的元素（表格/图片）整体成块，不递归切分，
        否则 markdown 表格会被切碎、图片描述会断裂。
        """
        parent_docs = []
        child_docs = []

        for doc in documents:
            if doc.get("metadata", {}).get("atomic"):
                self._add_atomic(doc, parent_docs, child_docs)
                continue
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

    def _add_atomic(self, doc, parent_docs, child_docs):
        """表格/图片：父块存完整内容（喂 LLM），子块只放检索文本（向量召回）。

        表格父块含 markdown 原表；子块取摘要段（[表格原文] 之前），避免裸表入向量库。
        """
        meta = doc["metadata"]
        parent_id = f"{meta['source']}_{meta['page']}_{meta.get('element_id', 'atomic')}"

        parent_docs.append({
            "id": parent_id,
            "content": doc["content"],
            "metadata": {**meta, "parent_id": parent_id},
        })

        retrieval_text = doc["content"].split("\n\n[表格原文]\n", 1)[0].strip()
        child_docs.append({
            "id": f"{parent_id}_0",
            "content": retrieval_text or doc["content"],
            "metadata": {**meta, "parent_id": parent_id, "child_idx": 0},
        })
