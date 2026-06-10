import os
from src.document_parser import DocumentParser
from src.chunker import SmartChunker
from src.vector_store import VectorStore
from src.retriever import Retriever

def ingest_all_documents():
    print("=" * 50)
    print(" 开始构建知识库")
    print("=" * 50)

    # 1. 初始化组件
    parser = DocumentParser()
    chunker = SmartChunker()
    vector_store = VectorStore()
    retriever = Retriever()

    # 2. 解析文档
    documents = []
    data_dir = "data"

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            print(f"解析：{filename}")
            docs = parser.parse(file_path)
            documents.extend(docs)

    print(f"\n解析完成，共 {len(documents)} 个元素")

    # 3. 分块
    parent_docs, child_docs = chunker.split_documents(documents)
    print(f"分块完成：父块 {len(parent_docs)}，子块 {len(child_docs)}")

    # 4. 存入向量数据库
    print("正在生成向量并入库...")
    vector_store.add_documents(child_docs)

    # 5. 构建 BM25 索引并保存到本地
    retriever.build_bm25_index(child_docs)
    retriever.save_bm25_index("bm25_index.pkl")

    print("\n✅ 知识库构建完成！")
    return True

if __name__ == "__main__":
    ingest_all_documents()