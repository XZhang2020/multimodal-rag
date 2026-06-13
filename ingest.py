import os
from collections import Counter
from src.document_parser import DocumentParser
from src.chunker import SmartChunker
from src.vector_store import VectorStore
from src.retriever import Retriever


def _make_progress():
    """返回一个进度回调：图片/表格慢操作时原地刷新 '第几张/共几张'。"""
    _stage_cn = {"figure": "图片", "table": "表格"}

    def progress(stage, done, total):
        label = _stage_cn.get(stage, stage)
        end = "\n" if done == total else ""
        print(f"\r    [{label}] 处理中 {done}/{total} ...", end=end, flush=True)

    return progress


def ingest_all_documents():
    print("=" * 50)
    print(" 开始构建知识库")
    print("=" * 50)

    parser = DocumentParser()
    chunker = SmartChunker()
    vector_store = VectorStore()
    # Bug7：复用同一个 vector_store，避免重复实例化
    retriever = Retriever(vector_store=vector_store)

    # ingest 阶段需要清空历史脏数据，显式调用一次
    vector_store.reset_collection()

    # 解析文档
    documents = []
    data_dir = "data"
    files = [f for f in sorted(os.listdir(data_dir))
             if os.path.isfile(os.path.join(data_dir, f))]
    progress = _make_progress()

    for i, filename in enumerate(files, start=1):
        file_path = os.path.join(data_dir, filename)
        print(f"\n[{i}/{len(files)}] 解析：{filename}")
        docs = parser.parse(file_path, progress=progress)
        # 本文件各模态分布，一眼看清抽到了什么
        dist = Counter(d["metadata"].get("modality", "?") for d in docs)
        dist_str = "，".join(f"{k} {v}" for k, v in dist.items()) or "空"
        print(f"    -> {len(docs)} 个元素（{dist_str}）")
        documents.extend(docs)

    print(f"\n解析完成，共 {len(documents)} 个元素")

    # 分层分块：父块 + 子块
    parent_docs, child_docs = chunker.split_documents(documents)
    print(f"分块完成：父块 {len(parent_docs)}，子块 {len(child_docs)}")

    # ========== 父子块全部入库 ==========
    print("正在存入父块...")
    vector_store.add_parent_documents(parent_docs)

    print("正在生成子块向量并入库...")
    vector_store.add_child_documents(child_docs)

    # 基于子块构建BM25索引（检索用子块）
    retriever.build_bm25_index(child_docs)
    retriever.save_bm25_index("bm25_index.pkl")

    print("\n[OK] 知识库构建完成！")
    return True

if __name__ == "__main__":
    ingest_all_documents()
