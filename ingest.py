import os
import sys
from collections import Counter
from src.document_parser import DocumentParser
from src.chunker import SmartChunker
from src.vector_store import VectorStore
from src.retriever import Retriever
from src import manifest

DATA_DIR = "data"
BM25_PATH = "bm25_index.pkl"


def _make_progress():
    """返回一个进度回调：图片/表格慢操作时原地刷新 '第几张/共几张'。"""
    _stage_cn = {"figure": "图片", "table": "表格"}

    def progress(stage, done, total):
        label = _stage_cn.get(stage, stage)
        end = "\n" if done == total else ""
        print(f"\r    [{label}] 处理中 {done}/{total} ...", end=end, flush=True)

    return progress


def _parse_files(parser, filenames, progress):
    """解析给定文件列表，返回元素 list。"""
    documents = []
    for i, filename in enumerate(filenames, start=1):
        file_path = os.path.join(DATA_DIR, filename)
        print(f"\n[{i}/{len(filenames)}] 解析：{filename}")
        docs = parser.parse(file_path, progress=progress)
        dist = Counter(d["metadata"].get("modality", "?") for d in docs)
        dist_str = "，".join(f"{k} {v}" for k, v in dist.items()) or "空"
        print(f"    -> {len(docs)} 个元素（{dist_str}）")
        documents.extend(docs)
    return documents


def ingest_all_documents(full: bool = False):
    print("=" * 50)
    print(" 全量重建知识库" if full else " 增量更新知识库")
    print("=" * 50)

    parser = DocumentParser()
    chunker = SmartChunker()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store)
    progress = _make_progress()

    # 全量模式：清库 + 清 manifest，行为等价旧版
    old_manifest = {} if full else manifest.load_manifest()
    if full:
        vector_store.reset_collection()

    diff, new_manifest = manifest.diff_data_dir(DATA_DIR, old_manifest)
    print(f"新增 {len(diff.added)} | 修改 {len(diff.modified)} | "
          f"删除 {len(diff.deleted)} | 未变 {len(diff.unchanged)}")

    if not full and not diff.to_parse and not diff.deleted:
        print("\n[OK] 没有变化，知识库无需更新。")
        return True

    # 1. 删除即将重解析(新增+修改)和已删文件的旧点（全量模式已清库，跳过）。
    #    新增文件也删一次：在干净库上是空操作，但能保证"无 manifest 的老库"上
    #    重跑不产生重复点 —— 整个 ingest 因此幂等。
    sources_to_clear = sorted(set(diff.to_parse) | set(diff.deleted))
    if not full:
        for src in sources_to_clear:
            print(f"清理旧数据：{src}")
            vector_store.delete_by_source(src)

    # 2. 只解析"新增 + 修改"的文件（未变文件零成本跳过）
    documents = _parse_files(parser, diff.to_parse, progress)
    print(f"\n本次解析完成，共 {len(documents)} 个元素")

    # 3. 分块 + 入库（只入本次解析的文件）
    parent_docs, child_docs = chunker.split_documents(documents)
    print(f"分块完成：父块 {len(parent_docs)}，子块 {len(child_docs)}")
    if parent_docs:
        print("正在存入父块...")
        vector_store.add_parent_documents(parent_docs)
    if child_docs:
        print("正在生成子块向量并入库...")
        vector_store.add_child_documents(child_docs)

    # 4. BM25 重建：留存子块（剔除将重解析/已删来源）+ 本次新子块
    if full:
        all_children = child_docs
    else:
        kept = retriever.drop_sources(sources_to_clear)
        all_children = kept + child_docs
    retriever.build_bm25_index(all_children)
    retriever.save_bm25_index(BM25_PATH)
    print(f"BM25 索引重建：共 {len(all_children)} 个子块")

    # 5. 写回 manifest
    manifest.save_manifest(new_manifest)

    print("\n[OK] 知识库更新完成！")
    return True


if __name__ == "__main__":
    full = "--full" in sys.argv
    ingest_all_documents(full=full)
