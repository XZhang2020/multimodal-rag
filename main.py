# main.py 现在超级轻量！秒启动！
from src.retriever import Retriever
from src.generator import Generator

def main():
    print("正在加载索引...")
    
    # 初始化（Retriever 内部已自动加载 bm25_index.pkl 并重建 BM25 实例）
    retriever = Retriever()
    generator = Generator()

    print("\n[OK] RAG 服务已启动！")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "quit":
            break

        docs = retriever.retrieve(query)
        ans = generator.generate_answer(query, docs)
        print("\n[答案] ", ans)

if __name__ == "__main__":
    main()