# main.py 现在超级轻量！秒启动！
from src.retriever import Retriever
from src.generator import Generator

def main():
    print("正在加载索引...")
    
    # 初始化
    retriever = Retriever()
    generator = Generator()

    # 直接加载已经构建好的索引（秒级）
    retriever.load_bm25_index("bm25_index.pkl")

    print("\n🚀 RAG 服务已启动！秒级问答！")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "quit":
            break

        docs = retriever.retrieve(query)
        ans = generator.generate_answer(query, docs)
        print("\n💡 答案：", ans)

if __name__ == "__main__":
    main()