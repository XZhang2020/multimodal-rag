from src.config import settings
from src.retriever import Retriever
from src.generator import Generator


def main():
    print("正在加载索引...")

    # 初始化（Retriever 内部已自动加载 bm25_index.pkl 并重建 BM25 实例）
    retriever = Retriever()
    generator = Generator()

    print("\n[OK] RAG 服务已启动！")
    print("    输入 quit  退出")
    print("    输入 reset 清空对话历史")

    history = []
    max_msgs = settings.conversation_history_turns * 2  # turn = user + assistant

    while True:
        query = input("\n请输入问题：").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            history = []
            print("[已清空对话历史]")
            continue

        # 1. 改写：把含指代的 query 改成 standalone（首轮 history=[] 自动跳过）
        standalone_query = generator.rewrite_query(query, history)
        if standalone_query != query:
            print(f"[改写] {standalone_query}")

        # 2. 检索：用改写后的 query 召回，提高指代场景的命中率
        docs = retriever.retrieve(standalone_query)

        # 3. 生成：原始 query + 历史一起喂给 LLM，保证答案语气连贯
        ans = generator.generate_answer(query, docs, history=history)
        print("\n[答案]", ans)

        # 4. 写入历史（存原始 query，不存改写后的，方便用户回看）
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ans})

        # 5. 滑动窗口裁剪
        if len(history) > max_msgs:
            history = history[-max_msgs:]


if __name__ == "__main__":
    main()
