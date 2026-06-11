from openai import OpenAI
from src.config import settings


SYSTEM_PROMPT = """你是一个严谨的知识库问答助手，必须遵守下面的全部规则：

1. 仅依据【参考资料】中的内容回答，绝不使用资料外的常识或推断。
2. 如果【参考资料】不足以回答，只回复：无法解答。不要编造，不要补充常识。
3. 答案要直接、简洁，不要寒暄、不要复述问题。
4. **引用必须自带证据**：每一个事实性陈述后面都要紧跟一个引用块，格式严格为：
   〔来源N · 文件名 p.页码：「原文摘录」〕
   - 「原文摘录」必须从对应来源中**逐字截取**，长度控制在 15~40 个汉字（或等价英文），能直接支撑你的陈述。
   - 同一句话同时被多条来源支撑时，并列附多个引用块。
   - 不允许出现没有引用块的事实陈述；也不允许出现不带「原文摘录」的空引用。
5. 严禁把整段原文复制粘贴当作答案，要先用自己的话总结，再用引用块给出证据。
"""


def _format_context(context_docs):
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "未知文件")
        page = meta.get("page", "?")
        blocks.append(f"【来源{i} · {source} p.{page}】\n{doc['content']}")
    return "\n\n".join(blocks)


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.llm_model

    def generate_answer(self, query, context_docs):
        if not context_docs:
            return "无法解答：知识库中未检索到相关内容。"

        context = _format_context(context_docs)

        user_prompt = (
            f"问题：{query}\n\n"
            f"参考资料：\n{context}\n\n"
            f"请按 system 中的规则作答，每条事实后必须带「原文摘录」的引用块。"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.choices[0].message.content
