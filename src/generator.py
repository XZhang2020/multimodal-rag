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


REWRITE_SYSTEM_PROMPT = """你的任务：根据【对话历史】把【当前问题】改写成一个独立完整、不依赖上文的检索问题。

规则：
1. 只解析当前问题中的指代（"它"、"这个"、"刚才"、"上面提到的"等），把它们替换成具体名词。
2. 不要新增历史中没有的信息，不要扩写、不要总结、不要回答问题。
3. 如果当前问题本身已经独立完整，原样返回，不要修改。
4. 直接输出改写后的问题本身，不要任何解释、引号、前缀（不要写"改写后："之类）。
"""


_REWRITE_PREFIXES = (
    "改写后的问题：", "改写后：", "独立问题：",
    "改写：", "答：", "Q:", "Question:",
)


def _format_context(context_docs):
    # 模态标签：让 LLM 和引用块知道这条来源是正文/表格/图/扫描件
    _MOD_LABEL = {
        "table": "表格", "figure": "图", "ocr_text": "扫描件", "text": "正文",
    }
    blocks = []
    for i, doc in enumerate(context_docs, start=1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "未知文件")
        page = meta.get("page", "?")
        mod = _MOD_LABEL.get(meta.get("modality", "text"), "正文")
        blocks.append(f"【来源{i} · {source} p.{page} · {mod}】\n{doc['content']}")
    return "\n\n".join(blocks)


def _strip_rewrite_artifacts(text: str) -> str:
    text = text.strip().strip("\"'`「」“”")
    for prefix in _REWRITE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.llm_model

    def rewrite_query(self, query: str, history: list) -> str:
        """根据对话历史把含指代的 query 改写成 standalone query。
        失败 / 输出异常一律回退到原 query，保证检索链路不会被改写器拖挂。"""
        if not history:
            return query

        # 只用最近 N 轮做指代消解（远距离上下文对改写没帮助，反而增加 token 和干扰）
        recent = history[-settings.rewrite_history_turns * 2:]
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        history_text = "\n".join(lines)

        user_prompt = (
            f"【对话历史】\n{history_text}\n\n"
            f"【当前问题】\n{query}\n\n"
            f"【改写后的独立问题】"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = _strip_rewrite_artifacts(resp.choices[0].message.content or "")
        except Exception:
            return query

        # 兜底：空 / 异常长 → 回退原 query
        if not rewritten:
            return query
        if len(rewritten) > max(len(query) * 4, 120):
            return query
        return rewritten

    def generate_answer(self, query, context_docs, history=None):
        if not context_docs:
            return "无法解答：知识库中未检索到相关内容。"

        history = history or []
        context = _format_context(context_docs)
        user_prompt = (
            f"问题：{query}\n\n"
            f"参考资料：\n{context}\n\n"
            f"请按 system 中的规则作答，每条事实后必须带「原文摘录」的引用块。"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 历史只塞 user/assistant 文本，不重复带 context（context 只挂当前轮）
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.choices[0].message.content
