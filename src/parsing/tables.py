"""表格处理：双存策略。

问题：表格直接转 markdown 入向量库，召回效果差——向量模型对"行列数字"
不敏感，用户问"某年营收多少"很难命中一张裸表。

方案：
- raw  = markdown 原表，喂给 LLM（结构完整，能精确读数）；
- text = LLM 生成的自然语言摘要，进向量库/BM25 做召回（语义友好）。
无 key 时摘要降级为"标题 + 表头 + 首行"拼接，仍比裸 markdown 可召回。
"""
from __future__ import annotations

from openai import OpenAI

from src.config import settings
from .models import Element
from .cache import ParseCache

_CACHE = ParseCache(enabled=settings.enable_parse_cache)
_CLIENT = None

_SUMMARY_PROMPT = (
    "下面是一张从文档中抽取的表格（markdown）。请用简体中文写一段检索友好的摘要：\n"
    "说明这张表在讲什么、包含哪些列/指标、覆盖的主要对象或时间范围、可回答哪类问题。\n"
    "不要逐格复述数据，控制在 150 字内，只输出摘要本身。\n\n表格：\n"
)


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=settings.openai_api_key,
                         base_url=settings.openai_base_url)
    return _CLIENT


def _fallback_summary(markdown: str, caption: str) -> str:
    """无 LLM 时的降级摘要：标题 + 表头 + 首行。"""
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    head = lines[:2] if len(lines) >= 2 else lines
    parts = []
    if caption:
        parts.append(caption)
    parts.append("表格内容：" + " ｜ ".join(
        c.strip(" |") for c in (head[0].split("|") if head else []) if c.strip(" |")
    ))
    return "；".join(p for p in parts if p).strip()


def summarize_table(el: Element) -> Element:
    """就地填充表格 Element 的 text（NL 摘要）。raw 保持 markdown 不变。"""
    markdown = el.raw or ""
    caption = el.extra.get("caption", "")
    if not markdown:
        el.text = caption
        return el

    if settings.enable_table_summary and settings.openai_api_key:
        cached = _CACHE.get(markdown.encode("utf-8"), namespace="table")
        if cached is not None:
            el.text = cached
            return el
        try:
            resp = _client().chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": _SUMMARY_PROMPT + markdown}],
                temperature=0.0,
                max_tokens=300,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                if caption:
                    summary = f"{caption}。{summary}"
                _CACHE.set(markdown.encode("utf-8"), namespace="table", value=summary)
                el.text = summary
                return el
        except Exception as e:
            print(f"[TABLE] 摘要生成失败，降级为表头拼接: {e!r}")

    el.text = _fallback_summary(markdown, caption)
    return el
