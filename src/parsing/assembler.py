"""Element[] → 下游契约 list[{"content", "metadata"}]。

下游 chunker/embedder/vector_store 只认这两个键，这里负责归一：
- content：可检索文本。表格用 NL 摘要 + markdown 原表拼接（既能召回又能喂 LLM）。
- metadata：带 modality / page / bbox / element_id / raw 做多模态溯源，
  其中 source、page 是 chunker 已依赖的字段，必须保留。
"""
from __future__ import annotations

from .models import Element, MODALITY_TABLE


def _content_for(el: Element) -> str:
    if el.modality == MODALITY_TABLE:
        # 摘要在前（利于召回），markdown 原表在后（喂 LLM 精确读数）
        if el.raw:
            return f"{el.text}\n\n[表格原文]\n{el.raw}"
        return el.text
    return el.text


def to_documents(elements: list[Element]) -> list[dict]:
    docs = []
    for el in elements:
        content = _content_for(el).strip()
        if not content:
            continue
        docs.append({
            "content": content,
            "metadata": {
                "source": el.source,
                "page": el.page,
                "type": el.modality,          # 兼容旧字段名 type
                "modality": el.modality,
                "element_id": el.element_id,
                "bbox": list(el.bbox) if el.bbox else None,
                "atomic": el.modality in ("table", "figure"),
                # 表格原文单独留一份，溯源/重排时可用
                "table_markdown": el.raw if el.modality == MODALITY_TABLE else "",
            },
        })
    return docs
