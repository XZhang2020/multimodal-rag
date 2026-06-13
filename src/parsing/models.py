"""parsing 层统一中间表示。

整条解析链（docling / PaddleOCR / VLM）都先产出 Element，再由 assembler
归一成下游 chunker/embedder/vector_store 认的 ``{"content", "metadata"}`` 契约。
好处：多模态的差异都收敛在 Element 里，下游一行不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# 模态枚举：决定 assembler 怎么拼 content、chunker 是否当原子块
MODALITY_TEXT = "text"          # 正文段落（含 docling 抽出的电子文本）
MODALITY_TABLE = "table"        # 表格：raw=markdown 原表，text=NL 摘要
MODALITY_FIGURE = "figure"      # 图片：VLM caption + 图内 OCR
MODALITY_OCR_TEXT = "ocr_text"  # 扫描页整页 OCR 文本

# 原子模态：chunker 不得把它们切碎（markdown 表格被递归切分会烂掉）
ATOMIC_MODALITIES = frozenset({MODALITY_TABLE, MODALITY_FIGURE})


@dataclass
class Element:
    """一个可检索的最小语义单元。

    text : 进 embedding / BM25 的可检索文本（表格=NL 摘要，图=caption+OCR）。
    raw  : 喂给 LLM 的完整原文（表格 markdown），text 不足以还原时用。
    """
    modality: str
    text: str
    source: str
    page: int = 0
    element_id: str = ""
    bbox: Optional[tuple[float, float, float, float]] = None  # (l, t, r, b)
    raw: str = ""                       # 表格 markdown 等结构化原文
    extra: dict = field(default_factory=dict)  # caption / ocr / 表格行列数等溯源信息

    def is_empty(self) -> bool:
        return not (self.text or "").strip() and not (self.raw or "").strip()
