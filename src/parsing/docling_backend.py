"""docling 后端：电子文档的版面分析 + 表格结构 + 图片定位。

刻意 **关掉 docling 内置 OCR**（do_ocr=False）：
- 有文本层的 PDF/docx 直接抽，不浪费算力；
- 扫描页由 router 单独路由给 PaddleOCR（中文远强于 docling 默认的 EasyOCR）。
docling 在这里只干它最强的事：layout / reading-order / 表格识别 / 图片切出。
"""
from __future__ import annotations

import io
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TextItem, TableItem, PictureItem

from .models import (
    Element,
    MODALITY_TEXT,
    MODALITY_TABLE,
    MODALITY_FIGURE,
)

# docling 里不该进知识库的结构性文本标签（页眉页脚等）
_SKIP_LABELS = {"page_header", "page_footer"}


def _build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False                  # 见模块 docstring：OCR 交给 PaddleOCR
    opts.do_table_structure = True       # 表格结构识别（行列还原）
    opts.generate_picture_images = True  # 切出图片 bytes，供 VLM/OCR 用
    opts.images_scale = 2.0              # 2x 渲染，OCR/VLM 更清晰
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
        }
    )


_CONVERTER: Optional[DocumentConverter] = None


def _converter() -> DocumentConverter:
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_converter()
    return _CONVERTER


def _prov_page_bbox(item):
    """从 prov 取页码与 bbox，缺失则给安全默认。"""
    prov = getattr(item, "prov", None)
    if prov:
        p = prov[0]
        bbox = getattr(p, "bbox", None)
        bbox_t = (bbox.l, bbox.t, bbox.r, bbox.b) if bbox else None
        return getattr(p, "page_no", 0) or 0, bbox_t
    return 0, None


def parse_with_docling(file_path: str, source: str) -> tuple[list[Element], list[Element]]:
    """返回 (text_table_elements, figure_elements)。

    图片单独分一路：它们还要再过 VLM/OCR（慢、可能花钱），由上层决定并发与缓存。
    文本/表格则已是终态。
    """
    result = _converter().convert(file_path)
    doc = result.document

    text_table: list[Element] = []
    figures: list[Element] = []
    seq = 0

    for item, _level in doc.iterate_items():
        seq += 1
        page, bbox = _prov_page_bbox(item)

        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc=doc).strip()
            if not md:
                continue
            text_table.append(Element(
                modality=MODALITY_TABLE,
                text="",                 # NL 摘要由 tables.py 之后填
                raw=md,                  # markdown 原表喂 LLM
                source=source,
                page=page,
                bbox=bbox,
                element_id=f"{source}#tab#{seq}",
                extra={"caption": (item.caption_text(doc) or "").strip()},
            ))

        elif isinstance(item, PictureItem):
            try:
                pil = item.get_image(doc)
            except Exception:
                pil = None
            img_bytes = b""
            if pil is not None:
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="PNG")
                img_bytes = buf.getvalue()
            figures.append(Element(
                modality=MODALITY_FIGURE,
                text="",                 # caption+OCR 由 vision/ocr 之后填
                source=source,
                page=page,
                bbox=bbox,
                element_id=f"{source}#fig#{seq}",
                extra={
                    "caption": (item.caption_text(doc) or "").strip(),
                    "image_bytes": img_bytes,
                },
            ))

        elif isinstance(item, TextItem):
            label = str(getattr(item, "label", "") or "")
            if label in _SKIP_LABELS:
                continue
            txt = (item.text or "").strip()
            if not txt:
                continue
            text_table.append(Element(
                modality=MODALITY_TEXT,
                text=txt,
                source=source,
                page=page,
                bbox=bbox,
                element_id=f"{source}#txt#{seq}",
            ))

    return text_table, figures
