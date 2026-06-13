"""解析路由：分级、成本感知的统一入口。

核心思想——不无脑全量 OCR/VLM，按需路由：
  非 PDF 电子文档        → docling 抽结构（不 OCR）
  PDF 文本页            → docling 抽结构（有文本层，不浪费 OCR）
  PDF 扫描页            → PyMuPDF 渲染 → PaddleOCR
  图片元素              → VLM caption + 图内 OCR 双通道
任一环节失败都降级，绝不让一个坏文件打断整批 ingest。
"""
from __future__ import annotations

import os

import fitz  # PyMuPDF

from src.config import settings
from .models import (
    Element,
    MODALITY_TEXT,
    MODALITY_OCR_TEXT,
    MODALITY_FIGURE,
)
from . import docling_backend, ocr_paddle, vision, tables

# docling 直接吃的电子文档格式
_DOCLING_EXTS = {".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv", ".adoc"}


def parse(file_path: str, progress=None) -> list[Element]:
    """解析文件为 Element 列表。

    progress: 可选回调 progress(stage: str, done: int, total: int)，
    用于上报图片/表格等慢操作进度。库层不直接 print，由调用方决定如何展示。
    """
    source = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        elements = _parse_pdf(file_path, source)
    elif ext in _DOCLING_EXTS:
        elements = _parse_docling_generic(file_path, source)
    else:
        # 兜底：当作可抽文本的文件，docling 尝试，失败则空
        elements = _parse_docling_generic(file_path, source)

    # 终态化：表格补 NL 摘要、图片补 caption+OCR
    return _enrich(elements, progress=progress)


# ── PDF：逐页判断文本覆盖率，分流 docling / OCR ──────────────────────────────
def _parse_pdf(path: str, source: str) -> list[Element]:
    scanned_pages = _detect_scanned_pages(path)

    # 文本部分交 docling（它内部不 OCR）；图片单独收集
    try:
        text_table, figures = docling_backend.parse_with_docling(path, source)
    except Exception as e:
        print(f"[PARSE] docling 解析 PDF 失败，退回 PyMuPDF 纯文本: {e!r}")
        text_table, figures = _pymupdf_fallback(path, source), []

    # 扫描页 docling 抽不到文本 → 单独渲染整页跑 PaddleOCR 补回
    if scanned_pages:
        text_table.extend(_ocr_scanned_pages(path, source, scanned_pages))

    return text_table + figures


def _detect_scanned_pages(path: str) -> set[int]:
    """文本字符数低于阈值的页判为扫描页（页码从 1 起，对齐 docling）。"""
    scanned = set()
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                if len(page.get_text().strip()) < settings.ocr_text_coverage_threshold:
                    scanned.add(i)
    except Exception as e:
        print(f"[PARSE] 扫描页检测失败，按全文本页处理: {e!r}")
    return scanned


def _ocr_scanned_pages(path: str, source: str, pages: set[int]) -> list[Element]:
    out: list[Element] = []
    if not ocr_paddle.is_available():
        for p in sorted(pages):
            out.append(Element(modality=MODALITY_OCR_TEXT,
                               text="[扫描页：OCR 不可用]", source=source, page=p,
                               element_id=f"{source}#ocr#{p}"))
        return out
    try:
        with fitz.open(path) as doc:
            for p in sorted(pages):
                page = doc[p - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 提升 OCR 精度
                text = ocr_paddle.ocr_image(pix.tobytes("png"))
                out.append(Element(
                    modality=MODALITY_OCR_TEXT,
                    text=text or "[扫描页：未识别出文字]",
                    source=source, page=p,
                    element_id=f"{source}#ocr#{p}",
                    extra={"ocr": True},
                ))
    except Exception as e:
        print(f"[PARSE] 扫描页 OCR 失败: {e!r}")
    return out


def _pymupdf_fallback(path: str, source: str) -> list[Element]:
    """docling 整体失败时的纯文本兜底，保证至少有文本可用。"""
    out: list[Element] = []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                txt = page.get_text().strip()
                if txt:
                    out.append(Element(modality=MODALITY_TEXT, text=txt,
                                       source=source, page=i,
                                       element_id=f"{source}#txt#{i}"))
    except Exception as e:
        print(f"[PARSE] PyMuPDF 兜底也失败: {e!r}")
    return out


# ── 非 PDF：docling 通吃 ─────────────────────────────────────────────────────
def _parse_docling_generic(path: str, source: str) -> list[Element]:
    try:
        text_table, figures = docling_backend.parse_with_docling(path, source)
        return text_table + figures
    except Exception as e:
        print(f"[PARSE] docling 解析失败({source}): {e!r}")
        return []


# ── 终态化：图片/表格的慢操作集中在这里 ─────────────────────────────────────
def _enrich(elements: list[Element], progress=None) -> list[Element]:
    figures = [e for e in elements if e.modality == MODALITY_FIGURE]
    tables_ = [e for e in elements if e.modality == "table"]
    n_fig, n_tab = len(figures), len(tables_)
    fig_done = tab_done = 0

    def _report(stage, done, total):
        if progress and total:
            try:
                progress(stage, done, total)
            except Exception:
                pass

    out: list[Element] = []
    for el in elements:
        if el.modality == MODALITY_FIGURE:
            fig_done += 1
            _report("figure", fig_done, n_fig)   # 处理前上报，慢操作不至于看着像卡死
            out.append(_enrich_figure(el))
        elif el.modality == "table":
            tab_done += 1
            _report("table", tab_done, n_tab)
            out.append(tables.summarize_table(el))
        else:
            out.append(el)
    # 丢掉彻底空的元素，避免噪声入库
    return [e for e in out if not e.is_empty()]


def _enrich_figure(el: Element) -> Element:
    img = el.extra.get("image_bytes", b"")
    caption_doc = el.extra.get("caption", "")     # 文档自带图注
    caption_vlm = vision.caption_image(img)        # VLM 语义描述
    ocr_text = ocr_paddle.ocr_image(img) if img else ""

    parts = []
    if caption_doc:
        parts.append(f"图注：{caption_doc}")
    if caption_vlm:
        parts.append(caption_vlm)
    if ocr_text:
        parts.append(f"图中文字：{ocr_text}")

    el.text = "\n".join(parts) if parts else "[图：无法识别内容]"
    el.extra["caption_vlm"] = caption_vlm
    el.extra["ocr"] = ocr_text
    el.extra.pop("image_bytes", None)  # 不把图字节带进下游 metadata
    return el
