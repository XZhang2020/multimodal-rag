"""PaddleOCR 后端：中文扫描页 / 图内文字识别。

为什么是 PaddleOCR 而不是 docling 默认的 EasyOCR：
中文场景 PP-OCRv5 的检测+识别精度明显更高。这是主动的引擎替换，不是默认值。

两个工程坑（已实测）：
1. PaddleOCR 3.x 必须装底层 paddlepaddle，否则 import 成功但推理崩。
2. paddlepaddle CPU 版默认开 oneDNN，会触发算子不兼容崩溃 →
   初始化必须 enable_mkldnn=False。
3.x 的 predict() 返回 list[dict]，文字在 rec_texts，坐标在 rec_boxes。
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

from src.config import settings

_OCR = None
_OCR_FAILED = False  # 初始化失败后不再反复重试


def _engine():
    global _OCR, _OCR_FAILED
    if _OCR is not None or _OCR_FAILED:
        return _OCR
    try:
        import os
        os.environ.setdefault("GLOG_minloglevel", "3")
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(
            lang=settings.paddle_ocr_lang,
            use_textline_orientation=True,
            enable_mkldnn=False,   # CPU 版 oneDNN 崩溃的绕过
        )
    except Exception as e:
        _OCR_FAILED = True
        print(f"[OCR] PaddleOCR 初始化失败，扫描件/图内文字将降级跳过: {e!r}")
    return _OCR


def _to_array(image) -> Optional[np.ndarray]:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (bytes, bytearray)):
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(image)).convert("RGB"))
    # PIL.Image
    try:
        return np.array(image.convert("RGB"))
    except Exception:
        return None


def _reading_order(texts: list[str], boxes) -> str:
    """按版面阅读顺序（自上而下、同行自左而右）拼接，避免乱序。"""
    if boxes is None or len(boxes) != len(texts):
        return "\n".join(t for t in texts if t)
    items = []
    for t, b in zip(texts, boxes):
        if not t:
            continue
        arr = np.asarray(b).reshape(-1, 2)
        x = float(arr[:, 0].min())
        y = float(arr[:, 1].min())
        items.append((y, x, t))
    # 行容差：y 相近视为同一行
    items.sort(key=lambda it: (round(it[0] / 12), it[1]))
    return "\n".join(it[2] for it in items)


def ocr_image(image) -> str:
    """对一张图（np.ndarray / bytes / PIL.Image）做 OCR，返回阅读序文本。

    引擎不可用或识别为空时返回 ""，由调用方决定占位策略——绝不抛异常打断主流程。
    """
    eng = _engine()
    if eng is None:
        return ""
    arr = _to_array(image)
    if arr is None:
        return ""
    try:
        result = eng.predict(arr)
    except Exception as e:
        print(f"[OCR] 识别失败，跳过: {e!r}")
        return ""
    if not result:
        return ""
    r0 = result[0]
    if not isinstance(r0, dict):
        return ""
    texts = r0.get("rec_texts") or []
    boxes = r0.get("rec_boxes")
    if boxes is not None and not isinstance(boxes, list):
        boxes = list(boxes)
    return _reading_order(list(texts), boxes).strip()


def is_available() -> bool:
    return _engine() is not None
