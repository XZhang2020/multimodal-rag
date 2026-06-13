"""VLM 图生文：让"无字但有信息"的图（架构图/流程图/趋势图）也能进知识库。

OCR 只能抓图里的文字，抓不到图的语义。架构图画了什么模块、趋势图在涨还是跌，
必须靠多模态大模型描述。

端点独立于文本 LLM：默认走本地 ollama qwen3-vl:4b——
免费、不受公共 API "免费 key 不支持多模态" 的限制。文本任务仍走原 OpenAI 端点。

降级链：VLM 不可用 / 超时 / 报错 → 返回 ""，由调用方退回"仅 OCR / 占位"。
结果按图片字节哈希缓存，重复 ingest 不重复算。
"""
from __future__ import annotations

import base64

from openai import OpenAI

from src.config import settings
from .cache import ParseCache

_CACHE = ParseCache(enabled=settings.enable_parse_cache)
_CLIENT = None

_VLM_PROMPT = (
    "你是文档图片理解助手。请用简体中文客观描述这张图片，便于后续检索：\n"
    "1) 一句话说明图片类型（架构图/流程图/统计图表/示意图/照片/截图等）；\n"
    "2) 提炼图中的关键实体、模块、流程步骤或数据趋势；\n"
    "3) 若是图表，说明坐标轴/对比维度与主要结论。\n"
    "只输出描述本身，不要前缀、不要寒暄，控制在 200 字内。"
)


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=settings.vlm_api_key,
            base_url=settings.vlm_base_url,
        )
    return _CLIENT


def caption_image(image_bytes: bytes) -> str:
    """对图片字节生成中文语义描述。不可用时返回 ""，绝不抛异常。"""
    if not settings.enable_vlm_caption:
        return ""
    if not image_bytes:
        return ""

    cached = _CACHE.get(image_bytes, namespace="vlm")
    if cached is not None:
        return cached

    # 短期负缓存：上次失败且仍在 TTL 内 → 直接跳过，不再打本地模型
    if _CACHE.is_fresh(image_bytes, namespace="vlm_fail", ttl=settings.vlm_fail_ttl):
        return ""

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # qwen3-vl 思考通道约耗 600-700 token，正文要在其后输出，
    # 故 max_tokens 必须给足（实测 512 会 finish=length 且正文为空）。
    prompt = ("/no_think " + _VLM_PROMPT) if settings.vlm_no_think else _VLM_PROMPT
    try:
        resp = _client().chat.completions.create(
            model=settings.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.0,
            max_tokens=settings.vlm_max_tokens,
            timeout=settings.vlm_timeout,
        )
        caption = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[VLM] 图生文失败，降级为仅 OCR: {e!r}")
        # 打失败标记，TTL 内不再重试这张图（避免每次 ingest 都卡同一张崩溃图）
        _CACHE.set(image_bytes, namespace="vlm_fail", value="failed")
        return ""

    if caption:
        _CACHE.set(image_bytes, namespace="vlm", value=caption)
    return caption
