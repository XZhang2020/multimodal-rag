"""统一文档解析入口（多模态）。

历史上这里只做 PDF 纯文本抽取（扫描件丢占位串、图表全丢）。现在转调
src.parsing 分级管线：docling 版面/表格 + PaddleOCR 中文 + VLM 图生文。
返回契约不变：list[{"content", "metadata"}]，下游零改动。
"""
from src.parsing import parse_to_documents


class DocumentParser:
    def __init__(self):
        # 解析管线内部按需懒加载 docling/PaddleOCR/VLM，这里无需预热
        pass

    def parse(self, file_path: str, progress=None):
        """统一解析入口：返回 [{'content', 'metadata'}]。

        progress: 可选回调 progress(stage, done, total)，上报图片/表格处理进度。
        """
        return parse_to_documents(file_path, progress=progress)
