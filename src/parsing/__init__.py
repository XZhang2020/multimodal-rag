"""多模态解析层。

对外只暴露 parse_to_documents：输入文件路径，输出下游统一契约
list[{"content", "metadata"}]。内部分级路由 docling / PaddleOCR / VLM。
"""
from .router import parse as parse_elements
from .assembler import to_documents
from .models import Element


def parse_to_documents(file_path: str, progress=None) -> list[dict]:
    return to_documents(parse_elements(file_path, progress=progress))


__all__ = ["parse_to_documents", "parse_elements", "to_documents", "Element"]
