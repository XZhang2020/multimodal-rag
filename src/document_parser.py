import os
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter

class DocumentParser:
    def __init__(self):
        # 极简初始化，无任何报错
        self.converter = DocumentConverter()

    def parse(self, file_path: str):
        """统一解析入口"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        else:
            # 其他格式交给 docling
            conv = self.converter.convert(file_path)
            return self._to_docs(conv.document)

    def _parse_pdf(self, path):
        """只做文本提取，不做复杂OCR！"""
        docs = []
        doc = fitz.open(path)

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                text = "[无法提取文本：可能是扫描件]"

            docs.append({
                "content": text,
                "metadata": {
                    "source": os.path.basename(path),
                    "page": page_num + 1,
                    "type": "text"
                }
            })
        return docs

    def _to_docs(self, docling_doc):
        """轻量转格式"""
        docs = []
        for el in docling_doc.iterate_elements():
            if el.text.strip():
                docs.append({
                    "content": el.text,
                    "metadata": {
                        "source": docling_doc.path.name,
                        "page": el.page_no if hasattr(el, "page_no") else 0,
                        "type": "text"
                    }
                })
        return docs