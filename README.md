# multimodal-rag

中文场景的多模态 RAG 系统：**多模态解析**（docling 版面分析 + PaddleOCR 中文 OCR + 本地 VLM 图生文）→ 父子分块 → 混合检索（BM25 + 向量）→ Cross-Encoder 精排 → LLM 生成（带原文引用）。

> 核心理念：**检索质量的天花板 = 解析质量**。图片、表格、扫描页都先"翻译"成高质量、可溯源的文本，喂进统一的 `{content, metadata}` 契约，下游链路零改动。

## 特性

- **多模态解析（分级路由，成本/延迟感知）**
  - PDF 按文本覆盖率分流：有文本层的页直接抽（docling），扫描页才走 OCR。
  - 中文 OCR 用 **PaddleOCR**（PP-OCRv5），而非 docling 默认的 EasyOCR。
  - 图片**双通道**：本地 VLM（qwen3-vl）出语义描述 + PaddleOCR 出图内文字。
  - 表格**双存**：markdown 原表喂 LLM（精确读数）+ NL 摘要进向量库（语义召回）。
  - **降级链**：docling→PyMuPDF；VLM→仅 OCR；OCR 空→占位但留 bbox。绝不静默丢内容。
  - **缓存/负缓存**：OCR/VLM 结果按内容哈希缓存；VLM 失败写 TTL 负缓存，避免反复卡同一张图。
- **父子分块**：检索用子块（聚焦），喂 LLM 用父块（完整）。表格/图片为原子块，不被切碎。
- **混合检索 + 精排**：BM25（jieba 分词）抓精确词 + 向量（bge-m3）抓语义，CrossEncoder 重排。
- **多轮对话**：query 改写做指代消解 + 滑动窗口 history。
- **带证据的回答**：每条事实附「原文摘录」引用块，标注来源模态（正文/表格/图/扫描件）。

## 快速开始

```bash
conda create -n multimodal-rag
conda activate multimodal-rag
pip install -r requirements.txt

# 前置：本地起 Qdrant(:6333) 和 ollama(:11434, 已 pull qwen3-vl:4b)
# .env 配 OPENAI_API_KEY / OPENAI_BASE_URL(文本 LLM)，VLM 端点见 src/config.py

python ingest.py                 # 解析 data/ → 切块 → 灌库 + BM25 索引
python scripts/parse_smoke.py    # 冒烟：打印各模态元素分布
python main.py                   # 交互式问答(需真实终端)
```

架构详情见 [docs/architecture.md](docs/architecture.md)，开发约定与踩坑见 [CLAUDE.md](CLAUDE.md)。

## 支持的文件格式


| 格式            | 解析方式                                     |
| ------------- | ---------------------------------------- |
| PDF（文本/扫描/混合） | docling 版面 + PaddleOCR 扫描页兜底             |
| docx / pptx   | docling 版面分析                             |
| xlsx / csv    | docling 抽表 → 表格双存（**当前为语义检索，见 Roadmap**） |
| 图片（png/jpg…）  | VLM 图生文 + OCR                            |


## Roadmap

- **xlsx 走 Text-to-SQL**（结构化数据专用链路）
  - xlsx 直接用 pandas 读入、落地数据库（SQLite/DuckDB），保留结构化表而非压平成文本。
  - 用户问题经 LLM 转 SQL，直接查库做精确聚合/筛选/排序，结果再交 LLM 组织成自然语言。
  - **动机**：向量检索对"求和、按条件过滤、排序"等结构化查询天生弱；Text-to-SQL 对精确数值问答准得多。这是与当前 Text-to-RAG 并行、按文件/问题类型路由的独立链路。
- 引用块强制标注来源模态（当前 LLM 偶尔省略 `· 图` 标签）。
- VLM 失败图的离线补偿（环境恢复后重跑负缓存中已过期的失败项）。

## 已知限制

- 本地 VLM 在 CPU 上每张图约 40-90s；**勿背靠背连跑两次 `ingest.py`**（VLM+OCR+embedding 内存叠加易触发 OOM，而 ingest 开头会清库）。
- `qwen3-vl` 的思考模式当前无法关闭（`/no_think`、`think=False`、原生 `think:false` 实测均无效），故 `vlm_max_tokens` 须 ≥1500。
- 免费文本 API 限制 prompt < 4096 token，故 `max_context_chars` 默认压到 2800。

