# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

中文场景的多模态 RAG 原型：父子分块 + 混合检索（BM25 + 向量）+ Cross-Encoder 精排 + LLM 生成。所有用户问答和文档都假设是中文为主。

## 常用命令

```bash
# 一次性：构建知识库（解析 data/ 下文件 → 切块 → 灌入 Qdrant + 生成 bm25_index.pkl）
python ingest.py

# 启动交互式问答（必须在真实终端，不能在 Claude Code 的 ! shell 里跑，input() 会读到 EOF）
python main.py

# 安装依赖
pip install -r requirements.txt
```

前置要求：

- 本地必须有运行中的 Qdrant，监听 `localhost:6333`（持久化目录是仓库内的 `qdrant_storage/`，由本地 Qdrant 容器/进程挂载）。
- `.env` 至少要配 `OPENAI_API_KEY`、`OPENAI_BASE_URL`。模板见 `.env.example`。
- 国内网络下载 HuggingFace 模型（bge-m3 / bge-reranker-base）前先 `export HF_ENDPOINT=https://hf-mirror.com`。

仓库目前没有测试套件，也没有 lint 配置，不要假装有。

## 架构要点（多文件协同的部分）

### 父子分块策略（`src/chunker.py` + `src/vector_store.py` + `src/retriever.py`）

文档先切成 ~2000 字父块，再在每个父块内切 ~400 字子块。**检索只用子块**（粒度小、向量更聚焦），但**喂给 LLM 的是父块**（上下文完整）。子块通过 `metadata.parent_id` 反查父块。

### 单 collection 装父子两类点

Qdrant 里只有一个 collection (`multimodal_rag`)，父块和子块共存，靠 payload 里的 `doc_type` 字段区分：

- 父块：`vector=[0.0]*1024` 占位空向量，仅用于 ID 检索（`get_parent_documents`）。
- 子块：真实 embedding，参与向量召回。
- `VectorStore.search` **必须**在服务端用 `Filter(doc_type=child)` 过滤，否则父块零向量会挤占 top_k。
- Point ID 一律用 UUID 字符串，不要用自增整数 —— 早期 bug 是子块整数 ID 覆盖父块。

### 混合检索 → 精排 → 父块展开（`Retriever.retrieve`）

1. `bm25_search` + `vector_store.search` 各取 top_k，按 `id` 去重合并。
2. CrossEncoder 重排，留 `rerank_top_k` 个子块。
3. 取这些子块的 `parent_id` 集合，去 Qdrant 拉父块。
4. 父块按"其下子块的最高 rerank 分"排序 —— 聚合用 `max`，不能写成 dict last-write-wins。

### BM25 索引的两段式生命周期（`src/retriever.py` + `ingest.py`）

- `bm25_index.pkl` **只存** `all_child_docs` 列表，不存 `BM25Okapi` 实例。
- `Retriever.__init__` 启动时若 pkl 存在，自动加载文档并**重建** BM25 实例。`main.py` 不需要再手动调 `load_bm25_index`。
- 中文分词必须走 jieba（`_tokenize`），不能用 `query.split()` —— 后者会把整段中文当成一个 token，BM25 等于失效。

### VectorStore 生命周期

- `VectorStore.__init__` 只确保 collection 存在，**不会**删数据。
- 清库重建数据是 `reset_collection()` 的职责，**只在 `ingest.py` 开头调用一次**。新增任何"启动时清表"的逻辑都是退化。
- `Retriever` 接受外部 `vector_store` 参数；`ingest.py` 中要复用同一个实例，不要让 `Retriever()` 再造一个。

### 其它易踩坑

- Windows 控制台默认 GBK，**所有 `print` 不要用 emoji**（`✅` `🚀` `💡` 都会触发 `UnicodeEncodeError`），用 `[OK]` 之类 ASCII 替代。
- 配置全部经过 `pydantic-settings`（`src/config.py`），改默认值要同步检查 `.env.example`。
- `Embedder` 用 `BAAI/bge-m3`，维度 1024，与 `settings.embedding_dim` 必须一致；换模型要同步改维度，否则 collection schema 不匹配。
