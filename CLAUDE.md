# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述
这是一个检索增强生成（RAG）系统项目，专注于处理中国古典小说《红楼梦》。系统从文本片段创建嵌入向量，将其存储在 ChromaDB 中，并提供检索-生成的问答流水线。

## 项目结构
```
.
├── 红楼梦.txt                 # 源文档（中国古典小说）
├── common.py                  # 共享的嵌入模型功能
├── save_step.py              # 创建嵌入向量并存储到 ChromaDB
├── generate_step.py          # RAG 流水线：检索 → 重排 → 生成
├── chroma_hongloumeng/       # ChromaDB 持久化存储
├── chroma.db/                # 备用 ChromaDB 存储
├── pyproject.toml            # Poetry 依赖管理
├── .env                      # 环境变量（GEMINI_API_KEY）
└── main.ipynb               # 实验用的 Jupyter 笔记本
```

## 依赖项
项目使用 Poetry 进行依赖管理。主要依赖包括：
- `chromadb>=1.5.0`：嵌入向量的向量数据库
- `google-genai>=1.64.0`：Google Gemini API 客户端
- `sentence-transformers==2.6.1`：嵌入模型
- `torch==2.2.0`：深度学习框架
- `python-dotenv>=1.2.1`：环境变量管理

## 核心组件

### 1. 嵌入模型（`common.py`）
- 使用 `shibing624/text2vec-base-chinese` 模型进行中文文本嵌入
- 提供 `embed_chunk()` 函数将文本转换为向量

### 2. 数据处理（`save_step.py`）
- `split_into_chunks()`：通过双换行符分割文档
- `save_embeddings()`：将文档片段和嵌入向量存储到 ChromaDB
- `save_step()`：处理文档并创建向量存储的主函数

### 3. RAG 流水线（`generate_step.py`）
- `retrieve()`：使用向量相似度从 ChromaDB 初始检索
- `rerank()`：使用交叉编码器重新排序以提高相关性
- `generate()`：使用 Google Gemini API 生成答案

## 使用命令

### 环境设置
```bash
# 使用 Poetry 安装依赖
poetry install

# 激活虚拟环境
poetry shell

# 或使用现有的 .venv
source .venv/bin/activate
```

### 数据处理
```bash
# 处理文档并创建嵌入向量
python save_step.py

# 这会将 红楼梦.txt 分割成片段，并将嵌入向量存储到 chroma_hongloumeng/
```

### 查询
```bash
# 在问题上运行 RAG 流水线
python generate_step.py

# 默认查询："两弯似蹙非蹙罥烟眉是指谁?"
# 修改 generate_step.py:77 中的查询来提问不同问题
```

### 开发
```bash
# 检查 Python 版本（需要 3.11+）
python3 --version

# 运行测试（如果有测试文件）
poetry run pytest

# 安装新依赖
poetry add 包名
```

## 环境设置
`.env` 文件必须包含您的 Google Gemini API 密钥：
```
GEMINI_API_KEY=您的_api_密钥_在这里
```

## 重要提示
- 系统专门为中文文本处理设计
- ChromaDB 使用 `chroma_hongloumeng/` 目录进行持久化存储
- 嵌入模型和交叉编码器按需加载，可能需要较长时间
- Google Gemini API 的使用可能根据您的计划产生费用