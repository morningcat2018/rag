内容整理自 <https://www.bilibili.com/video/BV1wc3izUEUb>
```
本机环境:macosx_26_0_x86_64

所以不能使用torch>2.2.0版本
onnxruntime需要使用1.15.0版本
python需要使用python3.11
numpy只能使用1.x版本
sentence-transformers使用2.6.1版本
```

执行步骤
```
. /opt/anaconda3/bin/activate && conda activate /opt/anaconda3/envs/rag;
# 此conda环境下时 python3.11
uv init .

uv add "numpy<2"
uv add torch==2.2.0
uv add onnxruntime==1.15.0
uv add sentence_transformers chromadb google-genai python-dotenv

uv remove sentence-transformers
uv add sentence-transformers==2.6.1

uv run --with jupyter jupyter lab
```

需要在[google aistudio](https://aistudio.google.com/api-keys)申请API key;
并在项目目录下创建 .env 文件,内容为
```
GEMINI_API_KEY=此处填写申请的key值
```

pyproject.toml
```toml
[project]
name = "rag"
version = "0.1.0"
description = "构建RAG系统"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=1.5.0",
    "google-genai>=1.64.0",
    "numpy<2",
    "onnxruntime==1.15.0",
    "python-dotenv>=1.2.1",
    "sentence-transformers==2.6.1",
    "torch==2.2.0",
]
```