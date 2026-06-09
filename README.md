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

uv add openai
uv add zai-sdk # 智谱AI 开放平台
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

## 向量数据库 qdrant

https://qdrant.org.cn/documentation/quickstart/

下载: https://github.com/qdrant/qdrant/releases

config.yaml
```commandline
storage:
  storage_path: ~/.local/share/qdrant
  snapshots_path: ~/.local/share/qdrant/snapshots

service:
  host: 127.0.0.1
  http_port: 6333
  grpc_port: 6334
```

cd software

启动: ./qdrant --config-path ./config.yaml

访问测试：http://localhost:6333

数据目录 默认在：当前目录下的 storage/

### curl 访问

创建 Collection

```curl
curl -X PUT "http://localhost:6333/collections/my_collection" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
}'
```

插入数据

```curl
curl -X PUT "http://localhost:6333/collections/my_collection/points" \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.01, 0.02, ..., 0.99],
        "payload": {
          "text": "这是第一段文本",
          "source": "doc1"
        }
      }
    ]
}'
```

向量搜索

```curl
curl -X POST "http://localhost:6333/collections/my_collection/points/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.01, 0.02, ..., 0.99],
    "limit": 3
}'
```

```curl
curl -X DELETE http://localhost:6333/collections/my_collection
```

```curl
curl http://localhost:6333/collections
```


### Python 使用（推荐方式）

> pip install qdrant-client

or

> uv add qdrant-client

```py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient("localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# 插入数据
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.01]*768,
            payload={"text": "测试文本"}
        )
    ],
)

# 查询
hits = client.search(
    collection_name="my_collection",
    query_vector=[0.01]*768,
    limit=3,
)

print(hits)
```

## embedding model

1. shibing624/text2vec-base-chinese

https://huggingface.co/shibing624/text2vec-base-chinese

https://github.com/shibing624/text2vec

本机缓存位置 ~/.cache/huggingface/hub

- 📌 模型类型： 基于 CoSENT（Cosine Sentence）训练的方法，上层为一个 Transformer 编码器，底层使用 pooling 得到句子向量。
- 🧠 基础结构： 内部使用 hfl/chinese-macbert-base 预训练模型作为词表示基础，再通过对比学习（contrastive learning）方式 fine-tune。
- 📊 输出向量： 把句子映射到 768 维的密集向量
- 基于 MacBERT-base 架构（12层 Transformer，768 hidden）
- 采用 CoSENT 训练方式，专门优化语义相似度
- 在中文 STS / 相似度任务上表现稳定

2. BGE-base-zh

BAAI BGE（Beijing General Embedding）系列是高质量中英文向量模型。
其中： BAAI / Hugging Face 上的 bge-base-zh 是一个 中文 embedding 模型（768 维）

https://huggingface.co/BAAI/bge-base-zh

https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md


