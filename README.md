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

## qdrant

下载: https://github.com/qdrant/qdrant/releases

启动: qdrant --storage-path ~/.local/share/qdrant

访问测试：http://localhost:6333

数据目录 默认在：当前目录下的 storage/

可指定： qdrant --storage-path /自定义路径

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