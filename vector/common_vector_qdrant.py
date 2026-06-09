from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from log_config import logger

# import httpx
#
# r = httpx.get("http://127.0.0.1:6333")
# logger.debug(f"HTTP状态码: {r.status_code}")
# logger.debug(f"HTTP响应: {r.text}")

QDRANT_COLLECTION_NAME = "my_collection_hong3"
BATCH_SIZE = 100

qdrant_client = QdrantClient(
    url="http://127.0.0.1:6334",
    timeout=10,
    prefer_grpc=True,  # 🔥 改用 gRPC，绕过 httpx
)


def ensure_collection_exists():
    collections = [c.name for c in qdrant_client.get_collections().collections]

    if QDRANT_COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            ),
        )


ensure_collection_exists()


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    存入
    :param chunks:
    :param embeddings:
    :return:
    """
    points = [PointStruct(id=i, vector=embedding, payload={"text": chunk})
              for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
    # 分批插入数据
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        operation_info = qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True,
            points=batch,
        )
        logger.info(f"存储批次 {i}: {operation_info}")


def select_embeddings(query_embedding, top_k: int) -> List[str]:
    """
    根据目标向量{query_embedding}查询向量数据库中的相似向量,查询最近的{top_k}条向量数据
    :param query_embedding: 目标向量
    :param top_k: 查询返回条数
    :return: 相似度最近的{top_k}条向量数据
    """
    search_result = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=top_k
    ).points

    logger.debug(search_result)
    res = [i.payload['text'] for i in search_result]
    return res
