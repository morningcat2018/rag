from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# import httpx
#
# r = httpx.get("http://127.0.0.1:6333")
# print(r.status_code)
# print(r.text)

QDRANT_COLLECTION_NAME = "my_collection"

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
    operation_info = qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        wait=True,
        points=points,
    )
    print(operation_info)


def select_embeddings(query_embedding, top_k: int) -> List[str]:
    """
    获取
    :param query_embedding:
    :param top_k:
    :return:
    """
    search_result = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=top_k
    ).points

    # print(search_result)
    res = [i.payload['text'] for i in search_result]
    return res
