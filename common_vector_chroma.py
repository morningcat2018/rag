import chromadb
import time
from typing import List
from log_config import logger

start = time.perf_counter()
chromadb_client = chromadb.PersistentClient("./chroma_3")
chromadb_collection = chromadb_client.get_or_create_collection(name="default")
logger.info(f"加载chromadb: {(time.perf_counter() - start):.4f} 秒")


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    存入
    :param chunks:
    :param embeddings:
    :return:
    """
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )
        logger.debug(f"存储片段 {i}: {chunk[:10]}...")


def select_embeddings(query_embedding, top_k: int) -> List[str]:
    """
    获取
    :param query_embedding:
    :param top_k:
    :return:
    """
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]
