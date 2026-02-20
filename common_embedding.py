import time
from typing import List
from sentence_transformers import SentenceTransformer
from log_config import logger

embedding_model = None
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"


def embed_chunk(chunk: str) -> List[float]:
    """
    片段文本向量化
    :param chunk:
    :return:
    """
    global embedding_model
    if embedding_model is None:
        start = time.perf_counter()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"加载embedding_model: {(time.perf_counter() - start):.4f} 秒")

    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


if __name__ == "__main__":
    embedding = embed_chunk("红楼梦")
    logger.info(f"嵌入向量维度: {len(embedding)}, 向量: {embedding}")
