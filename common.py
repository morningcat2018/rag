import time
from typing import List
from sentence_transformers import SentenceTransformer

start = time.perf_counter()
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
print(f"加载embedding_model: {(time.perf_counter() - start):.4f} 秒")


def embed_chunk(chunk: str) -> List[float]:
    """
    片段文本向量化
    :param chunk:
    :return:
    """
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()
