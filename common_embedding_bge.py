import time
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from log_config import logger

EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh"

start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
logger.info(f"加载embedding_model: {(time.perf_counter() - start):.4f} 秒")


def embed_chunk_list(chunks: List[str]) -> List[List[float]] | List[float]:
    """
    片段文本向量化
    :param chunks:
    :return:
    """
    inputs = tokenizer(
        chunks,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS 向量

    # 归一化（非常重要）
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    logger.debug(f"嵌入向量形状: {embeddings.shape}")  # (x, 768)
    return embeddings.squeeze(0).tolist()


def embed_chunk(chunk: str) -> list[float]:
    return embed_chunk_list([chunk])


if __name__ == "__main__":
    embedding = embed_chunk_list(["红楼梦", "西游记"])
    logger.info(f"嵌入向量数量: {len(embedding)}, 向量: {embedding}")
