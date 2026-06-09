import time
from typing import List
from sentence_transformers import CrossEncoder
from log.log_config import logger


def retrieve(query: str, top_k: int, select_embeddings, embed_chunk) -> List[str]:
    """
    召回
    :param query:
    :param top_k:
    :param select_embeddings:
    :param embed_chunk:
    :return:
    """
    query_embedding = embed_chunk(query)
    retrieved_chunks = select_embeddings(query_embedding, top_k)
    for i, chunk in enumerate(retrieved_chunks):
        logger.info(f"召回 --- [{i}] {chunk}")
    return retrieved_chunks


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    """
    重排
    :param query:
    :param retrieved_chunks:
    :param top_k:
    :return:
    """
    start = time.perf_counter()
    # 重排序模型
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    logger.info(f"加载cross_encoder: {(time.perf_counter() - start):.4f} 秒")
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    reranked_chunks = [chunk for chunk, _ in scored_chunks][:top_k]
    for i, chunk in enumerate(reranked_chunks):
        logger.info(f"重排 --- [{i}] {chunk}")
    return reranked_chunks


def generate(query: str, chunks: List[str], llm_call) -> str:
    """
    生成
    :param query:
    :param chunks:
    :return:
    """
    chunks_text = "\n\n".join(chunks)
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:{chunks_text}

请基于上述内容作答，不要编造信息。"""

    logger.debug(f"生成提示词:\n{prompt}\n\n---\n")
    return llm_call(prompt)
