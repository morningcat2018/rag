import time
from typing import List
from common_embedding import embed_chunk
from common_vector_qdrant import select_embeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from google import genai

load_dotenv()
google_client = genai.Client()


def retrieve(query: str, top_k: int) -> List[str]:
    """
    召回
    :param query:
    :param top_k:
    :return:
    """
    query_embedding = embed_chunk(query)
    return select_embeddings(query_embedding, top_k)


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    """
    重排
    :param query:
    :param retrieved_chunks:
    :param top_k:
    :return:
    """
    start = time.perf_counter()
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    print(f"加载cross_encoder: {(time.perf_counter() - start):.4f} 秒")
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


def generate(query: str, chunks: List[str]) -> str:
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

    # print(f"{prompt}\n\n---\n")

    start = time.perf_counter()
    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print(f"gemini响应: {(time.perf_counter() - start):.4f} 秒")

    return response.text


if __name__ == "__main__":
    query = "哆啦A梦使用的3个秘密道具分别是什么？"
    # query = "宝玉初见黛玉的描写"

    retrieved_chunks = retrieve(query, 10)
    # for i, chunk in enumerate(retrieved_chunks):
    #     print(f"[{i}] {chunk}\n")
    reranked_chunks = rerank(query, retrieved_chunks, 3)
    # for i, chunk in enumerate(reranked_chunks):
    #     print(f"[{i}] {chunk}\n")
    answer = generate(query, reranked_chunks)
    print(answer)
