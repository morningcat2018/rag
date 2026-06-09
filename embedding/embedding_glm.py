import os
from typing import List

from dotenv import load_dotenv
from zai import ZhipuAiClient

from log.log_config import logger

load_dotenv()
MODULE_NAME = "embedding-3"


def embed_chunk_list(chunks: List[str]) -> List[List[float]] | List[float]:
    client = ZhipuAiClient(api_key=os.environ.get('ZHIPU_API_KEY'))
    """
    维度选项：
    2048维（默认）：最高精度，适合对准确性要求极高的场景
    1024维：高精度与效率的平衡，适合大多数应用场景
    512维：中等精度，适合大规模部署的场景
    256维：较高效率，适合实时性要求高的场景
    """
    response = client.embeddings.create(
        model=MODULE_NAME,  # 填写需要调用的模型编码
        input=chunks,
        dimensions=768  # 没有768
    )
    # print(response)
    data = response.data
    return [i.embedding for i in data]


def embed_chunk(chunk: str) -> list[float]:
    return embed_chunk_list([chunk])[0]


if __name__ == "__main__":
    input = [
        "美食非常美味，服务员也很友好。",
        "这部电影既刺激又令人兴奋。",
        "阅读书籍是扩展知识的好方法。"
    ]
    embedding = embed_chunk_list(input)
    logger.info(f"嵌入向量数量: {len(embedding)}, 向量: {embedding}")
