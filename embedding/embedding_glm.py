import os
from typing import List

from dotenv import load_dotenv
from zai import ZhipuAiClient

from log.log_config import logger

load_dotenv()
MODULE_NAME = "embedding-3"


def embed_chunk_list_bak(chunks: List[str]) -> List[List[float]] | List[float]:
    client = ZhipuAiClient(api_key=os.environ.get('ZHIPU_API_KEY'))
    response = client.embeddings.create(
        model=MODULE_NAME,  # 填写需要调用的模型编码
        input=chunks,
        dimensions=1024  # 没有768
    )
    # print(response)
    data = response.data
    return [i.embedding for i in data]

    """
    维度选项：
    2048维（默认）：最高精度，适合对准确性要求极高的场景
    1024维：高精度与效率的平衡，适合大多数应用场景
    512维：中等精度，适合大规模部署的场景
    256维：较高效率，适合实时性要求高的场景
    """


def embed_chunk_list(chunks: List[str], dimensions=1024,
                     batch_size: int = 64) -> List[List[float]]:
    """
    将文本列表分批生成 Embedding 向量。

        :param batch_size: 每批最大数量（API 限制 <=64）
        :param chunks: 文本列表
        :param dimensions:
        :return 每个文本对应的向量列表
    """
    client = ZhipuAiClient(api_key=os.environ.get('ZHIPU_API_KEY'))
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=MODULE_NAME,
                input=batch,
                dimensions=dimensions  # 注意：模型 embedding-3 的默认维度是 1024，不支持手动指定为 768
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except Exception as e:
            logger.error(f"批次 {i // batch_size + 1} 请求失败: {e}")
            raise

    return all_embeddings


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
