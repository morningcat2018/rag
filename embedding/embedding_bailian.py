import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
MODULE_NAME = "text-embedding-v4"

"""
text-embedding-v4 属于Qwen3-Embedding系列

向量维度 2,048、1,536、1,024（默认）、768、512、256、128、64
"""


def embed_chunk_list(chunks: List[str]) -> List[List[float]] | List[float]:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下是北京地域base-url，如果使用新加坡地域的模型，需要将base_url替换为：https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.embeddings.create(
        model=MODULE_NAME,
        dimensions=768,
        input=chunks
    )

    # print(completion.model_dump_json())
    data = completion.data
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
    print(f"嵌入向量数量: {len(embedding)}, 向量维度: {len(embedding[0])}")
