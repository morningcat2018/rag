import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from log.log_config import logger

load_dotenv()
# 从环境变量读取 API Key
client = OpenAI(
    api_key=os.environ.get('BAILIAN_API_KEY'),
    base_url="https://ws-9k3ei565104nquu4.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

"""
ws-9k3ei565104nquu4.cn-beijing.maas.aliyuncs.com
https://ws-9k3ei565104nquu4.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
https://ws-9k3ei565104nquu4.cn-beijing.maas.aliyuncs.com/api/v1
"""

MODULE_NAME = "qwen3.6-flash-2026-04-16"


def call(prompt):
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODULE_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # print(response.choices[0].message.content)
    logger.info(f"LLM响应: {(time.perf_counter() - start):.4f} 秒")
    return response.choices[0].message.content


if __name__ == "__main__":
    print(call("你是谁"))
