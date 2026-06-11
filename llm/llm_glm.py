import os
import time

from dotenv import load_dotenv
from zai import ZhipuAiClient

from log.log_config import logger

load_dotenv()
# 从环境变量读取 API Key
client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

MODULE_NAME = "glm-4.7"


def call(prompt):
    start = time.perf_counter()
    # Create chat completion
    response = client.chat.completions.create(
        model=MODULE_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # print(response.choices[0].message.content)
    logger.info(f"ZHIPU响应: {(time.perf_counter() - start):.4f} 秒")
    return response.choices[0].message.content
