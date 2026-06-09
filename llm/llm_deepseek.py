import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from log_config import logger

load_dotenv()

MODULE_NAME = "deepseek-v4-pro"


def call(prompt):
    start = time.perf_counter()
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=MODULE_NAME,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    # print(response.choices[0].message.content)
    logger.info(f"deepseek响应: {(time.perf_counter() - start):.4f} 秒")
    return response.choices[0].message.content
