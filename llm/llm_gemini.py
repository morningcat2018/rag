import time
from dotenv import load_dotenv
from google import genai
from log.log_config import logger

load_dotenv()
google_client = genai.Client()


def call(prompt):
    start = time.perf_counter()
    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    logger.info(f"gemini响应: {(time.perf_counter() - start):.4f} 秒")

    return response.text
