import logging
import sys
import os
from typing import Optional


def setup_logger(name: str = "rag_app", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    配置并返回一个日志记录器

    :param name: 日志记录器名称
    :param level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_file: 日志文件路径，如果为None则只输出到控制台
    :return: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 默认日志记录器 - 同时输出到控制台和项目目录下的日志文件
logger = setup_logger(log_file="logs/rag_app.log")