import time
from typing import List

from pymilvus import MilvusClient

from log.log_config import logger
from tools.data import DEFAULT_COLLECTION_NAME

start = time.perf_counter()
client = MilvusClient("./db/my_milvus_demo.db")
logger.info(f"加载milvus: {(time.perf_counter() - start):.4f} 秒")


def save_embeddings(chunks: List[str],
                    embeddings: List[List[float]],
                    collection_name=DEFAULT_COLLECTION_NAME,
                    dimension=768) -> None:
    """
    存入向量和文本到 Milvus
    :param chunks: 文本块列表
    :param embeddings: 对应的向量列表
    :param collection_name: 集合名称
    :param dimension: 向量维度
    """
    # 检查集合是否已存在
    if client.has_collection(collection_name):
        logger.info(f"集合 {collection_name} 已存在，跳过创建")
    else:
        # 定义 schema（支持文本字段）
        from pymilvus import DataType

        schema = MilvusClient.create_schema(
            auto_id=False,
            metric_type="COSINE",
            enable_dynamic_field=True  # 允许动态字段，这样就不需要预定义 text 字段了
        )

        # 添加主键字段
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 添加向量字段
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

        # 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema
        )

        # # 使用 prepare_index_params 创建索引参数
        # index_params = client.prepare_index_params()
        # index_params.add_index(
        #     field_name="vector",
        #     index_type="IVF_FLAT",  # 索引参数不合适：IVF_FLAT 索引不适合小数据集
        #     metric_type="COSINE",
        #     params={"nlist": 128}
        # )
        # # 创建索引
        # client.create_index(collection_name, index_params)
        logger.info(f"创建集合 {collection_name} 成功")

    # 准备插入数据
    data = [
        {"id": i, "vector": embeddings[i], "text": chunks[i]}
        for i in range(len(chunks))
    ]

    # 插入数据
    try:
        result = client.insert(
            collection_name=collection_name,
            data=data
        )
        logger.info(f"成功插入 {len(result['ids'])} 条数据")
    except Exception as e:
        logger.error(f"插入失败: {e}")
        # 如果失败，尝试分批插入
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = client.insert(collection_name=collection_name, data=batch)
            logger.info(f"成功插入批次 {i // batch_size + 1}: {len(result['ids'])} 条")


def select_embeddings(query_embedding,
                      top_k: int,
                      collection_name=DEFAULT_COLLECTION_NAME) \
        -> List[str]:
    """
    根据目标向量{query_embedding}查询向量数据库中的相似向量,查询最近的{top_k}条向量数据
    :param collection_name:
    :param query_embedding: 目标向量
    :param top_k: 查询返回条数
    :return: 相似度最近的{top_k}条向量数据
    """
    # 1. 加载集合到内存（关键步骤！）
    try:
        # 检查集合状态
        collection_info = client.describe_collection(collection_name)
        logger.debug(collection_info)

        # 如果未加载，则加载
        if not client.has_collection(collection_name):
            raise ValueError(f"集合 {collection_name} 不存在")

        # 加载集合（只有加载后才能搜索）
        client.load_collection(collection_name)
        logger.info(f"集合 {collection_name} 已加载到内存")

    except Exception as e:
        logger.error(f"加载集合失败: {e}")
        return []

    # 2. 执行搜索
    res = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,  # 返回最相似的k条
        output_fields=["text"]  # 同时把原文也返回给我
    )
    # print(res)
    # 3. 处理结果
    if not res or not res[0]:
        logger.info("查询无结果")
        return []
    for i in res[0]:
        logger.info(f"查询结果: {i['entity']['text'][:100]}...")
        logger.info(f"相似度得分: {i['distance']}")
    # return [item['entity']['text'] for item in res[0]]
    return [i['entity']['text'] for i in res[0]]
