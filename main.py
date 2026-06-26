from tools.generate_step import retrieve, rerank, generate
from tools.save_step import split_into_chunks, split_into_chunks_simple
from embedding.embedding_glm import embed_chunk_list, embed_chunk
from vector.common_vector_milvus import select_embeddings, save_embeddings
from llm.llm_bailian import call
from log.log_config import logger
from tools.data import DEFAULT_COLLECTION_NAME

DIMENSION = 1024


def save(doc_name, split_func=split_into_chunks,
         collection_name=DEFAULT_COLLECTION_NAME):
    """
    系统初始化时执行一遍即可
    :return:
    """
    chunks = split_func(doc_name)
    # embeddings = [embed_chunk(chunk) for chunk in chunks]
    embeddings = embed_chunk_list(chunks, dimensions=DIMENSION)
    logger.debug(f"生成 {len(embeddings)} 个嵌入向量")
    logger.debug(f"嵌入向量维度: {len(embeddings[0])}")
    save_embeddings(chunks, embeddings, collection_name, dimension=DIMENSION)


def query(question, collection_name=DEFAULT_COLLECTION_NAME):
    retrieved_chunks = retrieve(question, 10, select_embeddings, embed_chunk,
                                collection_name)
    reranked_chunks = rerank(question, retrieved_chunks, 3)
    answer = generate(question, reranked_chunks, call)
    logger.info(f"LLM响应内容:\n{answer}")


if __name__ == "__main__":
    """
        第一步:将文档切片,存入向量数据库
        执行一次即可
    """
    collection_name = "fangkai"
    # save("doc/放开那个女巫.txt", split_into_chunks_simple, collection_name)

    """
    查询
    """
    # question = "哆啦A梦使用的3个秘密道具分别是什么？"
    # question = "宝玉初见黛玉的描写"
    question = "罗兰是应对神造之神的威胁的"
    query(question, collection_name)
