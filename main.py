from tools.generate_step import retrieve, rerank, generate
from tools.save_step import split_into_chunks, split_into_chunks_simple
from llm.llm_deepseek import call
from vector.common_vector_milvus import select_embeddings, save_embeddings
from embedding.embedding_glm import embed_chunk_list, embed_chunk
from log.log_config import logger


def save(doc_name, split_func=split_into_chunks):
    """
    系统初始化时执行一遍即可
    :return:
    """
    chunks = split_func(doc_name)
    # embeddings = [embed_chunk(chunk) for chunk in chunks]
    embeddings = embed_chunk_list(chunks)
    logger.debug(f"生成 {len(embeddings)} 个嵌入向量")
    logger.debug(f"嵌入向量维度: {len(embeddings[0])}")
    save_embeddings(chunks, embeddings)


def query():
    query = "哆啦A梦使用的3个秘密道具分别是什么？"
    # query = "宝玉初见黛玉的描写"
    retrieved_chunks = retrieve(query, 10, select_embeddings, embed_chunk)
    reranked_chunks = rerank(query, retrieved_chunks, 3)
    answer = generate(query, reranked_chunks, call)
    logger.info(f"LLM响应内容:\n{answer}")


if __name__ == "__main__":
    """
        第一步:将文档切片,存入向量数据库
        执行一次即可
    """
    # save("doc/doc.md", split_into_chunks_simple)

    """
    查询
    """
    query()
