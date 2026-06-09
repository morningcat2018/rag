from generate_step import *
from save_step import *
from llm.llm_deepseek import call
from vector.common_vector_qdrant import select_embeddings, save_embeddings
from embedding.embedding_bailian import embed_chunk_list, embed_chunk


def save_step(doc_name):
    """
    系统初始化时执行一遍即可
    :return:
    """
    chunks = split_into_chunks(doc_name)
    # embeddings = [embed_chunk(chunk) for chunk in chunks]
    embeddings = embed_chunk_list(chunks)
    logger.debug(f"生成 {len(embeddings)} 个嵌入向量")
    logger.debug(f"嵌入向量维度: {len(embeddings[0])}")
    save_embeddings(chunks, embeddings)


if __name__ == "__main__":
    """
        第一步:将文档切片,存入向量数据库
        执行一次即可
    """
    # save_step("红楼梦.txt")

    """
    查询
    """
    # query = "哆啦A梦使用的3个秘密道具分别是什么？"
    query = "宝玉初见黛玉的描写"

    retrieved_chunks = retrieve(query, 10, select_embeddings, embed_chunk)
    # for i, chunk in enumerate(retrieved_chunks):
    #     print(f"[{i}] {chunk}\n")
    reranked_chunks = rerank(query, retrieved_chunks, 3)
    # for i, chunk in enumerate(reranked_chunks):
    #     print(f"[{i}] {chunk}\n")
    answer = generate(query, reranked_chunks, call)
    logger.info(f"LLM响应内容:\n{answer}")
