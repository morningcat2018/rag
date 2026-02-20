import re
from typing import List
from common_embedding import embed_chunk
from common_vector_qdrant import save_embeddings


def split_by_chapters(content: str) -> List[str]:
    """
    按章节分割《红楼梦》文本
    模式：第X回  章节标题（X为中文数字）
    """
    # 匹配"第X回"格式，支持"第X回"和"第X回  标题"两种格式
    chapter_pattern = r'第[一二三四五六七八九十百]+回\s*[^\n]*'

    # 找到所有章节标题位置
    chapters = []
    for match in re.finditer(chapter_pattern, content):
        chapters.append({
            'start': match.start(),
            'title': match.group()
        })

    # 按章节分割内容
    chunks = []
    for i, chapter in enumerate(chapters):
        start = chapter['start']
        end = chapters[i + 1]['start'] if i + 1 < len(chapters) else len(content)
        chunks.append(content[start:end].strip())

    return chunks


def smart_split_chunks(content: str, max_length: int = 2000) -> List[str]:
    """
    智能分块，优先保持章节完整，超长章节进行分段
    """
    chapters = split_by_chapters(content)
    chunks = []

    for chapter in chapters:
        if len(chapter) <= max_length:
            chunks.append(chapter)
        else:
            # 对超长章节按段落进行二次分割
            paragraphs = chapter.split('\n\n')
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_length:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + '\n\n'

            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks


def split_into_chunks(doc_file: str) -> List[str]:
    """
    分片 - 使用智能分块策略处理《红楼梦》文本
    :param doc_file:
    :return:
    """
    with open(doc_file, 'r') as file:
        content = file.read()

    return smart_split_chunks(content)


def split_into_chunks_simple(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


def save_step(doc_name):
    """
    系统初始化时执行一遍即可
    :return:
    """
    chunks = split_into_chunks(doc_name)
    # for i, chunk in enumerate(chunks):
    #     print(f"[{i}] {chunk}\n")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    # print(len(embeddings[0]))
    save_embeddings(chunks, embeddings)


if __name__ == "__main__":
    save_step("红楼梦.txt")
