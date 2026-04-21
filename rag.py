from config import get_chroma
from logger import get_logger


log = get_logger(__name__)
vector_store = get_chroma()


def retrieve_docs(query: str, k: int = 2, score_threshold=None) -> list:
    """搜索相似文本，返回结构化字典列表。"""
    results = vector_store.similarity_search_with_score(query=query, k=k)

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score <= score_threshold]

    log.info("RAG 检索完成，query=%s, 返回=%s", query[:40], len(results))

    if not results:
        return []

    docs_data = []
    for doc, score in results:
        docs_data.append({
            "source": doc.metadata.get('source', '未知'),
            "similarity_score": round(float(score), 3),
            "content": doc.page_content
        })
        
    return docs_data


if __name__ == "__main__":
    print(retrieve_docs("裂缝分级标准"))
