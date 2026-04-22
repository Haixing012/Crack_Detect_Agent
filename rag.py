from config import get_chroma
from logger import get_logger


log = get_logger(__name__)
_vector_store = None


def _get_vector_store():
    """惰性加载 Chroma 向量库"""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_chroma()
    return _vector_store


def retrieve_docs(query: str, k: int = 2, score_threshold=None) -> list:
    """搜索相似文本，返回结构化字典列表。"""
    store = _get_vector_store()
    try:
        results = store.similarity_search_with_score(query=query, k=k)
    except ConnectionError as e:
        log.error("RAG 检索失败：无法连接 Ollama 服务，请确认 ollama serve 已启动。%s", str(e))
        return []
    except Exception as e:
        log.exception("RAG 检索异常")
        return []

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score <= score_threshold]

    log.info("RAG 检索完成，query=%s, 返回=%s", query[:40], len(results))

    if not results:
        return []

    docs_data = []
    for doc, score in results:
        docs_data.append({
            "source": doc.metadata.get('source', '未知'),
            "score": round(float(score), 3),
            "content": doc.page_content
        })
        
    return docs_data


if __name__ == "__main__":
    print(retrieve_docs("裂缝分级标准"))
