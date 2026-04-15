from config import get_chroma
from logger import get_logger


log = get_logger(__name__)
vector_store = get_chroma()


def retrieve_docs(query: str, k: int = 2, score_threshold=None) -> str:
    """搜索相似文本，可选相似度阈值过滤。"""
    results = vector_store.similarity_search_with_score(query=query, k=k)

    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score <= score_threshold]

    log.info("RAG 检索完成，query=%s, 返回=%s", query[:40], len(results))

    if not results:
        return "未检索到相关资料。"

    lines = []
    for i, (doc, score) in enumerate(results, 1):
        lines.append(f"文档 {i} (相似度 {score:.3f}):")
        lines.append(f"来源: {doc.metadata.get('source', '未知')}")
        lines.append(f"内容: {doc.page_content}")
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    print(retrieve_docs("裂缝分级标准"))
