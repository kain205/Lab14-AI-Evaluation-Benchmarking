import logging

from rag.vectorstore import get_collection

logger = logging.getLogger(__name__)


def retrieve(query: str, user_type: str, top_k: int = 2) -> list[dict]:
    collection = get_collection()

    def _query(where: dict | None, n: int = top_k) -> list[dict]:
        kwargs = dict(query_texts=[query], n_results=n)
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        chunks = []
        if results and results["metadatas"]:
            for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
                chunks.append({
                    "question": meta.get("question", ""),
                    "answer": meta.get("answer", ""),
                    "category": meta.get("category", ""),
                    "score": distance,
                })
        return chunks

    # Search 1: filtered by user_type (skip if not provided)
    typed_chunks = _query({"user_type": user_type}) if user_type else []

    # Search 2: no filter — double top_k if no role to compensate
    all_chunks = _query(None, n=top_k if user_type else top_k * 2)

    # Merge, deduplicate by question text, typed results first
    seen = set()
    merged = []
    for chunk in typed_chunks + all_chunks:
        key = chunk["question"]
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    logger.info(
        "[RAG] user_type=%s | typed=%d all=%d merged=%d | query=%r",
        user_type, len(typed_chunks), len(all_chunks), len(merged), query[:80],
    )
    for i, c in enumerate(merged):
        logger.info(
            "[RAG] chunk[%d] score=%.4f cat=%s | Q: %s | A: %s",
            i, c["score"], c["category"], c["question"][:100], c["answer"][:100],
        )

    return merged
