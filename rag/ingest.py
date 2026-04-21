import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import get_collection

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qa.json")


def ingest():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        print("data/qa.json is empty — nothing to ingest.")
        return

    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    for i, item in enumerate(items):
        item_id = item.get("id") or f"{item.get('user_type', 'unknown')}_{i}"
        ids.append(item_id)
        documents.append(f"{item['question']}\n{item['answer']}")
        metadatas.append({
            "user_type": item["user_type"],
            "category": item.get("category", ""),
            "question": item["question"],
            "answer": item["answer"],
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(ids)} items into ChromaDB collection '{collection.name}'.")


if __name__ == "__main__":
    ingest()
