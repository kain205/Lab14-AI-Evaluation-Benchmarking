import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import get_collection

col = get_collection()
print(f"Total docs in collection: {col.count()}\n")

result = col.get(include=["metadatas", "documents"], limit=10)

for i, (doc_id, meta, doc) in enumerate(zip(result["ids"], result["metadatas"], result["documents"]), 1):
    question = meta.get("question", "N/A")
    answer = meta.get("answer", "N/A")
    user_type = meta.get("user_type", "N/A")
    print(f"[{i}] ID: {doc_id}")
    print(f"     User Type : {user_type}")
    print(f"     Question  : {question}")
    print(f"     Answer    : {answer[:200]}")
    print()
