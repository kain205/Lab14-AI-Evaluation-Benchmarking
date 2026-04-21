"""
Ingest Facebook group posts into ChromaDB.
- question = post text
- answer   = top comments joined together
- user_type is mapped from groupTitle
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import get_collection

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Dataset",
    "dataset_facebook-groups-scraper_2026-04-08_10-56-16-341.json",
)

# Map groupTitle keywords → user_type (edit as needed)
GROUP_TYPE_MAP = {
    "Bike & Taxi": "tai_xe_taxi",   # mixed community → taxi (adjust if needed)
    "Taxi":        "tai_xe_taxi",
    "Bike":        "tai_xe_bike",
    "Nhà hàng":    "nha_hang",
    "Hành khách":  "nguoi_dung",
}

DEFAULT_USER_TYPE = "tai_xe_taxi"


def map_user_type(group_title: str) -> str:
    for keyword, user_type in GROUP_TYPE_MAP.items():
        if keyword.lower() in group_title.lower():
            return user_type
    return DEFAULT_USER_TYPE


def ingest_facebook():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    # Only keep posts that have at least one comment with non-empty text
    valid_posts = [
        p for p in posts
        if p.get("text", "").strip()
        and p.get("topComments")
        and any(c.get("text", "").strip() for c in p["topComments"])
    ]

    if not valid_posts:
        print("No posts with comments found — nothing to ingest.")
        return

    collection = get_collection()

    ids, documents, metadatas = [], [], []

    for post in valid_posts:
        post_id = "fb_" + post.get("legacyId", post["id"])
        question = post["text"].strip()

        # Join all non-empty comment texts as the answer
        answer_parts = [
            c["text"].strip()
            for c in post["topComments"]
            if c.get("text", "").strip()
        ]
        answer = "\n---\n".join(answer_parts)

        user_type = map_user_type(post.get("groupTitle", ""))

        ids.append(post_id)
        documents.append(f"{question}\n{answer}")
        metadatas.append({
            "user_type": user_type,
            "category": "community",
            "question": question,
            "answer": answer,
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(ids)} Facebook posts into ChromaDB (source: community).")


if __name__ == "__main__":
    ingest_facebook()
