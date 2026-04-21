import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = ".chromadb"
COLLECTION_NAME = "xanhsm_qa"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        chroma_abs = os.path.join(base_dir, CHROMA_PATH)
        client = chromadb.PersistentClient(path=chroma_abs)
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    return _collection
