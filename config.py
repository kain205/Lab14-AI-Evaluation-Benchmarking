import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
CHROMA_PATH = ".chromadb"
COLLECTION_NAME = "xanhsm_qa"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3
