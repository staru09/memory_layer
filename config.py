import os
from dotenv import load_dotenv

load_dotenv()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# PostgreSQL
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "54345"))
PG_USER = os.getenv("PG_USER", "evermemos")
PG_PASSWORD = os.getenv("PG_PASSWORD", "evermemos")
PG_DB = os.getenv("PG_DB", "evermemos")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Memory budgets (tokens)
PROFILE_TOKEN_BUDGET = int(os.getenv("PROFILE_TOKEN_BUDGET", "3000"))
SUMMARY_TOKEN_BUDGET = int(os.getenv("SUMMARY_TOKEN_BUDGET", "25000"))
COMPRESSION_THRESHOLD = float(os.getenv("COMPRESSION_THRESHOLD", "0.9"))  # 90%

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
