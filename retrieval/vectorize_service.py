import time
from config import GEMINI_EMBEDDING_MODEL
from gemini import gemini_client as client


def embed_text(text: str) -> list[float]:
    """Embed a single text. Retries up to 3 times."""
    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise e


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one call. Retries up to 3 times."""
    if not texts:
        return []
    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=texts,
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise e
