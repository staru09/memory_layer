from qdrant_client import QdrantClient, models
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_API_KEY

EMBEDDING_DIM = 3072
COLLECTION_NAME = "facts"

_client = None
qdrant_client = None  # Module-level reference, initialized by init_collections()


def _date_to_int(date_val) -> int:
    """Convert date string or date object → YYYYMMDD integer for Qdrant range filters."""
    try:
        s = str(date_val)
        return int(s.replace("-", ""))
    except (ValueError, AttributeError, TypeError):
        return 0


def _get_client():
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        else:
            _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=False, timeout=30)
    return _client


def init_collections():
    global qdrant_client
    client = _get_client()
    qdrant_client = client
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
        # Create payload index for date filtering (required by Qdrant Cloud)
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="date_int",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        print(f"[qdrant] Created collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}-dim) with date_int index")



def upsert_facts_batch(facts: list[dict]):
    """Batch upsert. Each dict: {fact_id, embedding, fact_text, conversation_date, category}."""
    if not facts:
        return
    client = _get_client()
    points = [
        models.PointStruct(
            id=f["fact_id"],
            vector=f["embedding"],
            payload={
                "fact_text": f["fact_text"],
                "conversation_date": f.get("conversation_date", ""),
                "date_int": _date_to_int(f.get("conversation_date", "")),
                "category": f.get("category", ""),
                "fact_id": f["fact_id"],
            },
        )
        for f in facts
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def search_facts(query_embedding: list[float], top_k: int = 5,
                 date_filter: dict = None) -> list[dict]:
    """Vector search on facts. Optional date filter."""
    client = _get_client()

    search_filter = None
    if date_filter:
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="date_int",
                    range=models.Range(
                        gte=_date_to_int(date_filter["date_from"]),
                        lte=_date_to_int(date_filter["date_to"]),
                    ),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=search_filter,
        limit=top_k,
    )

    return [
        {
            "fact_id": hit.payload["fact_id"],
            "fact_text": hit.payload["fact_text"],
            "conversation_date": hit.payload.get("conversation_date"),
            "category": hit.payload.get("category"),
            "score": hit.score,
        }
        for hit in results.points
    ]


def rebuild_from_db():
    """Delete Qdrant collection and re-embed all facts from PostgreSQL."""
    import db
    from retrieval.vectorize_service import embed_texts

    facts = db.get_all_facts()
    if not facts:
        print("[qdrant] No facts in DB, nothing to rebuild.")
        return

    delete_collection()
    init_collections()

    BATCH_SIZE = 50
    total = len(facts)
    for i in range(0, total, BATCH_SIZE):
        batch = facts[i:i + BATCH_SIZE]
        texts = [f["fact_text"] for f in batch]
        embeddings = embed_texts(texts)

        points = [
            models.PointStruct(
                id=f["id"],
                vector=emb,
                payload={
                    "fact_text": f["fact_text"],
                    "conversation_date": f.get("conversation_date") or "",
                    "date_int": _date_to_int(f.get("conversation_date") or ""),
                    "category": f.get("category") or "",
                    "fact_id": f["id"],
                },
            )
            for f, emb in zip(batch, embeddings)
        ]
        _get_client().upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  [qdrant] Rebuilt {min(i + BATCH_SIZE, total)}/{total} facts")

    print(f"[qdrant] Rebuild complete: {total} facts indexed.")


def delete_collection():
    client = _get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
