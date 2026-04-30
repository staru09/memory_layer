import time
import asyncio
import db
import vector_store
from retrieval.vectorize_service import embed_text
from config import RETRIEVAL_TOP_K


def hybrid_search_facts(query_embedding: list[float], kw_results: list[dict],
                        top_k: int = RETRIEVAL_TOP_K, date_filter: dict = None) -> tuple[list[dict], dict]:
    """Hybrid search: keyword results (pre-fetched from PG) + vector (Qdrant) → RRF fusion → top-k facts.
    Returns (facts, search_timings)."""

    # Vector search (Qdrant) — keyword results already fetched in the PG compound query
    t_vec = time.time()
    vec_results = vector_store.search_facts(query_embedding, top_k * 2, date_filter)
    vec_time = round(time.time() - t_vec, 3)

    # RRF fusion (k=60)
    t_rrf = time.time()
    # Vector weighted higher — users query in Hinglish, facts stored in English.
    K = 60
    KEYWORD_WEIGHT = 0.5
    VECTOR_WEIGHT = 1.5

    scores = {}
    fact_map = {}

    for rank, r in enumerate(kw_results):
        fid = r["id"]
        scores[fid] = scores.get(fid, 0) + KEYWORD_WEIGHT / (K + rank + 1)
        fact_map[fid] = {
            "fact_id": fid,
            "fact_text": r["fact_text"],
            "conversation_date": r.get("conversation_date"),
            "category": r.get("category"),
        }

    for rank, r in enumerate(vec_results):
        fid = r["fact_id"]
        scores[fid] = scores.get(fid, 0) + VECTOR_WEIGHT / (K + rank + 1)
        if fid not in fact_map:
            fact_map[fid] = {
                "fact_id": fid,
                "fact_text": r["fact_text"],
                "conversation_date": r.get("conversation_date"),
                "category": r.get("category"),
            }

    # Sort by RRF score, return top-k
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    results = []
    for fid, score in ranked:
        entry = fact_map[fid]
        entry["rrf_score"] = score
        results.append(entry)

    rrf_time = round(time.time() - t_rrf, 3)

    search_timings = {
        "vector_s": vec_time,
        "keyword_results": len(kw_results),
        "vector_results": len(vec_results),
        "rrf_s": rrf_time,
    }

    return results, search_timings

async def retrieve_for_query(query: str, query_time=None, thread_id: str = None) -> dict:
    """Query retrieval: embed + hybrid search top-5 + profile + foresight."""
    t0 = time.time()
    timings = {}

    date_filter = None

    # Step 1: Embed + PG context (with keyword) in parallel via asyncio
    async def _timed_embed():
        t = time.time()
        result = await asyncio.to_thread(embed_text, query)
        return result, round(time.time() - t, 3)

    async def _timed_context():
        t = time.time()
        result = await asyncio.to_thread(
            db.get_query_context,
            query_time, thread_id,
            query, RETRIEVAL_TOP_K * 2,
            date_filter
        )
        return result, round(time.time() - t, 3)

    (query_embedding, timings["embed_s"]), (ctx, timings["context_s"]) = await asyncio.gather(
        _timed_embed(),
        _timed_context(),
    )

    # Step 3: Qdrant vector search + RRF fusion
    t_search = time.time()
    facts, search_timings = hybrid_search_facts(query_embedding, ctx.get("keyword_results", []), RETRIEVAL_TOP_K, date_filter)
    timings["hybrid_search_s"] = round(time.time() - t_search, 3)
    timings.update(search_timings)

    timings["total_retrieval_s"] = round(time.time() - t0, 3)
    print(f"  [query-retrieval] {len(facts)} facts | embed: {timings['embed_s']}s | vector: {search_timings['vector_s']}s | keyword: {len(ctx.get('keyword_results', []))} hits | rrf: {search_timings['rrf_s']}s | ctx: {timings['context_s']}s | total: {timings['total_retrieval_s']}s")

    return {
        "profile": ctx["profile"],
        "foresight": ctx["foresight"],
        "facts": facts,
        "recent_messages": ctx.get("recent_messages", []),
        "timings": timings,
    }


## TODO: Combined Function to compose context for both chat and query, with some formatting differences if needed. For now, keep separate for clarity and flexibility.

def compose_chat_context(result: dict) -> str:
    """Compose context for chat: summary + foresight + profile."""
    parts = []
    # String Formatting for Chat Endpoints only
    # Rolling summary
    summary = result.get("summary", {})
    archive = summary.get("archive_text", "")
    recent = summary.get("recent_text", "")
    if archive or recent:
        parts.append("=== CONVERSATION HISTORY ===")
        if archive:
            parts.append("[Archive]")
            parts.append(archive)
            parts.append("")
        if recent:
            parts.append("[Recent]")
            parts.append(recent)
        parts.append("")

    # Foresight
    foresight = result.get("foresight", [])
    if foresight:
        parts.append("=== UPCOMING / TIME-BOUNDED ===")
        for fs in foresight:
            until = fs.get("valid_until")
            until_str = str(until) if until else "indefinite"
            evidence = fs.get("evidence")
            evidence_str = f" [source: {evidence}]" if evidence else ""
            parts.append(f"- {fs['description']} (valid until: {until_str}){evidence_str}")
        parts.append("")

    # Profile
    profile = result.get("profile", "")
    if profile:
        parts.append("=== USER PROFILE ===")
        parts.append(profile)

    return "\n".join(parts)


def compose_query_context(result: dict) -> str:
    """Compose context for query: matched facts + foresight + profile."""
    parts = []
    # String Formatting for query endpoints only
    # Matched facts first (highest priority for query)
    facts = result.get("facts", [])
    if facts:
        parts.append("=== MATCHED FACTS ===")
        for f in facts:
            date_tag = f" [{f['conversation_date']}]" if f.get("conversation_date") else ""
            parts.append(f"- {f['fact_text']}{date_tag}")
        parts.append("")

    # Foresight
    foresight = result.get("foresight", [])
    if foresight:
        parts.append("=== UPCOMING / TIME-BOUNDED ===")
        for fs in foresight:
            until = fs.get("valid_until")
            until_str = str(until) if until else "indefinite"
            evidence = fs.get("evidence")
            evidence_str = f" [source: {evidence}]" if evidence else ""
            parts.append(f"- {fs['description']} (valid until: {until_str}){evidence_str}")
        parts.append("")

    # Profile
    profile = result.get("profile", "")
    if profile:
        parts.append("=== USER PROFILE ===")
        parts.append(profile)

    return "\n".join(parts)

