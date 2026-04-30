import re
import time
import asyncio
from datetime import datetime, timezone, timedelta
import db
import vector_store
from models import Foresight
from ingestion.extractor import extract_from_conversation
from ingestion.profile_extractor import update_user_profile
from ingestion.profile_manager import maybe_compress_profile
from ingestion.summary_manager import maybe_compress_summary
from retrieval.vectorize_service import embed_texts


def ingest_conversation(conversation: list[dict], source_id: str = "default",
                        current_date: str = None, interactive: bool = False,
                        extract_all_speakers: bool = False,
                        force_profile_update: bool = False):
    """Ingestion pipeline: extract → facts (PG + Qdrant) + summary + foresight + profile (every 5th)."""
    IST = timezone(timedelta(hours=5, minutes=30))
    if current_date is None:
        current_date = datetime.now(IST).strftime("%Y-%m-%d")

    # Extract conversation time from first message's created_at
    conversation_time = None
    first_msg = conversation[0] if conversation else None
    if first_msg and first_msg.get("created_at"):
        ts = first_msg["created_at"]
        if hasattr(ts, 'astimezone'):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            conversation_time = ts.astimezone(IST).strftime("%I:%M %p").lstrip("0")
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                conversation_time = dt.astimezone(IST).strftime("%-I:%M %p")
            except ValueError:
                pass

    timings = {}
    pipeline_start = time.time()

    # ── Step 0: Fetch existing summary for coreference resolution ──
    t = time.time()
    summary = db.get_conversation_summary()
    existing_context = ""
    if summary["archive_text"]:
        existing_context += summary["archive_text"] + "\n"
    if summary["recent_text"]:
        existing_context += summary["recent_text"]
    timings["ctx_fetch"] = time.time() - t
    print(f"[1/5 context] {len(existing_context)} chars ({timings['ctx_fetch']:.1f}s)")

    # ── Step 1: Extract facts + foresight (1 LLM call) ──
    t = time.time()
    result = extract_from_conversation(
        conversation, current_date, existing_context,
        extract_all_speakers=extract_all_speakers
    )
    facts = result["facts"]
    foresight = result["foresight"]
    timings["extract"] = time.time() - t

    # Sanitize dates — LLM sometimes returns invalid date strings
    _date_re = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    for f in facts:
        if f.get("date") and not _date_re.match(str(f["date"])):
            print(f"  [warn] Invalid date '{f['date']}' in fact, using {current_date}")
            f["date"] = current_date
    _datetime_re = re.compile(r'^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$')
    for fs in foresight:
        for key in ("valid_from", "valid_until"):
            if fs.get(key) and not _datetime_re.match(str(fs[key])):
                print(f"  [warn] Invalid {key} '{fs[key]}' in foresight, setting to None")
                fs[key] = None

    print(f"[2/5 extract] {len(facts)} facts, {len(foresight)} foresight ({timings['extract']:.1f}s)")

    if not facts and not foresight:
        print(f"  Nothing extracted, skipping pipeline.")
        return

    # Get ingestion count
    ingestion_count = db.get_and_increment_ingestion_count()
    should_update_profile = True

    # ── Step 2: PG store + Embed + Profile (all parallel via asyncio) ──
    def _run_embed_sync():
        t = time.time()
        embeddings = embed_texts([f["text"] for f in facts])
        elapsed = time.time() - t
        print(f"    [embed] {len(facts)} facts embedded in {elapsed:.1f}s")
        return elapsed, embeddings

    def _run_profile_sync():
        if extract_all_speakers:
            # Benchmark mode: skip profile update — rolling summary has both speakers' facts
            # Profile would mix speakers and confuse the LLM
            print(f"    [profile] Skipped (multi-speaker benchmark mode)")
            return 0.0, 0
        t = time.time()
        profile, conflicts = update_user_profile(facts)
        elapsed = time.time() - t
        print(f"    [profile] Updated in {elapsed:.1f}s ({len(conflicts)} conflicts)")
        return elapsed, len(conflicts)

    print(f"[3/5 parallel] PG store + Embed" + (" + Profile" if should_update_profile else "") + "...")
    t_parallel = time.time()

    # Prepare PG batch args
    foresight_entries = [
        Foresight(
            description=fs["description"],
            valid_from=fs.get("valid_from"),
            valid_until=fs.get("valid_until"),
            evidence=fs.get("evidence", ""),
            duration_days=fs.get("duration_days"),
        )
        for fs in foresight
    ]
    new_entries = []
    for f in facts:
        date = f.get("date", current_date)
        date_tag = f"{date} {conversation_time} IST" if conversation_time else date
        new_entries.append(f"[{date_tag}] {f['text']}")
    summary_new_text = "\n".join(new_entries)

    async def _timed_pg():
        t = time.time()
        result = await db.ingest_pg_batch(
            facts, foresight_entries, current_date, summary_new_text,
            source_id=source_id, ingestion_number=ingestion_count
        )
        elapsed = time.time() - t
        return elapsed, result

    async def _run_parallel():
        tasks = [
            _timed_pg(),
            asyncio.to_thread(_run_embed_sync),
        ]
        if should_update_profile:
            tasks.append(asyncio.to_thread(_run_profile_sync))

        return await asyncio.gather(*tasks)

    # Ingestion runs in a background daemon thread — safe to create event loop
    results = asyncio.run(_run_parallel())

    timings["pg_store"], (fact_ids, expired, token_count) = results[0]
    if expired:
        print(f"    [pg] Expired {expired} foresight entries")
    print(f"    [pg] {len(facts)} facts + {len(foresight)} foresight + summary ({token_count} tokens) in {timings['pg_store']:.1f}s")

    timings["embed"], embeddings = results[1]

    if should_update_profile:
        timings["profile"], conflict_count = results[2]
    else:
        timings["profile"] = 0
        conflict_count = 0

    timings["parallel_total"] = time.time() - t_parallel

    # ── Step 3: Qdrant upsert (needs fact_ids + embeddings) ──
    t = time.time()
    qdrant_facts = []
    for f, fid, emb in zip(facts, fact_ids, embeddings):
        qdrant_facts.append({
            "fact_id": fid,
            "embedding": emb,
            "fact_text": f["text"],
            "conversation_date": f.get("date", current_date),
            "category": f.get("category", "general"),
        })
    vector_store.upsert_facts_batch(qdrant_facts)
    timings["qdrant_upsert"] = time.time() - t
    print(f"[4/5 qdrant] {len(facts)} facts upserted in {timings['qdrant_upsert']:.1f}s")

    # ── Step 4: Compression if needed ──
    t = time.time()
    maybe_compress_summary()
    timings["summary_compress"] = time.time() - t

    t = time.time()
    if should_update_profile:
        maybe_compress_profile()
    timings["profile_compress"] = time.time() - t

    timings["total"] = time.time() - pipeline_start

    # ── Final summary ──
    print(f"\n[5/5 done] {len(facts)} facts, {len(foresight)} foresight, {conflict_count} conflicts (ingestion #{ingestion_count})")
    print(f"  ctx_fetch:        {timings['ctx_fetch']:.1f}s")
    print(f"  extract (LLM):    {timings['extract']:.1f}s")
    print(f"  pg_store:         {timings['pg_store']:.1f}s  (parallel)")
    print(f"  embed (Gemini):   {timings['embed']:.1f}s  (parallel)")
    print(f"  profile (LLM):    {timings['profile']:.1f}s  (parallel)")
    print(f"  parallel wall:    {timings['parallel_total']:.1f}s")
    print(f"  qdrant_upsert:    {timings['qdrant_upsert']:.1f}s")
    print(f"  summary_compress: {timings['summary_compress']:.1f}s")
    print(f"  profile_compress: {timings['profile_compress']:.1f}s")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:            {timings['total']:.1f}s")


