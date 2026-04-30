import json
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from config import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB
from models import Foresight, ConflictLog, UserProfile, ChatThread, ChatMessage, QueryLog

_pool = None


def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        _pool = ThreadedConnectionPool(
            minconn=5, maxconn=40,
            host=PG_HOST, port=PG_PORT,
            user=PG_USER, password=PG_PASSWORD,
            dbname=PG_DB
        )
    return _pool


def get_connection():
    return _get_pool().getconn()


def release_connection(conn):
    try:
        _get_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def close_pool():
    global _pool
    if _pool and not _pool.closed:
        _pool.closeall()
        _pool = None


def init_schema():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id              SERIAL PRIMARY KEY,
            profile_text    TEXT NOT NULL DEFAULT '',
            updated_at      TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS foresight (
            id              SERIAL PRIMARY KEY,
            description     TEXT NOT NULL,
            valid_from      TIMESTAMP,
            valid_until     TIMESTAMP,
            evidence        TEXT DEFAULT '',
            duration_days   INTEGER,
            is_active       BOOLEAN DEFAULT TRUE,
            created_at      TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS facts (
            id              SERIAL PRIMARY KEY,
            fact_text       TEXT NOT NULL,
            fact_tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', fact_text)) STORED,
            category        VARCHAR(100),
            conversation_date DATE,
            source_id       VARCHAR(100),
            ingestion_number INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_facts_tsv ON facts USING GIN (fact_tsv);
        CREATE INDEX IF NOT EXISTS idx_facts_date ON facts (conversation_date);
        CREATE INDEX IF NOT EXISTS idx_facts_category ON facts (category);

        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id              SERIAL PRIMARY KEY,
            archive_text    TEXT NOT NULL DEFAULT '',
            recent_text     TEXT NOT NULL DEFAULT '',
            token_count     INTEGER DEFAULT 0,
            updated_at      TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS conflict_log (
            id              SERIAL PRIMARY KEY,
            category        VARCHAR(100) NOT NULL,
            old_value       TEXT NOT NULL,
            new_value       TEXT NOT NULL,
            resolution      VARCHAR(50) DEFAULT 'recency_wins',
            detected_at     TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chat_threads (
            id          VARCHAR(100) PRIMARY KEY,
            title       VARCHAR(200),
            created_at  TIMESTAMP DEFAULT NOW(),
            updated_at  TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id          SERIAL PRIMARY KEY,
            thread_id   VARCHAR(100) NOT NULL REFERENCES chat_threads(id),
            role        VARCHAR(20) NOT NULL,
            content     TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT NOW(),
            ingested    BOOLEAN DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_thread ON chat_messages (thread_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_unprocessed ON chat_messages (ingested) WHERE ingested = FALSE;

        CREATE TABLE IF NOT EXISTS query_logs (
            id                  SERIAL PRIMARY KEY,
            thread_id           VARCHAR(100),
            query_text          TEXT NOT NULL,
            response_text       TEXT,
            memory_context      TEXT,
            retrieval_metadata  JSONB,
            created_at          TIMESTAMP DEFAULT NOW(),
            query_time          TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    release_connection(conn)


def reset_all_tables():
    """Wipe and recreate all tables. Does NOT reset Qdrant — call vector_store separately."""
    import vector_store
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE IF EXISTS query_logs CASCADE;
        DROP TABLE IF EXISTS conflict_log CASCADE;
        DROP TABLE IF EXISTS conversation_summaries CASCADE;
        DROP TABLE IF EXISTS facts CASCADE;
        DROP TABLE IF EXISTS foresight CASCADE;
        DROP TABLE IF EXISTS user_profile CASCADE;
        DROP TABLE IF EXISTS chat_messages CASCADE;
        DROP TABLE IF EXISTS chat_threads CASCADE;
    """)
    conn.commit()
    cur.close()
    release_connection(conn)

    vector_store.delete_collection()
    vector_store.init_collections()

    init_schema()
    print("Databases reset.")


# ── Facts CRUD ──


def insert_facts_batch(facts: list[dict], source_id: str = None, ingestion_number: int = 0) -> list[int]:
    """Batch insert facts. Each dict: {text, category, date}."""
    if not facts:
        return []
    conn = get_connection()
    cur = conn.cursor()
    ids = []
    for f in facts:
        cur.execute(
            "INSERT INTO facts (fact_text, category, conversation_date, source_id, ingestion_number) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (f["text"], f.get("category", "general"), f.get("date"), source_id, ingestion_number)
        )
        ids.append(cur.fetchone()[0])
    conn.commit()
    cur.close()
    release_connection(conn)
    return ids


def insert_facts_and_foresight(facts: list[dict], foresight_entries: list, current_date,
                                source_id: str = None, ingestion_number: int = 0) -> tuple[list[int], int]:
    """Insert facts + expire/insert foresight in a single connection.
    Returns (fact_ids, expired_count)."""
    conn = get_connection()
    cur = conn.cursor()

    # Insert facts
    fact_ids = []
    for f in facts:
        cur.execute(
            "INSERT INTO facts (fact_text, category, conversation_date, source_id, ingestion_number) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (f["text"], f.get("category", "general"), f.get("date"), source_id, ingestion_number)
        )
        fact_ids.append(cur.fetchone()[0])

    # Expire old foresight
    cur.execute(
        "UPDATE foresight SET is_active = FALSE WHERE valid_until IS NOT NULL AND valid_until < %s AND is_active = TRUE",
        (current_date,)
    )
    expired = cur.rowcount

    # Insert new foresight
    for f in foresight_entries:
        cur.execute(
            "INSERT INTO foresight (description, valid_from, valid_until, evidence, duration_days) VALUES (%s, %s, %s, %s, %s)",
            (f.description, f.valid_from, f.valid_until, f.evidence, f.duration_days)
        )

    conn.commit()
    cur.close()
    release_connection(conn)
    return fact_ids, expired


async def ingest_pg_batch(facts: list[dict], foresight_entries: list, current_date: str,
                    summary_new_text: str, source_id: str = None,
                    ingestion_number: int = 0) -> tuple[list[int], int, int]:
    """Parallel PG writes across 3 connections via asyncio:
    Task 1: facts insert (needs RETURNING id)
    Task 2: foresight expire + insert
    Task 3: summary read + append
    Returns (fact_ids, expired_foresight_count, summary_token_count)."""
    import asyncio

    def _insert_facts():
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        ids = []
        for f in facts:
            cur.execute(
                "INSERT INTO facts (fact_text, category, conversation_date, source_id, ingestion_number) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (f["text"], f.get("category", "general"), f.get("date"), source_id, ingestion_number)
            )
            ids.append(cur.fetchone()["id"])
        conn.commit()
        cur.close()
        release_connection(conn)
        return ids

    def _upsert_foresight():
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE foresight SET is_active = FALSE WHERE valid_until IS NOT NULL AND valid_until < %s AND is_active = TRUE",
            (current_date,)
        )
        expired = cur.rowcount
        for f in foresight_entries:
            cur.execute(
                "INSERT INTO foresight (description, valid_from, valid_until, evidence, duration_days) VALUES (%s, %s, %s, %s, %s)",
                (f.description, f.valid_from, f.valid_until, f.evidence, f.duration_days)
            )
        conn.commit()
        cur.close()
        release_connection(conn)
        return expired

    def _append_summary():
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, archive_text, recent_text, token_count FROM conversation_summaries LIMIT 1")
        row = cur.fetchone()

        archive = row["archive_text"] if row else ""
        recent = row["recent_text"] if row else ""

        if recent:
            recent = summary_new_text + "\n" + recent
        else:
            recent = summary_new_text

        total_text = archive + "\n" + recent if archive else recent
        token_count = max(1, int(len(total_text) / 4))

        if row:
            cur.execute(
                "UPDATE conversation_summaries SET recent_text = %s, token_count = %s, updated_at = NOW() WHERE id = %s",
                (recent, token_count, row["id"])
            )
        else:
            cur.execute(
                "INSERT INTO conversation_summaries (archive_text, recent_text, token_count) VALUES (%s, %s, %s)",
                (archive, recent, token_count)
        )

        conn.commit()
        cur.close()
        release_connection(conn)
        return token_count

    fact_ids, expired, token_count = await asyncio.gather(
        asyncio.to_thread(_insert_facts),
        asyncio.to_thread(_upsert_foresight),
        asyncio.to_thread(_append_summary),
    )

    return fact_ids, expired, token_count


def get_facts_by_date(date_from: str, date_to: str) -> list[dict]:
    """Return all facts within a date range."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT id, fact_text, category, conversation_date FROM facts WHERE conversation_date BETWEEN %s AND %s ORDER BY id",
        (date_from, date_to)
    )
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return [dict(r) for r in rows]


def get_all_facts() -> list[dict]:
    """Return all facts for Qdrant rebuild."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, fact_text, category, conversation_date FROM facts ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return [dict(r) for r in rows]


def _to_or_query(query: str) -> str:
    """Convert query to OR-separated words for websearch_to_tsquery."""
    words = query.strip().split()
    return " OR ".join(words) if words else query


def keyword_search_facts(query: str, top_k: int = 5, date_filter: dict = None) -> list[dict]:
    """Full-text search on facts using websearch_to_tsquery with OR logic."""
    or_query = _to_or_query(query)
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if date_filter:
        cur.execute("""
            SELECT id, fact_text, category, conversation_date,
                   ts_rank(fact_tsv, websearch_to_tsquery('english', %s), 32) AS rank
            FROM facts
            WHERE fact_tsv @@ websearch_to_tsquery('english', %s)
              AND conversation_date BETWEEN %s AND %s
            ORDER BY rank DESC
            LIMIT %s
        """, (or_query, or_query, date_filter["date_from"], date_filter["date_to"], top_k))
    else:
        cur.execute("""
            SELECT id, fact_text, category, conversation_date,
                   ts_rank(fact_tsv, websearch_to_tsquery('english', %s), 32) AS rank
            FROM facts
            WHERE fact_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (or_query, or_query, top_k))

    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return [dict(r) for r in rows]


# ── User Profile ──

def get_user_profile() -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT profile_text FROM user_profile LIMIT 1")
    row = cur.fetchone()
    cur.close()
    release_connection(conn)
    return row[0] if row else ""


def upsert_user_profile(profile_text: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM user_profile LIMIT 1")
    existing = cur.fetchone()
    if existing:
        cur.execute(
            "UPDATE user_profile SET profile_text = %s, updated_at = NOW() WHERE id = %s",
            (profile_text, existing[0])
        )
    else:
        cur.execute(
            "INSERT INTO user_profile (profile_text) VALUES (%s)",
            (profile_text,)
        )
    conn.commit()
    cur.close()
    release_connection(conn)


def get_and_upsert_profile(new_profile: str = None, conflicts: list = None) -> str:
    """Get current profile and optionally update it + log conflicts in a single connection."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, profile_text FROM user_profile LIMIT 1")
    row = cur.fetchone()
    current_profile = row[1] if row else ""
    current_id = row[0] if row else None

    if new_profile is not None:
        if current_id:
            cur.execute(
                "UPDATE user_profile SET profile_text = %s, updated_at = NOW() WHERE id = %s",
                (new_profile, current_id)
            )
        else:
            cur.execute(
                "INSERT INTO user_profile (profile_text) VALUES (%s)",
                (new_profile,)
            )

    if conflicts:
        for c in conflicts:
            cur.execute(
                "INSERT INTO conflict_log (category, old_value, new_value, resolution) VALUES (%s, %s, %s, %s)",
                (c.get("category", "unknown"), c.get("old_value", ""), c.get("new_value", ""), "recency_wins")
            )

    if new_profile is not None or conflicts:
        conn.commit()

    cur.close()
    release_connection(conn)
    return current_profile


# ── Foresight ──

def insert_foresight(f: Foresight) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO foresight (description, valid_from, valid_until, evidence, duration_days) VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (f.description, f.valid_from, f.valid_until, f.evidence, f.duration_days)
    )
    fid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    release_connection(conn)
    return fid


def expire_and_insert_foresight(current_date, foresight_entries: list) -> int:
    """Expire old foresight + insert new entries in a single connection."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE foresight SET is_active = FALSE WHERE valid_until IS NOT NULL AND valid_until < %s AND is_active = TRUE",
        (current_date,)
    )
    expired = cur.rowcount
    for f in foresight_entries:
        cur.execute(
            "INSERT INTO foresight (description, valid_from, valid_until, evidence, duration_days) VALUES (%s, %s, %s, %s, %s)",
            (f.description, f.valid_from, f.valid_until, f.evidence, f.duration_days)
        )
    conn.commit()
    cur.close()
    release_connection(conn)
    return expired


def get_active_foresight(query_time) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, description, valid_from, valid_until, evidence, created_at
        FROM foresight
        WHERE is_active = TRUE
          AND valid_from <= %s
          AND (valid_until IS NULL OR valid_until >= %s)
        ORDER BY created_at DESC
    """, (query_time, query_time))
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return [dict(r) for r in rows]


def get_query_context(query_time, thread_id: str = None,
                      keyword_query: str = None, keyword_top_k: int = 10,
                      date_filter: dict = None) -> dict:
    """Fetch all query context in a single SQL query — 1 round-trip:
    profile + active foresight + unprocessed messages + keyword search results."""
    conn = get_connection()
    cur = conn.cursor()

    # Build keyword subquery
    if keyword_query:
        or_query = _to_or_query(keyword_query)
        if date_filter:
            kw_subquery = """
                (SELECT COALESCE(json_agg(row_to_json(k)), '[]'::json) FROM (
                    SELECT id, fact_text, category, conversation_date,
                           ts_rank(fact_tsv, websearch_to_tsquery('english', %s), 32) AS rank
                    FROM facts
                    WHERE fact_tsv @@ websearch_to_tsquery('english', %s)
                      AND conversation_date BETWEEN %s AND %s
                    ORDER BY rank DESC
                    LIMIT %s
                ) k) as keyword_results
            """
            kw_params = (or_query, or_query, date_filter["date_from"], date_filter["date_to"], keyword_top_k)
        else:
            kw_subquery = """
                (SELECT COALESCE(json_agg(row_to_json(k)), '[]'::json) FROM (
                    SELECT id, fact_text, category, conversation_date,
                           ts_rank(fact_tsv, websearch_to_tsquery('english', %s), 32) AS rank
                    FROM facts
                    WHERE fact_tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                ) k) as keyword_results
            """
            kw_params = (or_query, or_query, keyword_top_k)
    else:
        kw_subquery = ""
        kw_params = ()

    # Build messages subquery — last 20 messages regardless of ingestion status
    if thread_id:
        msg_subquery = """
            (SELECT COALESCE(json_agg(row_to_json(m)), '[]'::json) FROM (
                SELECT * FROM (
                    SELECT id, thread_id, role, content, created_at, ingested
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at DESC
                    LIMIT 20
                ) last_20
                ORDER BY created_at ASC
            ) m) as recent_messages
        """
        msg_params = (thread_id,)
    else:
        msg_subquery = ""
        msg_params = ()

    # Combine into single query
    subqueries = [
        "(SELECT profile_text FROM user_profile LIMIT 1) as profile",
        """(SELECT COALESCE(json_agg(row_to_json(f)), '[]'::json) FROM (
                SELECT id, description, valid_from, valid_until, evidence, created_at
                FROM foresight
                WHERE is_active = TRUE
                  AND valid_from <= %s
                  AND (valid_until IS NULL OR valid_until >= %s)
                ORDER BY created_at DESC
            ) f) as foresight""",
    ]
    params = [query_time, query_time]

    if msg_subquery:
        subqueries.append(msg_subquery)
        params.extend(msg_params)
    if kw_subquery:
        subqueries.append(kw_subquery)
        params.extend(kw_params)

    sql = "SELECT " + ",\n".join(subqueries)
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    cur.close()
    release_connection(conn)

    idx = 0
    profile = row[idx] or ""; idx += 1
    foresight = row[idx] if row[idx] else []; idx += 1
    recent_messages = []
    if thread_id:
        recent_messages = row[idx] if row[idx] else []; idx += 1
    keyword_results = []
    if keyword_query:
        keyword_results = row[idx] if row[idx] else []; idx += 1

    return {
        "profile": profile,
        "foresight": foresight,
        "recent_messages": recent_messages,
        "keyword_results": keyword_results,
    }


def get_chat_context(thread_id: str, query_time) -> dict:
    """Fetch all chat context in a single SQL query — 1 round-trip:
    profile + active foresight + conversation summary + recent messages.
    Messages: all unprocessed + last 10 (whichever gives more), deduped by id."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            (SELECT profile_text FROM user_profile LIMIT 1) as profile,
            (SELECT COALESCE(json_agg(row_to_json(f)), '[]'::json) FROM (
                SELECT id, description, valid_from, valid_until, evidence, created_at
                FROM foresight
                WHERE is_active = TRUE
                  AND valid_from <= %s
                  AND (valid_until IS NULL OR valid_until >= %s)
                ORDER BY created_at DESC
            ) f) as foresight,
            (SELECT row_to_json(s) FROM (
                SELECT archive_text, recent_text, token_count
                FROM conversation_summaries LIMIT 1
            ) s) as summary,
            (SELECT COALESCE(json_agg(row_to_json(m)), '[]'::json) FROM (
                SELECT * FROM (
                    SELECT id, thread_id, role, content, created_at, ingested
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at DESC
                    LIMIT 20
                ) last_20
                ORDER BY created_at ASC
            ) m) as recent_messages
    """, (query_time, query_time, thread_id))

    row = cur.fetchone()
    cur.close()
    release_connection(conn)

    profile = row[0] or ""
    foresight = row[1] if row[1] else []
    summary = row[2] if row[2] else {"archive_text": "", "recent_text": "", "token_count": 0}
    recent_messages = row[3] if row[3] else []

    return {
        "profile": profile,
        "foresight": foresight,
        "summary": summary,
        "recent_messages": recent_messages,
    }


def expire_foresight(current_date):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE foresight SET is_active = FALSE WHERE valid_until IS NOT NULL AND valid_until < %s AND is_active = TRUE",
        (current_date,)
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    release_connection(conn)
    return count


# ── Conflict Log ──

def insert_conflict_log(conflict: ConflictLog) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conflict_log (category, old_value, new_value, resolution) VALUES (%s, %s, %s, %s) RETURNING id",
        (conflict.category, conflict.old_value, conflict.new_value, conflict.resolution)
    )
    cid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    release_connection(conn)
    return cid


# ── Conversation Summaries (two-tier: archive + recent) ──

def get_conversation_summary() -> dict:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT archive_text, recent_text, token_count FROM conversation_summaries LIMIT 1")
    row = cur.fetchone()
    cur.close()
    release_connection(conn)
    if not row:
        return {"archive_text": "", "recent_text": "", "token_count": 0}
    return dict(row)


def get_and_upsert_summary(new_archive: str = None, new_recent: str = None, new_token_count: int = None) -> dict:
    """Get current summary and optionally update it in a single connection."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, archive_text, recent_text, token_count FROM conversation_summaries LIMIT 1")
    row = cur.fetchone()

    current = dict(row) if row else {"id": None, "archive_text": "", "recent_text": "", "token_count": 0}

    if new_archive is not None or new_recent is not None or new_token_count is not None:
        archive = new_archive if new_archive is not None else current["archive_text"]
        recent = new_recent if new_recent is not None else current["recent_text"]
        token_count = new_token_count if new_token_count is not None else current["token_count"]

        if current["id"]:
            cur.execute(
                "UPDATE conversation_summaries SET archive_text = %s, recent_text = %s, token_count = %s, updated_at = NOW() WHERE id = %s",
                (archive, recent, token_count, current["id"])
            )
        else:
            cur.execute(
                "INSERT INTO conversation_summaries (archive_text, recent_text, token_count) VALUES (%s, %s, %s)",
                (archive, recent, token_count)
            )
        conn.commit()

    cur.close()
    release_connection(conn)
    return {"archive_text": current["archive_text"], "recent_text": current["recent_text"], "token_count": current["token_count"]}


def upsert_conversation_summary(archive_text: str, recent_text: str, token_count: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM conversation_summaries LIMIT 1")
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE conversation_summaries SET archive_text = %s, recent_text = %s, token_count = %s, updated_at = NOW() WHERE id = %s",
            (archive_text, recent_text, token_count, row[0])
        )
    else:
        cur.execute(
            "INSERT INTO conversation_summaries (archive_text, recent_text, token_count) VALUES (%s, %s, %s)",
            (archive_text, recent_text, token_count)
        )
    conn.commit()
    cur.close()
    release_connection(conn)


# ── Ingestion Counter ──

_ingestion_count = 0


def get_and_increment_ingestion_count() -> int:
    global _ingestion_count
    _ingestion_count += 1
    return _ingestion_count


# ── System Stats ──

def get_system_stats() -> dict:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM conflict_log")
    total_conflicts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM foresight WHERE is_active = TRUE")
    active_foresight = cur.fetchone()[0]
    profile = get_user_profile()
    cur.close()
    release_connection(conn)
    return {
        "total_conflicts": total_conflicts,
        "active_foresight": active_foresight,
        "has_profile": bool(profile),
    }


# ── Chat CRUD ──

def get_thread(thread_id: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM chat_threads WHERE id = %s", (thread_id,))
    row = cur.fetchone()
    cur.close()
    release_connection(conn)
    return row


def create_thread(thread_id: str, title: str = None) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_threads (id, title) VALUES (%s, %s) RETURNING id",
        (thread_id, title)
    )
    tid = cur.fetchone()[0]
    conn.commit()
    cur.close()
    release_connection(conn)
    return tid


def list_threads(limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT * FROM chat_threads ORDER BY updated_at DESC LIMIT %s", (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return rows


def insert_message(thread_id: str, role: str, content: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s) RETURNING id",
        (thread_id, role, content)
    )
    msg_id = cur.fetchone()[0]
    cur.execute(
        "UPDATE chat_threads SET updated_at = NOW() WHERE id = %s", (thread_id,)
    )
    conn.commit()
    cur.close()
    release_connection(conn)
    return msg_id


def get_thread_messages(thread_id: str, limit: int = 50, before_id: int = None) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    if before_id:
        cur.execute(
            "SELECT * FROM chat_messages WHERE thread_id = %s AND id < %s ORDER BY id DESC LIMIT %s",
            (thread_id, before_id, limit)
        )
    else:
        cur.execute(
            "SELECT * FROM chat_messages WHERE thread_id = %s ORDER BY id DESC LIMIT %s",
            (thread_id, limit)
        )
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return list(reversed(rows))


def get_unprocessed_messages(thread_id: str) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT * FROM chat_messages WHERE thread_id = %s AND ingested = FALSE ORDER BY created_at",
        (thread_id,)
    )
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return rows


def mark_messages_ingested(message_ids: list[int]):
    if not message_ids:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE chat_messages SET ingested = TRUE WHERE id = ANY(%s)",
        (message_ids,)
    )
    conn.commit()
    cur.close()
    release_connection(conn)


def get_threads_with_old_unprocessed(minutes: int = 10) -> list[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT thread_id FROM chat_messages
        WHERE ingested = FALSE
          AND created_at < NOW() - INTERVAL '%s minutes'
    """, (minutes,))
    thread_ids = [row[0] for row in cur.fetchall()]
    cur.close()
    release_connection(conn)
    return thread_ids


# ── Query Log ──

def insert_query_log(log: QueryLog) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO query_logs (thread_id, query_text, response_text, memory_context,
                                   retrieval_metadata, query_time)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (log.thread_id, log.query_text, log.response_text, log.memory_context,
         json.dumps(log.retrieval_metadata) if log.retrieval_metadata else None,
         log.query_time)
    )
    log_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    release_connection(conn)
    return log_id
