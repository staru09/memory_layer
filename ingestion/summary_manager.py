from config import GEMINI_MODEL, SUMMARY_TOKEN_BUDGET, COMPRESSION_THRESHOLD
from gemini import gemini_client as client
import db
import os

ARCHIVE_MAX_RATIO = 0.20
_SUMMARY_COMPRESS_PATH = os.path.join(os.path.dirname(__file__), "prompts", "summary_compression.txt")


def _load_prompt(path):
    with open(path) as f:
        return f.read()


def _count_tokens(text: str) -> int:
    """Estimate token count from character length. 4 chars per token for English."""
    return max(1, int(len(text) / 4))


def append_to_rolling_summary(facts: list[dict], current_date: str, conversation_time: str = None):
    """Append new date-tagged entries to the Recent section of rolling summary.

    conversation_time: optional IST time string like '3:34 PM' to include with date.
    """
    if not facts:
        return

    # Format facts as date-tagged entries
    new_entries = []
    for f in facts:
        date = f.get("date", current_date)
        if conversation_time:
            date_tag = f"{date} {conversation_time} IST"
        else:
            date_tag = date
        new_entries.append(f"[{date_tag}] {f['text']}")
    new_text = "\n".join(new_entries)

    # Get current summary + update in single connection
    summary = db.get_and_upsert_summary()
    archive = summary["archive_text"]
    recent = summary["recent_text"]

    # Prepend to recent (newest first — LLM pays more attention to early tokens)
    if recent:
        recent = new_text + "\n" + recent
    else:
        recent = new_text

    # Count tokens
    total_text = archive + "\n" + recent if archive else recent
    token_count = _count_tokens(total_text)

    db.get_and_upsert_summary(new_recent=recent, new_token_count=token_count)
    print(f"  [summary] Appended {len(new_entries)} entries to Recent ({token_count} tokens)")


def maybe_compress_summary():
    """Compress rolling summary if it exceeds 90% of token budget.
    Also cap archive at 20% of budget — re-compress if needed."""
    summary = db.get_conversation_summary()
    token_count = summary["token_count"]
    threshold = int(SUMMARY_TOKEN_BUDGET * COMPRESSION_THRESHOLD)

    if token_count < threshold:
        return

    archive = summary["archive_text"]
    recent = summary["recent_text"]

    if not recent:
        return

    # Split recent into lines, take oldest ~25% for compression
    recent_lines = recent.strip().split("\n")
    if len(recent_lines) <= 2:
        return

    # Oldest entries are at the bottom (newest-first order), evict bottom 25%
    split_point = max(1, len(recent_lines) // 4)
    remaining_recent = "\n".join(recent_lines[:len(recent_lines) - split_point])
    entries_to_compress = "\n".join(recent_lines[len(recent_lines) - split_point:])

    print(f"  [summary] Compressing ({token_count} tokens, threshold {threshold})...")
    print(f"  [summary] Moving {split_point}/{len(recent_lines)} entries to archive")

    prompt = _load_prompt(_SUMMARY_COMPRESS_PATH)
    prompt = prompt.replace("{existing_archive}", archive if archive else "(Empty — first compression)")
    prompt = prompt.replace("{entries_to_compress}", entries_to_compress)

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        if response.text:
            new_archive = response.text.strip()

            # Check if archive exceeds 20% cap
            archive_max_tokens = int(SUMMARY_TOKEN_BUDGET * ARCHIVE_MAX_RATIO)
            archive_tokens = _count_tokens(new_archive)

            if archive_tokens > archive_max_tokens:
                print(f"  [summary] Archive too large ({archive_tokens} tokens, cap {archive_max_tokens}), re-compressing...")
                recompress_prompt = _load_prompt(_SUMMARY_COMPRESS_PATH)
                recompress_prompt = recompress_prompt.replace("{existing_archive}", "(Condense this into a shorter summary)")
                recompress_prompt = recompress_prompt.replace("{entries_to_compress}", new_archive)
                try:
                    recompress_response = client.models.generate_content(model=GEMINI_MODEL, contents=recompress_prompt)
                    if recompress_response.text:
                        new_archive = recompress_response.text.strip()
                        archive_tokens = _count_tokens(new_archive)
                        print(f"  [summary] Archive re-compressed to {archive_tokens} tokens")
                except Exception as e:
                    print(f"  [summary] Archive re-compression failed: {e}")

            total_text = new_archive + "\n" + remaining_recent
            new_token_count = _count_tokens(total_text)
            db.upsert_conversation_summary(new_archive, remaining_recent, new_token_count)
            print(f"  [summary] Compressed: {token_count} → {new_token_count} tokens (archive: {archive_tokens}, recent: {new_token_count - archive_tokens})")
    except Exception as e:
        print(f"  [summary] Compression failed: {e}")
