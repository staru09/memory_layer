import os
from config import GEMINI_MODEL, PROFILE_TOKEN_BUDGET, COMPRESSION_THRESHOLD
from gemini import gemini_client as client
import db


_PROFILE_COMPRESS_PATH = os.path.join(os.path.dirname(__file__), "prompts", "profile_compression.txt")


def _load_prompt(path):
    with open(path) as f:
        return f.read()


def _count_tokens(text: str) -> int:
    """Estimate token count from character length. 4 chars per token for English."""
    return max(1, int(len(text) / 4))


def maybe_compress_profile():
    """Compress profile if it exceeds 80% of token budget."""
    profile = db.get_user_profile()
    if not profile:
        return

    token_count = _count_tokens(profile)
    threshold = int(PROFILE_TOKEN_BUDGET * COMPRESSION_THRESHOLD)

    if token_count < threshold:
        return

    print(f"  [profile] Compressing ({token_count} tokens, threshold {threshold})...")
    prompt = _load_prompt(_PROFILE_COMPRESS_PATH)
    prompt = prompt.replace("{current_profile}", profile)

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        if response.text:
            compressed = response.text.strip()
            db.upsert_user_profile(compressed)
            new_count = _count_tokens(compressed)
            print(f"  [profile] Compressed: {token_count} → {new_count} tokens")
    except Exception as e:
        print(f"  [profile] Compression failed: {e}")
