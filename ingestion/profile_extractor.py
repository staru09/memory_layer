import json
import os
from config import GEMINI_MODEL
from gemini import gemini_client as client
from ingestion.profile_ops import number_profile_lines, apply_operations
import db

_PROFILE_UPDATE_PATH = os.path.join(os.path.dirname(__file__), "prompts", "profile_update.txt")

def _load_prompt(path):
    with open(path) as f:
        return f.read()


def update_user_profile(new_facts: list[dict]) -> tuple[str, list[dict]]:
    """Update profile with new facts using add/update/delete operations.

    The LLM outputs structured operations instead of rewriting the full profile.
    Operations are applied programmatically for predictability.

    Args:
        new_facts: list of {"text": ..., "category": ..., "date": ...}

    Returns:
        (updated_profile_text, conflicts_list)
    """
    if not new_facts:
        return db.get_user_profile(), []

    existing_profile = db.get_and_upsert_profile()
    is_first_update = not existing_profile

    # For the LLM prompt, show a placeholder so it knows to only add
    prompt_profile = existing_profile if existing_profile else "(Empty — first update. Use only 'add' operations.)"

    # Number lines so LLM can reference them
    numbered_profile = number_profile_lines(prompt_profile)

    # Format new facts grouped by category
    facts_text = "\n".join(
        f"- [{f['category']}] ({f.get('date', 'unknown')}) {f['text']}"
        for f in new_facts
    )

    prompt = _load_prompt(_PROFILE_UPDATE_PATH)
    prompt = prompt.replace("{existing_profile}", numbered_profile)
    prompt = prompt.replace("{new_facts}", facts_text)

    for attempt in range(3):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if response.text is None:
                if attempt < 2:
                    import time; time.sleep(2)
                    continue
                return existing_profile, []

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]

            parsed = json.loads(text)
            operations = parsed.get("operations", [])
            conflicts = parsed.get("conflicts", [])

            # Check if no-op
            if len(operations) == 1 and operations[0].get("action") == "none":
                print(f"  [profile] No changes needed")
                return existing_profile, []

            # Apply operations programmatically (use empty base on first update)
            base_profile = "" if is_first_update else existing_profile
            updated_profile = apply_operations(base_profile, operations)

            # Store profile + log conflicts in single connection
            db.get_and_upsert_profile(new_profile=updated_profile, conflicts=conflicts)

            op_summary = ", ".join(
                f"{sum(1 for o in operations if o.get('action') == a)} {a}"
                for a in ["add", "update", "delete"]
                if any(o.get("action") == a for o in operations)
            )
            print(f"  [profile] Updated ({op_summary}, {len(conflicts)} conflicts, {len(updated_profile)} chars)")
            return updated_profile, conflicts

        except json.JSONDecodeError as e:
            if attempt < 2:
                print(f"  [profile] JSON parse error ({e}), retrying...")
                import time; time.sleep(2)
                continue
            print(f"  [profile] Failed after 3 attempts: {e}")
            return existing_profile, []
        except Exception as e:
            if attempt < 2:
                import time; time.sleep(2)
                continue
            print(f"  [profile] Failed: {e}")
            return existing_profile, []
