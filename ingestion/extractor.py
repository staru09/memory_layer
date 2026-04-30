import json
import os
from config import GEMINI_MODEL
from gemini import gemini_client as client

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "extraction.txt")


def _load_prompt():
    with open(_PROMPT_PATH) as f:
        return f.read()


def extract_from_conversation(conversation: list[dict], current_date: str,
                               existing_summary: str = "",
                               extract_all_speakers: bool = False) -> dict:
    """Extract consolidated facts + foresight from a conversation.

    Returns: {"facts": [...], "foresight": [...]}
    """
    # Skip extraction if no user messages
    user_msgs = [t for t in conversation if t["role"] == "user"]
    if not user_msgs:
        return {"facts": [], "foresight": []}

    # Format conversation text with timestamps
    from datetime import timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
    lines = []
    for i, turn in enumerate(conversation):
        speaker = turn["role"]
        content = turn["content"]
        # Include timestamp if available
        ts = turn.get("created_at")
        time_str = ""
        if ts:
            if hasattr(ts, 'astimezone'):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                time_str = ts.astimezone(IST).strftime('%H:%M')
            elif isinstance(ts, str):
                try:
                    from datetime import datetime as dt
                    parsed = dt.fromisoformat(ts)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    time_str = parsed.astimezone(IST).strftime('%H:%M')
                except ValueError:
                    pass

        prefix = f"[{time_str}]" if time_str else f"[Turn {i}]"
        if not extract_all_speakers and speaker == "assistant":
            lines.append(f"{prefix} [CONTEXT ONLY] {content}")
        else:
            lines.append(f"{prefix} {content}")
    conv_text = "\n".join(lines)

    # Build prior context block
    if existing_summary:
        prior_block = f"PRIOR CONTEXT (rolling summary of earlier conversations — use ONLY for coreference resolution, do NOT extract facts from this):\n{existing_summary}"
    else:
        prior_block = "PRIOR CONTEXT: None (this is the first conversation)."

    prompt = _load_prompt()
    prompt = prompt.replace("{conversation}", conv_text)
    prompt = prompt.replace("{current_date}", current_date)
    prompt = prompt.replace("{prior_context_block}", prior_block)

    for attempt in range(3):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            if response.text is None:
                if attempt < 2:
                    import time; time.sleep(2)
                    continue
                return {"facts": [], "foresight": []}

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]

            result = json.loads(text)
            return {
                "facts": result.get("facts", []),
                "foresight": result.get("foresight", []),
            }
        except json.JSONDecodeError as e:
            if attempt < 2:
                print(f"  [extract] JSON parse error ({e}), retrying...")
                import time; time.sleep(2)
                continue
            print(f"  [extract] Failed after 3 attempts: {e}")
            return {"facts": [], "foresight": []}
        except Exception as e:
            if attempt < 2:
                import time; time.sleep(2)
                continue
            print(f"  [extract] Failed: {e}")
            return {"facts": [], "foresight": []}
