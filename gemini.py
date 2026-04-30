from datetime import datetime
from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL


gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def calculate_time_difference(from_time: str, to_time: str) -> dict:
    """Calculate the difference between two times/dates.

    Args:
        from_time: Start time in HH:MM or YYYY-MM-DD or YYYY-MM-DD HH:MM format
        to_time: End time in HH:MM or YYYY-MM-DD or YYYY-MM-DD HH:MM format

    Returns:
        Dictionary with total_minutes, hours, minutes, and a human-readable summary.
    """
    formats = ["%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
    from_dt = to_dt = None
    for fmt in formats:
        try:
            from_dt = datetime.strptime(from_time.strip(), fmt)
            break
        except ValueError:
            continue
    for fmt in formats:
        try:
            to_dt = datetime.strptime(to_time.strip(), fmt)
            break
        except ValueError:
            continue

    if from_dt is None or to_dt is None:
        return {"error": f"Could not parse times: from_time='{from_time}', to_time='{to_time}'"}

    diff = to_dt - from_dt
    total_minutes = int(diff.total_seconds() / 60)
    hours = abs(total_minutes) // 60
    minutes = abs(total_minutes) % 60
    if total_minutes < 0:
        summary = f"{from_time} is {hours}hr {minutes}min after {to_time}" if hours else f"{from_time} is {abs(total_minutes)} minutes after {to_time}"
    else:
        summary = f"{hours}hr {minutes}min have passed from {from_time} to {to_time}" if hours else f"{total_minutes} minutes have passed from {from_time} to {to_time}"

    return {
        "total_minutes": total_minutes,
        "hours": hours,
        "minutes": minutes,
        "summary": summary,
    }


_time_tool = genai.types.Tool(
    function_declarations=[
        genai.types.FunctionDeclaration(
            name="calculate_time_difference",
            description="Calculate the exact time difference between two timestamps. Use this whenever you need to compute elapsed time, remaining time, or duration between events. Pass times in HH:MM or YYYY-MM-DD HH:MM format.",
            parameters=genai.types.Schema(
                type="OBJECT",
                properties={
                    "from_time": genai.types.Schema(
                        type="STRING",
                        description="Start time in HH:MM or YYYY-MM-DD HH:MM format"
                    ),
                    "to_time": genai.types.Schema(
                        type="STRING",
                        description="End time in HH:MM or YYYY-MM-DD HH:MM format"
                    ),
                },
                required=["from_time", "to_time"],
            ),
        )
    ]
)


def call_gemini_with_tools(prompt: str) -> str:
    """Call Gemini with the time calculator tool. Handles tool call loop."""
    user_content = genai.types.Content(
        parts=[genai.types.Part.from_text(text=prompt)],
        role="user",
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[user_content],
        config=genai.types.GenerateContentConfig(tools=[_time_tool]),
    )

    max_rounds = 5

    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
        print(f"[ToolCall] Empty response from Gemini — no candidates/parts")
        return response.text.strip() if response.text else "Sorry yaar, kuch issue ho gaya."

    contents = [user_content, response.candidates[0].content]

    for round_num in range(max_rounds):
        part = response.candidates[0].content.parts[0]
        if not hasattr(part, 'function_call') or part.function_call is None:
            print(f"[ToolCall] No tool call — returning text response (after {round_num} tool rounds)")
            return response.text.strip()

        fc = part.function_call
        print(f"[ToolCall] Round {round_num + 1}: Gemini called '{fc.name}' with args: {dict(fc.args)}")

        if fc.name == "calculate_time_difference":
            result = calculate_time_difference(
                from_time=fc.args.get("from_time", ""),
                to_time=fc.args.get("to_time", ""),
            )
        else:
            result = {"error": f"Unknown function: {fc.name}"}

        print(f"[ToolCall] Round {round_num + 1}: Result: {result}")

        function_response = genai.types.Part.from_function_response(
            name=fc.name,
            response=result,
        )
        contents.append(genai.types.Content(parts=[function_response], role="tool"))

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=genai.types.GenerateContentConfig(tools=[_time_tool]),
        )
        contents.append(response.candidates[0].content)

    print(f"[ToolCall] Max rounds ({max_rounds}) reached — returning fallback")
    return response.text.strip() if response.text else "Sorry, kuch issue ho gaya yaar."
