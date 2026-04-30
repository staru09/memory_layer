def number_profile_lines(profile: str) -> str:
    """Add line numbers to profile for LLM reference.

    Returns numbered profile text like:
      [1] - Rampal is 24 years old
      [2] - Rampal works as an MLE
      ...
      [15] ## IMPLICIT TRAITS
      [16] - [Health-Conscious] — goes to gym 6 days a week
    """
    lines = profile.split("\n")
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1))


def apply_operations(profile: str, operations: list[dict]) -> str:
    """Apply add/update/delete operations to profile text programmatically."""
    lines = profile.split("\n")

    # Collect deletes and updates by line number
    deletes = set()
    updates = {}
    adds_explicit = []
    adds_implicit = []

    for op in operations:
        action = op.get("action", "none")
        if action == "none":
            continue
        elif action == "delete":
            line_num = op.get("line")
            if line_num and 1 <= line_num <= len(lines):
                deletes.add(line_num)
                print(f"    [op] DELETE line {line_num}: {lines[line_num - 1].strip()}")
        elif action == "update":
            line_num = op.get("line")
            new_fact = op.get("new_fact", "")
            if line_num and 1 <= line_num <= len(lines) and new_fact:
                updates[line_num] = new_fact
                print(f"    [op] UPDATE line {line_num}: {lines[line_num - 1].strip()} → {new_fact.strip()}")
        elif action == "add":
            fact = op.get("fact", "")
            section = op.get("section", "explicit")
            if fact:
                if section == "implicit":
                    adds_implicit.append(fact)
                else:
                    adds_explicit.append(fact)
                print(f"    [op] ADD ({section}): {fact.strip()}")

    # Apply updates and deletes
    new_lines = []
    for i, line in enumerate(lines, 1):
        if i in deletes:
            continue
        elif i in updates:
            new_lines.append(updates[i])
        else:
            new_lines.append(line)

    # Find the implicit traits header to know where to insert
    implicit_idx = None
    for i, line in enumerate(new_lines):
        if "## IMPLICIT TRAITS" in line:
            implicit_idx = i
            break

    # Add new explicit facts (before ## IMPLICIT TRAITS, or at the end)
    if adds_explicit:
        if implicit_idx is not None:
            insert_at = implicit_idx
            while insert_at > 0 and new_lines[insert_at - 1].strip() == "":
                insert_at -= 1
            for fact in reversed(adds_explicit):
                new_lines.insert(insert_at, fact)
            # Recalculate implicit_idx after insertion
            implicit_idx = None
            for i, line in enumerate(new_lines):
                if "## IMPLICIT TRAITS" in line:
                    implicit_idx = i
                    break
        else:
            for fact in adds_explicit:
                new_lines.append(fact)

    # Add new implicit traits (at the end)
    if adds_implicit:
        if implicit_idx is None:
            new_lines.append("")
            new_lines.append("## IMPLICIT TRAITS")
        for fact in adds_implicit:
            new_lines.append(fact)

    return "\n".join(new_lines)
