"""Render model-selected references from evidence, never model-written numbers."""
import math
import re

REFERENCE = re.compile(r"\{\{fact:(f[0-9]+)\}\}")
BLOCKED = {"customer_id", "customer_ids", "customerid", "email", "phone"}


def evidence_references(evidence, limit=120):
    result = {}

    def walk(value, path):
        if len(result) >= limit:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() not in BLOCKED:
                    walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                return
            if len(str(value)) <= 160:
                result[f"f{len(result)}"] = {"path": path, "value": value}

    walk(evidence, "")
    return result


def render_evidence_text(text, references):
    """Numbers must come through references; include paths to retain field identity.

    This verifies provenance, not the truth of every surrounding natural-language claim.
    """
    remaining = REFERENCE.sub("", text)
    if "{{" in remaining or "}}" in remaining or re.search(r"\d|[%£$€]", remaining):
        raise ValueError("Narrative contains unsupported numeric text or malformed evidence references.")

    def replace(match):
        key = match.group(1)
        if key not in references:
            raise ValueError("Narrative references evidence that does not exist.")
        item = references[key]
        # Escape evidence as plain Markdown text, preventing injected links/headings.
        value = f"{item['path']} = {item['value']}"
        return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", value).replace("\n", " ")

    return REFERENCE.sub(replace, text)
