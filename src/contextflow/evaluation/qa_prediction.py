from __future__ import annotations


def parse_cacheblend_generation(text: str) -> str:
    """Parse a generated QA answer using CacheBlend's official minimal rules."""

    first_line = text.lstrip("\n").split("\n")[0].strip()
    if first_line.startswith("Yes") or first_line.startswith("yes"):
        return "Yes"

    tokens = first_line.split()
    if tokens and (tokens[0].startswith("No") or tokens[0].startswith("no")):
        return "No"
    return first_line
