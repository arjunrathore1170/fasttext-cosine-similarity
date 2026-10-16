import re

def tokenize(text: str) -> list[str]:
    """Lowercase text and return word-like tokens."""
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text.lower())
