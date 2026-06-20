from __future__ import annotations

import re


HEADER_FOOTER_PATTERNS = [
    r"Page\s+\d+\s+of\s+\d+",
    r"CONFIDENTIAL",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_repeated_noise(text: str) -> str:
    cleaned = text
    for pattern in HEADER_FOOTER_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def basic_clean(text: str) -> str:
    """Readable cleaning that preserves legal and procurement meaning."""
    cleaned = remove_repeated_noise(text)
    cleaned = normalize_whitespace(cleaned)
    return cleaned

