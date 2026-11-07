"""Text normalisation helpers shared by matching and genre engineering."""

import ast
import re
import unicodedata

import pandas as pd


def parse_list(value):
    """Parse a stringified Python list (as stored in the 550K CSV)."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def normalize_text(value):
    """Lowercase, strip accents, '&'->'and', punctuation->space, collapse spaces."""
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("&", "and")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def remove_featuring(artist_key):
    """Drop a Billboard-style 'featuring ...' suffix from a normalised artist key."""
    return re.sub(r"\s+featuring\s+.*$", "", artist_key).strip()


def remove_title_feature_credit(title):
    """Strip a feature credit Spotify embeds in a title, keeping version/remix text."""
    if pd.isna(title):
        return ""
    title = str(title)
    # Parenthetical feature credit, preserving any text that follows it.
    title = re.sub(
        r"\s*[\(\[]\s*(?:feat\.?|featuring)\s+[^\)\]]+[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    )
    # Terminal feature credit only when it actually ends the title.
    title = re.sub(
        r"\s+(?:feat\.?|featuring)\s+[^-]+$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return title.strip()
