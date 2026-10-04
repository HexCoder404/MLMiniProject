"""Shared text normalization. Preserve case and punctuation for linguistic features."""
import html
import re
import unicodedata


def clean_text(value):
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_content(value):
    return " ".join(re.findall(r"\w+", clean_text(value).lower()))
