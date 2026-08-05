"""Unicode/whitespace/HTML-leftover normalization applied to every field."""

from __future__ import annotations

import html
import re
import unicodedata

from schemas import ParsedEntry

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_STRAY_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")


def normalize_unicode(text: str) -> str:
    """NFC-normalize and strip control characters."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def strip_html_leftovers(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _STRAY_HTML_ENTITY_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_field(text: str | None) -> str | None:
    if text is None:
        return None
    text = strip_html_leftovers(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text or None


def normalize_entry(entry: ParsedEntry) -> ParsedEntry:
    """Return a new ParsedEntry with every text field normalized."""
    return entry.model_copy(
        update={
            "word": normalize_field(entry.word) or entry.word,
            "translation": normalize_field(entry.translation),
            "part_of_speech": normalize_field(entry.part_of_speech),
            "pronunciation": normalize_field(entry.pronunciation),
            "definition": normalize_field(entry.definition),
            "example_sentences": [
                s for s in (normalize_field(s) for s in entry.example_sentences) if s
            ],
            "synonyms": [s for s in (normalize_field(s) for s in entry.synonyms) if s],
            "antonyms": [s for s in (normalize_field(s) for s in entry.antonyms) if s],
            "related_words": [s for s in (normalize_field(s) for s in entry.related_words) if s],
        }
    )
