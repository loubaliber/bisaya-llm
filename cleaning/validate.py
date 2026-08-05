"""Validate ParsedEntry records into final CleanedEntry records, applying
the quality_filters block from configs/dataset.yaml and generating a
stable `id` for each surviving record."""

from __future__ import annotations

import hashlib

from pydantic import ValidationError

from common import setup_logging
from schemas import CleanedEntry, ParsedEntry

logger = setup_logging("cleaning")


def _make_id(entry: ParsedEntry) -> str:
    digest = hashlib.sha1(f"{entry.source}:{entry.word.lower()}".encode()).hexdigest()[:12]
    return f"{entry.source}-{digest}"


def passes_quality_filters(entry: ParsedEntry, filters: dict) -> bool:
    word_len = len(entry.word)
    if not (filters["min_word_chars"] <= word_len <= filters["max_word_chars"]):
        return False

    if filters.get("require_translation") and not (entry.translation or entry.definition):
        return False

    definition_source = entry.translation or entry.definition or ""
    if len(definition_source) < filters["min_definition_chars"]:
        return False

    if filters.get("drop_html_leftovers"):
        for field_val in (entry.translation, entry.definition):
            if field_val and ("<" in field_val and ">" in field_val):
                return False

    return True


def to_cleaned_entry(entry: ParsedEntry) -> CleanedEntry | None:
    translation = entry.translation or entry.definition
    if not translation:
        return None
    try:
        return CleanedEntry(
            id=_make_id(entry),
            word=entry.word,
            translation=translation,
            part_of_speech=entry.part_of_speech,
            pronunciation=entry.pronunciation,
            example_sentences=entry.example_sentences,
            synonyms=entry.synonyms,
            antonyms=entry.antonyms,
            related_words=entry.related_words,
            definition=entry.definition,
            source_url=entry.source_url,
            source=entry.source,
        )
    except ValidationError as exc:
        logger.warning("dropping invalid entry %s: %s", entry.word, exc)
        return None


def validate_entries(entries: list[ParsedEntry], filters: dict) -> list[CleanedEntry]:
    cleaned: list[CleanedEntry] = []
    for entry in entries:
        if not passes_quality_filters(entry, filters):
            continue
        result = to_cleaned_entry(entry)
        if result is not None:
            cleaned.append(result)
    logger.info("validated %d/%d entries passed quality filters", len(cleaned), len(entries))
    return cleaned
