"""Duplicate detection and removal.

Two levels:
1. Exact-key dedup: same normalized (word, source) pair.
2. Cross-source dedup: same normalized word from multiple sources gets
   merged into a single record (union of examples/synonyms/etc.), keeping
   the first-seen source_url as canonical and recording the rest.
"""

from __future__ import annotations

from schemas import ParsedEntry


def _norm_key(word: str) -> str:
    return word.strip().lower()


def drop_exact_duplicates(entries: list[ParsedEntry]) -> list[ParsedEntry]:
    seen: set[tuple[str, str]] = set()
    result: list[ParsedEntry] = []
    for entry in entries:
        key = (_norm_key(entry.word), entry.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


def merge_cross_source_duplicates(entries: list[ParsedEntry]) -> list[ParsedEntry]:
    """Merge entries that share the same normalized headword across sources.
    Keeps the entry with the most complete `translation`/`definition` as base
    and unions list fields from the others."""
    buckets: dict[str, list[ParsedEntry]] = {}
    for entry in entries:
        buckets.setdefault(_norm_key(entry.word), []).append(entry)

    merged: list[ParsedEntry] = []
    for group in buckets.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        base = max(
            group,
            key=lambda e: (len(e.translation or ""), len(e.definition or "")),
        )
        examples = _union_preserve_order(g.example_sentences for g in group)
        synonyms = _union_preserve_order(g.synonyms for g in group)
        antonyms = _union_preserve_order(g.antonyms for g in group)
        related = _union_preserve_order(g.related_words for g in group)

        merged.append(
            base.model_copy(
                update={
                    "example_sentences": examples,
                    "synonyms": synonyms,
                    "antonyms": antonyms,
                    "related_words": related,
                }
            )
        )
    return merged


def _union_preserve_order(lists: object) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:  # type: ignore[attr-defined]
        for item in lst:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out
