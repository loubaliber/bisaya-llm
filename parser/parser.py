"""Parse raw scraped HTML (RawEntry) into structured fields (ParsedEntry).

Uses selectolax for speed with a BeautifulSoup fallback for messier markup.
Each source gets its own extraction function since site markup differs; a
dispatch table picks the right one based on RawEntry.source.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser

from common import setup_logging
from schemas import ParsedEntry, RawEntry

logger = setup_logging("parser")

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def _split_list_field(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,;/]| and ", text)
    return [p.strip() for p in parts if p.strip()]


def parse_talkbisaya(raw: RawEntry) -> ParsedEntry:
    """Extract fields from a talkbisaya.com/dictionary/<word> page.

    talkbisaya.com is a Next.js site; the entry page renders a definition
    block, an optional pronunciation guide, part-of-speech tag, example
    sentence(s), and a "related words" section.
    """
    tree = HTMLParser(raw.html)

    def text_of(selector: str) -> str | None:
        node = tree.css_first(selector)
        return _clean_text(node.text(deep=True)) if node else None

    translation = text_of("[data-field='translation']") or text_of("h1 + p")
    pos = text_of("[data-field='part-of-speech']") or text_of(".pos, .part-of-speech")
    pronunciation = text_of("[data-field='pronunciation']") or text_of(".pronunciation")
    definition = text_of("[data-field='definition']") or text_of(".definition")

    examples: list[str] = []
    for node in tree.css(".example-sentence, [data-field='example']"):
        cleaned = _clean_text(node.text(deep=True))
        if cleaned:
            examples.append(cleaned)

    related_text = text_of("[data-field='related-words']") or text_of(".related-words")
    synonyms_text = text_of("[data-field='synonyms']") or text_of(".synonyms")
    antonyms_text = text_of("[data-field='antonyms']") or text_of(".antonyms")

    if translation is None:
        # Fallback: BeautifulSoup heuristic scan for a short English gloss
        # near the headword, for markup variants selectolax selectors miss.
        soup = BeautifulSoup(raw.html, "html.parser")
        header = soup.find(["h1", "h2"])
        if header:
            sibling = header.find_next(["p", "span"])
            translation = _clean_text(sibling.get_text()) if sibling else None

    return ParsedEntry(
        word=raw.word,
        translation=translation,
        part_of_speech=pos,
        pronunciation=pronunciation,
        example_sentences=examples,
        synonyms=_split_list_field(synonyms_text),
        antonyms=_split_list_field(antonyms_text),
        related_words=_split_list_field(related_text),
        definition=definition,
        source_url=raw.url,
        source=raw.source,
    )


def parse_binisaya(raw: RawEntry) -> ParsedEntry:
    """Extract fields from a binisaya.com/cebuano/<word> page.

    Provided for completeness; see docs/LEGAL_NOTICE.md before enabling
    this source's scraper or publishing anything parsed from it.
    """
    soup = BeautifulSoup(raw.html, "html.parser")
    body = soup.find(class_="entry") or soup.find("body")
    text = _clean_text(body.get_text(" ")) if body else None

    return ParsedEntry(
        word=raw.word,
        translation=None,
        definition=text,
        source_url=raw.url,
        source=raw.source,
    )


def parse_manual(raw: RawEntry) -> ParsedEntry:
    """`manual` source entries store pre-structured JSON in the `html` field
    (allows bypassing the scraper entirely for hand-compiled/permissively
    licensed data -- see configs/scraper.yaml `sources.manual`)."""
    payload = json.loads(raw.html)
    return ParsedEntry(
        word=raw.word,
        translation=payload.get("translation"),
        part_of_speech=payload.get("part_of_speech"),
        pronunciation=payload.get("pronunciation"),
        example_sentences=payload.get("example_sentences", []),
        synonyms=payload.get("synonyms", []),
        antonyms=payload.get("antonyms", []),
        related_words=payload.get("related_words", []),
        definition=payload.get("definition"),
        source_url=raw.url,
        source=raw.source,
    )


_PARSERS: dict[str, Callable[[RawEntry], ParsedEntry]] = {
    "talkbisaya": parse_talkbisaya,
    "binisaya": parse_binisaya,
    "pinoydictionary": parse_binisaya,  # structurally similar enough for a first pass
    "manual": parse_manual,
}


def parse_entry(raw: RawEntry) -> ParsedEntry | None:
    parser_fn = _PARSERS.get(raw.source)
    if parser_fn is None:
        logger.warning("no parser registered for source=%s", raw.source)
        return None
    try:
        return parser_fn(raw)
    except Exception as exc:  # noqa: BLE001 - one bad record shouldn't kill the batch
        logger.error("failed to parse %s (%s): %s", raw.word, raw.url, exc)
        return None


def iter_raw_entries(path: Path) -> Iterator[RawEntry]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield RawEntry.model_validate_json(line)


def parse_file(raw_path: Path, out_path: Path) -> int:
    """Parse every RawEntry in `raw_path` and write ParsedEntry JSONL to `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for raw in iter_raw_entries(raw_path):
            parsed = parse_entry(raw)
            if parsed is None:
                continue
            out_f.write(parsed.model_dump_json() + "\n")
            count += 1
    logger.info("parsed %d/%s entries from %s -> %s", count, raw_path.name, raw_path, out_path)
    return count
