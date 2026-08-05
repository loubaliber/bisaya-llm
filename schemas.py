"""Pydantic data contracts shared across pipeline stages.

Pipeline: scraper -> RawEntry -> parser -> ParsedEntry -> cleaning -> CleanedEntry
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SourceName = Literal["talkbisaya", "binisaya", "pinoydictionary", "manual"]


class RawEntry(BaseModel):
    """Exactly what the scraper writes to output/raw/*.jsonl — raw HTML, untouched."""

    word: str
    url: str
    html: str
    source: SourceName
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("word")
    @classmethod
    def word_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("word must not be blank")
        return v.strip()


class ParsedEntry(BaseModel):
    """Structured fields extracted from raw HTML by parser/parser.py."""

    word: str
    translation: str | None = None
    part_of_speech: str | None = None
    pronunciation: str | None = None
    example_sentences: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    antonyms: list[str] = Field(default_factory=list)
    related_words: list[str] = Field(default_factory=list)
    definition: str | None = None
    source_url: str
    source: SourceName

    @field_validator("word")
    @classmethod
    def word_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("word must not be blank")
        return v.strip()


class CleanedEntry(BaseModel):
    """Final, validated record that goes into the published dataset."""

    id: str
    word: str
    translation: str
    part_of_speech: str | None = None
    pronunciation: str | None = None
    example_sentences: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    antonyms: list[str] = Field(default_factory=list)
    related_words: list[str] = Field(default_factory=list)
    definition: str | None = None
    source_url: str
    source: SourceName

    @field_validator("word", "translation")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be blank")
        return v.strip()


class InstructionExample(BaseModel):
    """One instruction-tuning example (Alpaca-style schema)."""

    instruction: str
    input: str = ""
    output: str
    task_type: str
    source_id: str
