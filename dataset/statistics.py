"""Compute and print/save summary statistics about the cleaned dataset."""

from __future__ import annotations

import json
import statistics as pystats
from pathlib import Path

from rich.console import Console
from rich.table import Table

from schemas import CleanedEntry

console = Console()


def compute_statistics(entries: list[CleanedEntry]) -> dict:
    word_lengths = [len(e.word) for e in entries]
    translation_lengths = [len(e.translation) for e in entries]
    with_examples = sum(1 for e in entries if e.example_sentences)
    with_pos = sum(1 for e in entries if e.part_of_speech)
    with_pron = sum(1 for e in entries if e.pronunciation)
    source_counts: dict[str, int] = {}
    for e in entries:
        source_counts[e.source] = source_counts.get(e.source, 0) + 1

    return {
        "total_entries": len(entries),
        "unique_words": len({e.word.lower() for e in entries}),
        "avg_word_length": round(pystats.fmean(word_lengths), 2) if word_lengths else 0,
        "avg_translation_length": round(pystats.fmean(translation_lengths), 2)
        if translation_lengths
        else 0,
        "pct_with_example_sentences": round(100 * with_examples / len(entries), 1)
        if entries
        else 0,
        "pct_with_part_of_speech": round(100 * with_pos / len(entries), 1) if entries else 0,
        "pct_with_pronunciation": round(100 * with_pron / len(entries), 1) if entries else 0,
        "source_counts": source_counts,
    }


def print_statistics(stats: dict) -> None:
    table = Table(title="Dataset Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in stats.items():
        if isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items())
        table.add_row(key, str(value))
    console.print(table)


def main(cleaned_path: Path = Path("output/cleaned/master.jsonl")) -> None:
    entries = [
        CleanedEntry.model_validate_json(line)
        for line in cleaned_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = compute_statistics(entries)
    print_statistics(stats)
    out_path = cleaned_path.parent / "statistics.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    console.print(f"[green]Statistics written to {out_path}[/green]")


if __name__ == "__main__":
    main()
