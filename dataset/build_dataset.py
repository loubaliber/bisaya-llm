"""Assemble the cleaned master dataset into JSONL/Parquet/Arrow, split into
train/validation/test, and auto-generate a dataset card."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import DatasetDict

from common import load_yaml_config, setup_logging
from dataset.splitting import records_to_split_dataset
from schemas import CleanedEntry

logger = setup_logging("dataset")


def load_cleaned_entries(path: Path) -> list[CleanedEntry]:
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(CleanedEntry.model_validate_json(line))
    return entries


def to_hf_dataset(entries: list[CleanedEntry], cfg: dict) -> DatasetDict:
    return records_to_split_dataset(entries, cfg, shuffle=True)


def write_local_formats(dataset: DatasetDict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        split.to_json(str(out_dir / f"{split_name}.jsonl"))
        split.to_parquet(str(out_dir / f"{split_name}.parquet"))
    dataset.save_to_disk(str(out_dir / "arrow"))
    logger.info("wrote JSONL/Parquet/Arrow to %s", out_dir)


def build_dataset_card(cfg: dict, dataset: DatasetDict, source_counts: dict[str, int]) -> str:
    ds_cfg = cfg["dataset"]
    lines = [
        f"# {ds_cfg['name']}",
        "",
        ds_cfg["description"].strip(),
        "",
        "## Splits",
        "",
        "| split | rows |",
        "|---|---|",
    ]
    for split_name, split in dataset.items():
        lines.append(f"| {split_name} | {len(split)} |")

    lines += [
        "",
        "## Fields",
        "",
        "| field | type | description |",
        "|---|---|---|",
        "| id | string | stable per-record identifier |",
        "| word | string | Cebuano (Bisaya) headword |",
        "| translation | string | English gloss/translation |",
        "| part_of_speech | string? | grammatical category, if known |",
        "| pronunciation | string? | pronunciation guide, if known |",
        "| example_sentences | list[string] | example usage sentences |",
        "| synonyms | list[string] | synonyms, if any |",
        "| antonyms | list[string] | antonyms, if any |",
        "| related_words | list[string] | related headwords |",
        "| definition | string? | fuller definition text, if distinct from translation |",
        "| source_url | string | provenance URL |",
        "| source | string | which spider/source produced this record |",
        "",
        "## Sources & Licensing",
        "",
        "See `docs/LEGAL_NOTICE.md` in the repository for the full analysis. "
        "Per-source record counts in this build:",
        "",
        "| source | records |",
        "|---|---|",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines += [
        "",
        f"**License:** `{ds_cfg['license']}` -- see LEGAL_NOTICE.md before treating this "
        "as freely redistributable.",
        "",
        "## Citation",
        "",
        "```bibtex",
        ds_cfg["citation"].strip(),
        "```",
    ]
    return "\n".join(lines)


def main(
    cleaned_path: Path = Path("output/cleaned/master.jsonl"),
    out_dir: Path = Path("output/datasets/bisaya-cebuano-dictionary"),
) -> None:
    cfg = load_yaml_config("dataset.yaml")
    entries = load_cleaned_entries(cleaned_path)
    if not entries:
        logger.warning("no cleaned entries found at %s -- run cleaning first", cleaned_path)
        return

    source_counts: dict[str, int] = {}
    for e in entries:
        source_counts[e.source] = source_counts.get(e.source, 0) + 1

    dataset = to_hf_dataset(entries, cfg)
    write_local_formats(dataset, out_dir)

    card = build_dataset_card(cfg, dataset, source_counts)
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    metadata = {
        "version": cfg["dataset"]["version"],
        "total_records": len(entries),
        "source_counts": source_counts,
        "quality_filters": cfg["quality_filters"],
    }
    (out_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("dataset build complete: %s", out_dir)


if __name__ == "__main__":
    main()
