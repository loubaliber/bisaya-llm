"""Turn CleanedEntry dictionary records into diverse instruction-tuning
examples (Alpaca-style: instruction/input/output), covering translation in
both directions, definition lookup, example-sentence generation, synonym
questions, and fill-in-the-blank tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import DatasetDict

from common import load_yaml_config, setup_logging
from dataset.splitting import records_to_split_dataset
from schemas import CleanedEntry, InstructionExample

logger = setup_logging("dataset")


def _ceb_to_en_templates(e: CleanedEntry) -> list[InstructionExample]:
    templates = [
        (f'Unsay meaning sa "{e.word}"?', f'"{e.word}" means "{e.translation}".'),
        (f'Translate "{e.word}" to English.', e.translation),
        (f'What does the Bisaya word "{e.word}" mean in English?', e.translation),
    ]
    return [
        InstructionExample(instruction=i, output=o, task_type="ceb_to_en", source_id=e.id)
        for i, o in templates
    ]


def _en_to_ceb_templates(e: CleanedEntry) -> list[InstructionExample]:
    templates = [
        (f'Translate "{e.translation}" into Bisaya.', e.word),
        (f'How do you say "{e.translation}" in Cebuano?', e.word),
    ]
    return [
        InstructionExample(instruction=i, output=o, task_type="en_to_ceb", source_id=e.id)
        for i, o in templates
    ]


def _example_sentence_templates(e: CleanedEntry) -> list[InstructionExample]:
    if not e.example_sentences:
        return []
    sentence = random.choice(e.example_sentences)
    return [
        InstructionExample(
            instruction=f'Use "{e.word}" in a sentence.',
            output=sentence,
            task_type="example_sentence",
            source_id=e.id,
        )
    ]


def _pos_templates(e: CleanedEntry) -> list[InstructionExample]:
    if not e.part_of_speech:
        return []
    return [
        InstructionExample(
            instruction=f'What part of speech is "{e.word}" in Cebuano?',
            output=e.part_of_speech,
            task_type="part_of_speech",
            source_id=e.id,
        )
    ]


def _synonym_templates(e: CleanedEntry) -> list[InstructionExample]:
    if not e.synonyms:
        return []
    return [
        InstructionExample(
            instruction=f'Give a Bisaya synonym for "{e.word}".',
            output=random.choice(e.synonyms),
            task_type="synonym",
            source_id=e.id,
        )
    ]


def _fill_in_blank_templates(e: CleanedEntry) -> list[InstructionExample]:
    if not e.example_sentences:
        return []
    sentence = random.choice(e.example_sentences)
    if e.word.lower() not in sentence.lower():
        return []
    blanked = sentence.replace(e.word, "____", 1)
    if blanked == sentence:
        return []
    return [
        InstructionExample(
            instruction="Fill in the blank with the correct Bisaya word.",
            input=blanked,
            output=e.word,
            task_type="fill_in_blank",
            source_id=e.id,
        )
    ]


_TEMPLATE_FUNCS = [
    _ceb_to_en_templates,
    _en_to_ceb_templates,
    _example_sentence_templates,
    _pos_templates,
    _synonym_templates,
    _fill_in_blank_templates,
]


def generate_examples(entries: list[CleanedEntry], cfg: dict) -> list[InstructionExample]:
    random.seed(cfg["instruction_dataset"]["shuffle_seed"])
    per_entry_cap = cfg["instruction_dataset"]["templates_per_entry"]

    examples: list[InstructionExample] = []
    for e in entries:
        candidates: list[InstructionExample] = []
        for fn in _TEMPLATE_FUNCS:
            candidates.extend(fn(e))
        random.shuffle(candidates)
        examples.extend(candidates[:per_entry_cap])

    random.shuffle(examples)
    logger.info("generated %d instruction examples from %d entries", len(examples), len(entries))
    return examples


def to_hf_dataset(examples: list[InstructionExample], cfg: dict) -> DatasetDict:
    return records_to_split_dataset(examples, cfg, shuffle=False)


def main(
    cleaned_path: Path = Path("output/cleaned/master.jsonl"),
    out_dir: Path = Path("output/datasets/bisaya-instruction-tuning"),
) -> None:
    cfg = load_yaml_config("dataset.yaml")
    entries = [
        CleanedEntry.model_validate_json(line)
        for line in cleaned_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        logger.warning("no cleaned entries found at %s", cleaned_path)
        return

    examples = generate_examples(entries, cfg)
    dataset = to_hf_dataset(examples, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        split.to_json(str(out_dir / f"{split_name}.jsonl"))
    dataset.save_to_disk(str(out_dir / "arrow"))

    task_type_counts: dict[str, int] = {}
    for e in examples:
        task_type_counts[e.task_type] = task_type_counts.get(e.task_type, 0) + 1
    (out_dir / "build_metadata.json").write_text(
        json.dumps({"total": len(examples), "task_type_counts": task_type_counts}, indent=2),
        encoding="utf-8",
    )
    logger.info("instruction dataset build complete: %s", out_dir)


if __name__ == "__main__":
    main()
