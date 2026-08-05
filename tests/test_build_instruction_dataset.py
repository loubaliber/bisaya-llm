from dataset.build_instruction_dataset import generate_examples
from schemas import CleanedEntry

CFG = {"instruction_dataset": {"templates_per_entry": 6, "shuffle_seed": 42}}


def _entry(**overrides):
    base = dict(
        id="manual-abc123",
        word="balay",
        translation="house",
        part_of_speech="noun",
        example_sentences=["Nindot ang balay."],
        synonyms=["puy-anan"],
        source_url="https://example.com/balay",
        source="manual",
    )
    base.update(overrides)
    return CleanedEntry(**base)


def test_generate_examples_produces_expected_task_types():
    entries = [_entry()]
    examples = generate_examples(entries, CFG)
    task_types = {e.task_type for e in examples}
    assert "ceb_to_en" in task_types
    assert "en_to_ceb" in task_types
    assert all(e.source_id == "manual-abc123" for e in examples)


def test_generate_examples_respects_per_entry_cap():
    entries = [_entry()]
    cfg = {"instruction_dataset": {"templates_per_entry": 2, "shuffle_seed": 42}}
    examples = generate_examples(entries, cfg)
    assert len(examples) <= 2


def test_generate_examples_skips_missing_optional_fields():
    entries = [_entry(example_sentences=[], synonyms=[], part_of_speech=None)]
    examples = generate_examples(entries, CFG)
    task_types = {e.task_type for e in examples}
    assert "example_sentence" not in task_types
    assert "synonym" not in task_types
    assert "part_of_speech" not in task_types
