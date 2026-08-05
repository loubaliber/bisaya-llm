from cleaning.validate import passes_quality_filters, to_cleaned_entry, validate_entries
from schemas import ParsedEntry

FILTERS = {
    "min_word_chars": 1,
    "max_word_chars": 60,
    "min_definition_chars": 2,
    "require_translation": True,
    "drop_duplicate_words": True,
    "drop_html_leftovers": True,
}


def test_passes_quality_filters_rejects_missing_translation():
    entry = ParsedEntry(word="balay", source_url="https://example.com/balay", source="manual")
    assert not passes_quality_filters(entry, FILTERS)


def test_passes_quality_filters_rejects_html_leftovers():
    entry = ParsedEntry(
        word="balay",
        translation="<b>house</b>",
        source_url="https://example.com/balay",
        source="manual",
    )
    assert not passes_quality_filters(entry, FILTERS)


def test_passes_quality_filters_accepts_valid_entry():
    entry = ParsedEntry(
        word="balay",
        translation="house",
        source_url="https://example.com/balay",
        source="manual",
    )
    assert passes_quality_filters(entry, FILTERS)


def test_to_cleaned_entry_generates_stable_id():
    entry = ParsedEntry(
        word="balay",
        translation="house",
        source_url="https://example.com/balay",
        source="manual",
    )
    cleaned_a = to_cleaned_entry(entry)
    cleaned_b = to_cleaned_entry(entry)
    assert cleaned_a is not None
    assert cleaned_a.id == cleaned_b.id
    assert cleaned_a.id.startswith("manual-")


def test_validate_entries_filters_and_converts():
    good = ParsedEntry(word="balay", translation="house", source_url="u", source="manual")
    missing_translation = ParsedEntry(word="wala", source_url="u2", source="manual")
    result = validate_entries([good, missing_translation], FILTERS)
    assert len(result) == 1
    assert result[0].word == "balay"
