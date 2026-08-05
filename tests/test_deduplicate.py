from cleaning.deduplicate import drop_exact_duplicates, merge_cross_source_duplicates
from schemas import ParsedEntry


def _entry(word, source, **kwargs):
    return ParsedEntry(word=word, source_url=f"https://example.com/{word}", source=source, **kwargs)


def test_drop_exact_duplicates_same_word_same_source():
    entries = [
        _entry("balay", "talkbisaya"),
        _entry("balay", "talkbisaya"),
        _entry("balay", "manual"),
    ]
    result = drop_exact_duplicates(entries)
    assert len(result) == 2


def test_merge_cross_source_duplicates_unions_examples():
    a = _entry("balay", "talkbisaya", translation="house", example_sentences=["Nindot ang balay."])
    b = _entry("balay", "manual", translation="home, house", example_sentences=["Balay namo ni."])
    merged = merge_cross_source_duplicates([a, b])
    assert len(merged) == 1
    assert set(merged[0].example_sentences) == {"Nindot ang balay.", "Balay namo ni."}
    # base picked by longest translation/definition
    assert merged[0].translation == "home, house"


def test_merge_cross_source_duplicates_single_entry_passthrough():
    a = _entry("kaon", "talkbisaya", translation="to eat")
    assert merge_cross_source_duplicates([a]) == [a]
