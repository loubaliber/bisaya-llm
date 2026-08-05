from cleaning.normalize import normalize_entry, normalize_field, strip_html_leftovers
from schemas import ParsedEntry


def test_strip_html_leftovers_removes_tags_and_entities():
    dirty = "balay &nbsp;<b>house</b>&amp; home"
    cleaned = strip_html_leftovers(dirty)
    assert "<b>" not in cleaned
    assert "&nbsp;" not in cleaned


def test_normalize_field_collapses_whitespace():
    assert normalize_field("  balay   house  ") == "balay house"


def test_normalize_field_none_passthrough():
    assert normalize_field(None) is None


def test_normalize_entry_normalizes_all_text_fields():
    entry = ParsedEntry(
        word="  Balay ",
        translation="<i>house</i>  ",
        example_sentences=["  Naa   ko sa balay.  "],
        source_url="https://example.com/balay",
        source="manual",
    )
    normalized = normalize_entry(entry)
    assert normalized.word == "Balay"
    assert normalized.translation == "house"
    assert normalized.example_sentences == ["Naa ko sa balay."]
