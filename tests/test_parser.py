import json

from parser.parser import parse_entry
from schemas import RawEntry


def test_parse_manual_entry_reads_structured_json():
    payload = {
        "translation": "house",
        "part_of_speech": "noun",
        "example_sentences": ["Nindot ang balay."],
    }
    raw = RawEntry(
        word="balay",
        url="https://example.com/manual/balay",
        html=json.dumps(payload),
        source="manual",
    )
    parsed = parse_entry(raw)
    assert parsed is not None
    assert parsed.word == "balay"
    assert parsed.translation == "house"
    assert parsed.part_of_speech == "noun"
    assert parsed.example_sentences == ["Nindot ang balay."]


def test_parse_entry_returns_none_for_malformed_manual_json():
    raw = RawEntry(
        word="balay", url="https://example.com/manual/balay", html="not json", source="manual"
    )
    assert parse_entry(raw) is None


def test_parse_talkbisaya_extracts_from_data_attributes():
    html = """
    <html><body>
      <h1>balay</h1>
      <p data-field="translation">house</p>
      <p data-field="part-of-speech">noun</p>
      <p data-field="pronunciation">bah-LIGH</p>
      <p class="example-sentence">Nindot ang balay.</p>
    </body></html>
    """
    raw = RawEntry(
        word="balay",
        url="https://www.talkbisaya.com/dictionary/balay",
        html=html,
        source="talkbisaya",
    )
    parsed = parse_entry(raw)
    assert parsed is not None
    assert parsed.translation == "house"
    assert parsed.part_of_speech == "noun"
    assert parsed.pronunciation == "bah-LIGH"
    assert parsed.example_sentences == ["Nindot ang balay."]
