import pytest

from training.common import (
    format_prompt,
    read_backend_marker,
    require_trained_adapter,
    write_backend_marker,
)


def test_format_prompt_without_input():
    prompt = format_prompt("Translate 'house'.", "", "Balay.")
    assert prompt == "### Instruction:\nTranslate 'house'.\n\n### Response:\nBalay."


def test_format_prompt_with_input():
    prompt = format_prompt("Fill in the blank.", "Naa ko sa ____.", "balay")
    assert "### Input:\nNaa ko sa ____." in prompt
    assert prompt.endswith("balay")


def test_format_prompt_train_unsloth_and_train_mlx_agree(tmp_path):
    """The whole point of training/common.py: both backends must produce
    byte-identical prompts for the same example."""
    from training.train_mlx import _to_mlx_jsonl
    from training.train_unsloth import _format_prompt

    example = {"instruction": "Unsay meaning sa 'kaon'?", "input": "", "output": "to eat"}
    data_cfg = {"prompt_field": "instruction", "input_field": "input", "output_field": "output"}

    unsloth_text = _format_prompt(example, {"data": data_cfg})["text"]
    mlx_text = _to_mlx_jsonl(example, data_cfg)["text"]
    assert unsloth_text == mlx_text


def test_write_and_read_backend_marker(tmp_path):
    adapter_dir = tmp_path / "adapter"
    write_backend_marker(adapter_dir, "mlx", "some/base-model")
    marker = read_backend_marker(adapter_dir)
    assert marker == {"backend": "mlx", "base_model_id": "some/base-model"}


def test_read_backend_marker_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_backend_marker(tmp_path / "does_not_exist")


def test_require_trained_adapter_missing_dir_raises_clear_error(tmp_path):
    missing = tmp_path / "output" / "checkpoints" / "lora_adapter"
    with pytest.raises(FileNotFoundError, match="No trained adapter found"):
        require_trained_adapter(missing)


def test_require_trained_adapter_empty_dir_raises(tmp_path):
    empty = tmp_path / "adapter"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No trained adapter found"):
        require_trained_adapter(empty)


def test_require_trained_adapter_success(tmp_path):
    adapter_dir = tmp_path / "adapter"
    write_backend_marker(adapter_dir, "unsloth", "meta-llama/Llama-3.1-8B")
    marker = require_trained_adapter(adapter_dir)
    assert marker["backend"] == "unsloth"
