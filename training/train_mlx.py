"""Fine-tune an open-weight LLM on the Bisaya instruction dataset on Apple
Silicon using MLX. No `trl.SFTTrainer`, no PEFT, no bitsandbytes -- the MLX
tokenizer (`mlx_lm.tokenizer_utils.TokenizerWrapper`) is never handed to
anything from the Hugging Face training stack, which is what caused the
`processing_class must be PreTrainedTokenizerBase or ProcessorMixin` crash
in the old mixed pipeline.

Architecture choice: this shells out to `python -m mlx_lm.lora`, MLX's own
stable LoRA-training CLI, instead of calling `mlx_lm.tuner` internals
directly. `mlx_lm`'s internal trainer API has changed across releases;
its CLI is the part the maintainers keep backwards-compatible. This module
is responsible for (a) converting our instruction dataset into the
train/valid/test.jsonl layout `mlx_lm.lora` expects, (b) translating
configs/training.yaml into an mlx_lm config YAML, and (c) invoking it.

Requires (Apple Silicon only):
    pip install mlx mlx-lm

Usage:
    python -m training.train_mlx --config configs/training.yaml
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
import yaml

from common import load_yaml_config, setup_logging
from training.common import format_prompt, write_backend_marker

logger = setup_logging("training")
app = typer.Typer(add_completion=False)


def _assert_mlx_available() -> None:
    try:
        import mlx_lm  # noqa: F401,PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "mlx_lm is not installed. On Apple Silicon: `pip install mlx mlx-lm`. "
            "On CUDA/NVIDIA machines, use `python cli.py train` "
            "(auto-selects training/train_unsloth.py) instead."
        ) from exc


def _to_mlx_jsonl(example: dict, data_cfg: dict) -> dict:
    """mlx_lm.lora's default 'completions'-less mode just wants a `text`
    field per line (same convention as our Unsloth path), so both backends
    train on byte-identical prompts."""
    instruction = example[data_cfg["prompt_field"]]
    input_text = example.get(data_cfg["input_field"], "") or ""
    output_text = example[data_cfg["output_field"]]
    return {"text": format_prompt(instruction, input_text, output_text)}


def prepare_mlx_dataset(cfg: dict, out_dir: Path) -> Path:
    """Materialize output/mlx_data/{train,valid,test}.jsonl from the HF
    instruction dataset referenced in configs/training.yaml `data`."""
    from datasets import load_dataset  # noqa: PLC0415

    data_cfg = cfg["data"]
    dataset = load_dataset(data_cfg["train_dataset_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": data_cfg["train_split"],
        "valid": data_cfg.get("eval_split", "validation"),
    }
    for mlx_name, hf_split in split_map.items():
        if hf_split not in dataset:
            logger.warning("split '%s' not found in dataset, skipping %s.jsonl", hf_split, mlx_name)
            continue
        out_path = out_dir / f"{mlx_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for example in dataset[hf_split]:
                f.write(json.dumps(_to_mlx_jsonl(example, data_cfg)) + "\n")
        logger.info("wrote %d examples to %s", len(dataset[hf_split]), out_path)

    # mlx_lm.lora requires a valid.jsonl even if we only have train/test;
    # fall back to a small slice of train if no validation split exists.
    if not (out_dir / "valid.jsonl").exists():
        train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        holdout = max(1, len(train_lines) // 20)
        valid_text = "\n".join(train_lines[:holdout]) + "\n"
        (out_dir / "valid.jsonl").write_text(valid_text, encoding="utf-8")
        logger.info(
            "no eval split in dataset -- carved %d examples out of train for valid.jsonl", holdout
        )

    return out_dir


def build_mlx_lora_config(cfg: dict, data_dir: Path, adapter_dir: Path) -> dict:
    """Translate configs/training.yaml into an mlx_lm.lora config dict.
    Field names follow `python -m mlx_lm.lora --help` as of mlx-lm 0.x;
    if a newer mlx-lm renames a flag, this is the one place to update."""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    ta = cfg["training_args"]

    return {
        "model": model_cfg["base_model_id"],
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(adapter_dir),
        "iters": _estimate_iters(ta),
        "batch_size": ta["per_device_train_batch_size"],
        "learning_rate": ta["learning_rate"],
        "num_layers": lora_cfg.get("mlx_num_layers", 16),
        "seed": ta["seed"],
        "save_every": ta.get("save_steps", 100),
        "steps_per_eval": ta.get("eval_steps", 100),
        "resume_adapter_file": ta.get("resume_from_checkpoint") or None,
        "grad_checkpoint": True,
        "lora_parameters": {
            "rank": lora_cfg["r"],
            "dropout": lora_cfg["lora_dropout"],
            "scale": lora_cfg["lora_alpha"] / max(lora_cfg["r"], 1),
        },
    }


def _estimate_iters(ta: dict) -> int:
    """mlx_lm.lora is step-count driven, not epoch driven; approximate
    epochs * (dataset size / batch size) if the caller hasn't set
    training_args.mlx_iters explicitly."""
    return ta.get("mlx_iters", 1000)


def train(config_path: Path = Path("configs/training.yaml")) -> Path:
    _assert_mlx_available()
    if config_path.parent.name == "configs":
        cfg = load_yaml_config(config_path.name)
    else:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_dir = Path("output/mlx_data")
    adapter_dir = Path(cfg["output"]["save_adapter_dir"])
    prepare_mlx_dataset(cfg, data_dir)

    mlx_config = build_mlx_lora_config(cfg, data_dir, adapter_dir)
    config_path_out = Path("output/mlx_lora_config.yaml")
    config_path_out.write_text(yaml.safe_dump(mlx_config, sort_keys=False), encoding="utf-8")
    logger.info("wrote mlx_lm.lora config to %s", config_path_out)

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(config_path_out)]
    logger.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"mlx_lm.lora exited with code {result.returncode}. See its output above for the "
            "actual training error (this wrapper does not swallow mlx_lm's own diagnostics)."
        )

    write_backend_marker(adapter_dir, "mlx", cfg["model"]["base_model_id"])
    logger.info("MLX LoRA adapter saved to %s", adapter_dir)
    return adapter_dir


@app.command()
def main(config: Path = typer.Option(Path("configs/training.yaml"), "--config")) -> None:
    train(config)


if __name__ == "__main__":
    app()
