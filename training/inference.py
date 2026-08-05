"""Run inference with the fine-tuned model. Backend (MLX vs Unsloth) is
read from `backend.json` inside the adapter directory (written by
training/train_mlx.py or training/train_unsloth.py) -- callers never need
to know or guess which backend trained a given adapter."""

from __future__ import annotations

from pathlib import Path

import typer

from common import load_yaml_config, setup_logging
from training.common import format_prompt, require_trained_adapter

logger = setup_logging("training")
app = typer.Typer(add_completion=False)


def load_model(model_dir: Path, max_seq_length: int, load_in_4bit: bool = True):
    """Load a trained adapter for inference, dispatching on backend.json.
    Returns (backend_name, model, tokenizer)."""
    marker = require_trained_adapter(model_dir)
    backend = marker["backend"]

    if backend == "mlx":
        from mlx_lm import load as mlx_load  # noqa: PLC0415

        model, tokenizer = mlx_load(marker["base_model_id"], adapter_path=str(model_dir))
        return "mlx", model, tokenizer

    if backend == "unsloth":
        from unsloth import FastLanguageModel  # noqa: PLC0415

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return "unsloth", model, tokenizer

    raise ValueError(f"Unknown backend '{backend}' in {model_dir}/backend.json")


def generate(
    backend_and_model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 128,
) -> str:
    """`backend_and_model` is either a bare model (unsloth, legacy call
    signature) or a (backend, model) tuple; both are accepted so evaluate.py
    and the demo apps don't need to change their call sites."""
    if isinstance(backend_and_model, tuple):
        backend, model = backend_and_model
    else:
        backend, model = "unsloth", backend_and_model

    prompt = format_prompt(instruction, input_text)

    if backend == "mlx":
        from mlx_lm import generate as mlx_generate  # noqa: PLC0415

        return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens).strip()

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded.split("### Response:\n", 1)[-1].strip()


def main(
    instruction: str,
    input_text: str = "",
    model_dir: Path = Path("output/checkpoints/lora_adapter"),
    config: Path = Path("configs/training.yaml"),
) -> str:
    """Plain function with real defaults -- safe to call directly from
    cli.py, not just from the Typer CLI below."""
    cfg = load_yaml_config(config.name)
    backend, model, tokenizer = load_model(model_dir, cfg["model"]["max_seq_length"])
    response = generate((backend, model), tokenizer, instruction, input_text)
    logger.info("Response: %s", response)
    return response


@app.command(name="main")
def _cli(
    instruction: str = typer.Argument(..., help="Instruction/prompt to send to the model"),
    input_text: str = typer.Option("", "--input", help="Optional input context"),
    model_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--model-dir"),
    config: Path = typer.Option(Path("configs/training.yaml"), "--config"),
) -> None:
    typer.echo(main(instruction, input_text, model_dir, config))


if __name__ == "__main__":
    app()
