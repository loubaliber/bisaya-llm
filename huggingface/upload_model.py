"""Push the trained LoRA adapter (and/or merged model) to the Hugging Face
Hub, along with an auto-generated model card covering training metadata,
hyperparameters, evaluation metrics, and known limitations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

from common import fail_fast, load_yaml_config, setup_logging
from training.common import require_trained_adapter

load_dotenv()
logger = setup_logging("huggingface")
app = typer.Typer(add_completion=False)


def build_model_card(
    training_cfg: dict, eval_summary: dict | None, backend: str = "unsloth"
) -> str:
    model_cfg = training_cfg["model"]
    lora_cfg = training_cfg["lora"]
    ta = training_cfg["training_args"]

    trainer_note = (
        "[Unsloth](https://github.com/unslothai/unsloth) + TRL's SFTTrainer (CUDA)"
        if backend == "unsloth"
        else "[mlx-lm](https://github.com/ml-explore/mlx-examples)'s LoRA trainer "
        "(Apple Silicon / MLX)"
    )
    lines = [
        f"# {model_cfg['base_model_id']} fine-tuned for Cebuano (Bisaya)",
        "",
        f"LoRA/QLoRA adapter trained with {trainer_note} "
        "on a Cebuano (Bisaya) instruction-tuning dataset. See the repository's "
        "`docs/LEGAL_NOTICE.md` for data provenance and licensing caveats.",
        "",
        "## Base model",
        f"- `{model_cfg['base_model_id']}`",
        f"- max_seq_length: {model_cfg['max_seq_length']}",
        f"- 4-bit quantized: {model_cfg['load_in_4bit']}",
        "",
        "## LoRA hyperparameters",
        f"- r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, dropout={lora_cfg['lora_dropout']}",
        f"- target_modules: {', '.join(lora_cfg['target_modules'])}",
        "",
        "## Training hyperparameters",
        f"- epochs: {ta['num_train_epochs']}",
        f"- learning_rate: {ta['learning_rate']}",
        f"- per_device_train_batch_size: {ta['per_device_train_batch_size']}",
        f"- gradient_accumulation_steps: {ta['gradient_accumulation_steps']}",
        f"- optimizer: {ta['optim']}",
        f"- lr_scheduler: {ta['lr_scheduler_type']}",
        f"- seed: {ta['seed']}",
        "",
    ]

    if eval_summary:
        lines += [
            "## Evaluation",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        for k, v in eval_summary.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += [
        "## Known limitations",
        "",
        "- Trained on a small, dictionary-derived instruction set; broader",
        "  fluent generation and long-form Bisaya text quality are not guaranteed.",
        "- Automatic eval metrics (exact-match / token-overlap) are weak proxies;",
        "  see `output/eval/human_eval_checklist.md` for the intended human review process.",
        "- Underlying dictionary source data has licensing caveats -- see",
        "  `docs/LEGAL_NOTICE.md` in the source repository.",
        "- Not evaluated for safety-critical, medical, or legal use.",
    ]
    return "\n".join(lines)


def main(
    adapter_dir: Path = Path("output/checkpoints/lora_adapter"),
    repo_id: str | None = None,
    eval_summary_path: Path = Path("output/eval/eval_samples.json"),
    private: bool = True,
) -> None:
    """Plain function with real defaults -- safe to call directly from
    cli.py, not just from the Typer CLI below."""
    training_cfg = load_yaml_config("training.yaml")
    repo_id = repo_id or training_cfg["output"]["hub_repo_id"]

    try:
        marker = require_trained_adapter(adapter_dir)
    except FileNotFoundError as exc:
        fail_fast(str(exc))
        return
    logger.info("uploading %s adapter trained from %s", marker["backend"], marker["base_model_id"])

    token = os.getenv("HF_TOKEN")
    if not token:
        fail_fast("HF_TOKEN not set. Copy .env.example to .env and fill it in.")

    eval_summary = None
    if eval_summary_path.exists():
        eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8")).get("summary")

    card = build_model_card(training_cfg, eval_summary, marker["backend"])
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model")
    logger.info("uploaded adapter to https://huggingface.co/%s (private=%s)", repo_id, private)


@app.command(name="main")
def _cli(
    adapter_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--adapter-dir"),
    repo_id: str | None = typer.Option(None, "--repo-id"),
    eval_summary_path: Path = typer.Option(
        Path("output/eval/eval_samples.json"), "--eval-summary-path"
    ),
    private: bool = typer.Option(True, "--private/--public"),
) -> None:
    main(adapter_dir, repo_id, eval_summary_path, private)


if __name__ == "__main__":
    app()
