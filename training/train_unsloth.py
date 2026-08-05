"""Fine-tune an open-weight LLM on the Bisaya instruction dataset using
Unsloth + LoRA/QLoRA + TRL's SFTTrainer.

CUDA/NVIDIA only. For Apple Silicon, use training/train_mlx.py instead --
`cli.py train` picks the right one automatically. This module intentionally
contains zero MLX-specific logic; mixing the two was the root cause of the
`processing_class must be PreTrainedTokenizerBase` crash this file used to
have.

Requires the `train` extra / requirements-train.txt (GPU box only):
    pip install -r requirements.txt -r requirements-train.txt

Usage:
    python -m training.train_unsloth --config configs/training.yaml
"""

from __future__ import annotations

import platform
from pathlib import Path

import typer

from common import load_yaml_config, setup_logging
from training.common import format_prompt, write_backend_marker

logger = setup_logging("training")
app = typer.Typer(add_completion=False)


def _assert_cuda_backend() -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "train_unsloth.py requires torch + CUDA. On Apple Silicon, use "
            "`python cli.py train` (auto-selects training/train_mlx.py) instead."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA device visible on this machine (platform={platform.system()}). "
            "train_unsloth.py is CUDA-only; use training/train_mlx.py on Apple Silicon."
        )


def _load_model_and_tokenizer(cfg: dict):
    """Import unsloth lazily so this module can be imported (e.g. by tests
    that only check config plumbing) on machines without a GPU/unsloth."""
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model_cfg = cfg["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_id"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )
    return model, tokenizer


def _format_prompt(example: dict, cfg: dict) -> dict:
    data_cfg = cfg["data"]
    instruction = example[data_cfg["prompt_field"]]
    input_text = example.get(data_cfg["input_field"], "") or ""
    output_text = example[data_cfg["output_field"]]
    return {"text": format_prompt(instruction, input_text, output_text)}


def train(config_path: Path = Path("configs/training.yaml")) -> Path:
    _assert_cuda_backend()

    from datasets import load_dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    cfg = load_yaml_config(config_path.name) if config_path.parent.name == "configs" else None
    if cfg is None:
        import yaml  # noqa: PLC0415

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model, tokenizer = _load_model_and_tokenizer(cfg)

    data_cfg = cfg["data"]
    dataset = load_dataset(data_cfg["train_dataset_id"])
    train_ds = dataset[data_cfg["train_split"]].map(lambda ex: _format_prompt(ex, cfg))
    eval_ds = (
        dataset[data_cfg["eval_split"]].map(lambda ex: _format_prompt(ex, cfg))
        if data_cfg.get("eval_split") in dataset
        else None
    )

    ta = cfg["training_args"]
    sft_config = SFTConfig(
        output_dir=ta["output_dir"],
        per_device_train_batch_size=ta["per_device_train_batch_size"],
        gradient_accumulation_steps=ta["gradient_accumulation_steps"],
        warmup_ratio=ta["warmup_ratio"],
        num_train_epochs=ta["num_train_epochs"],
        learning_rate=ta["learning_rate"],
        fp16=ta["fp16"],
        bf16=ta["bf16"],
        logging_steps=ta["logging_steps"],
        optim=ta["optim"],
        weight_decay=ta["weight_decay"],
        lr_scheduler_type=ta["lr_scheduler_type"],
        seed=ta["seed"],
        save_strategy=ta["save_strategy"],
        save_steps=ta["save_steps"],
        save_total_limit=ta["save_total_limit"],
        eval_strategy=ta["eval_strategy"],
        eval_steps=ta["eval_steps"],
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        packing=data_cfg["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    logger.info(
        "starting training run (resume_from_checkpoint=%s)", ta.get("resume_from_checkpoint")
    )
    trainer.train(resume_from_checkpoint=ta.get("resume_from_checkpoint"))

    out_cfg = cfg["output"]
    adapter_dir = Path(out_cfg["save_adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_backend_marker(adapter_dir, "unsloth", cfg["model"]["base_model_id"])
    logger.info("saved LoRA adapter to %s", adapter_dir)

    if out_cfg.get("merge_and_save_16bit"):
        merged_dir = Path(out_cfg["save_merged_dir"])
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("saved merged 16-bit model to %s", merged_dir)

    if out_cfg.get("push_adapter_to_hub"):
        model.push_to_hub(out_cfg["hub_repo_id"])
        tokenizer.push_to_hub(out_cfg["hub_repo_id"])
        logger.info("pushed adapter to hub: %s", out_cfg["hub_repo_id"])

    return adapter_dir


@app.command()
def main(config: Path = typer.Option(Path("configs/training.yaml"), "--config")) -> None:
    train(config)


if __name__ == "__main__":
    app()
