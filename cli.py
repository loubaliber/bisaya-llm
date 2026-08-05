"""Unified CLI for the bisaya-llm pipeline.

Examples:
    python cli.py scrape --source talkbisaya --limit 50
    python cli.py parse --source talkbisaya
    python cli.py clean
    python cli.py build-dataset
    python cli.py build-instructions
    python cli.py stats
    python cli.py upload-dataset --i-have-permission
    python cli.py train                 # auto-picks MLX (Apple Silicon) or Unsloth (CUDA)
    python cli.py evaluate
    python cli.py inference "Translate 'house' into Bisaya."
    python cli.py upload-model
"""

from __future__ import annotations

from pathlib import Path

import typer

from cleaning.deduplicate import drop_exact_duplicates, merge_cross_source_duplicates
from cleaning.normalize import normalize_entry
from cleaning.validate import validate_entries
from common import fail_fast, load_yaml_config, setup_logging
from schemas import ParsedEntry

logger = setup_logging("cli")
app = typer.Typer(add_completion=False, help="Bisaya (Cebuano) NLP dataset + fine-tuning pipeline")


@app.command()
def scrape(
    source: str = typer.Option("talkbisaya", help="talkbisaya | binisaya | pinoydictionary"),
    limit: int | None = typer.Option(None, help="Max number of entries to fetch"),
    allow_disabled_source: bool = typer.Option(
        False, help="Required to run a source disabled in configs/scraper.yaml"
    ),
) -> None:
    """Scrape one source's dictionary entries into output/raw/<source>.jsonl."""
    cfg = load_yaml_config("scraper.yaml")
    src_cfg = cfg["sources"].get(source)
    if src_cfg is None:
        fail_fast(f"Unknown source '{source}'. Options: {list(cfg['sources'])}")

    if not src_cfg.get("enabled", False) and not allow_disabled_source:
        fail_fast(
            f"Source '{source}' is disabled in configs/scraper.yaml "
            f"(reason: {src_cfg.get('license_note', 'see docs/LEGAL_NOTICE.md')}). "
            "Re-run with --allow-disabled-source only after confirming you have "
            "the legal right to do so."
        )

    global_cfg = cfg["global"]
    common_kwargs = dict(
        base_url=src_cfg["base_url"],
        user_agent=global_cfg["user_agent"],
        rate_limit_seconds=src_cfg["rate_limit_seconds"],
        max_rate_limit_seconds=global_cfg["max_delay_seconds"],
        timeout_seconds=global_cfg["timeout_seconds"],
        max_retries=global_cfg["max_retries"],
        raw_output_dir=Path(global_cfg["raw_output_dir"]),
        checkpoint_every=global_cfg["checkpoint_every"],
    )

    if source == "talkbisaya":
        from scraper.talkbisaya import TalkBisayaSpider

        spider = TalkBisayaSpider(**common_kwargs)
    elif source == "binisaya":
        from scraper.binisaya import BinisayaSpider

        spider = BinisayaSpider(allow_disabled_source=allow_disabled_source, **common_kwargs)
    else:
        fail_fast(f"No spider implemented yet for source '{source}'")
        return

    with spider:
        spider.run(limit=limit)


@app.command()
def parse(
    source: str = typer.Option("talkbisaya"),
    raw_dir: Path = typer.Option(Path("output/raw")),
    out_dir: Path = typer.Option(Path("output/cleaned")),
) -> None:
    """Parse raw HTML for one source into structured ParsedEntry JSONL."""
    from parser.parser import parse_file

    raw_path = raw_dir / f"{source}.jsonl"
    if not raw_path.exists():
        fail_fast(f"No raw file at {raw_path} -- run `scrape` first")
    out_path = out_dir / f"{source}.parsed.jsonl"
    parse_file(raw_path, out_path)


@app.command()
def clean(
    parsed_dir: Path = typer.Option(Path("output/cleaned")),
    out_path: Path = typer.Option(Path("output/cleaned/master.jsonl")),
) -> None:
    """Normalize, deduplicate, and validate all *.parsed.jsonl files into one master file."""
    cfg = load_yaml_config("dataset.yaml")
    parsed_files = sorted(parsed_dir.glob("*.parsed.jsonl"))
    if not parsed_files:
        fail_fast(f"No *.parsed.jsonl files found in {parsed_dir} -- run `parse` first")

    entries: list[ParsedEntry] = []
    for path in parsed_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(ParsedEntry.model_validate_json(line))
    logger.info("loaded %d parsed entries from %d files", len(entries), len(parsed_files))

    entries = [normalize_entry(e) for e in entries]
    entries = drop_exact_duplicates(entries)
    if cfg["quality_filters"].get("drop_duplicate_words"):
        entries = merge_cross_source_duplicates(entries)

    cleaned = validate_entries(entries, cfg["quality_filters"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in cleaned:
            f.write(e.model_dump_json() + "\n")
    logger.info("wrote %d cleaned entries to %s", len(cleaned), out_path)


@app.command()
def build_dataset() -> None:
    """Build the JSONL/Parquet/Arrow dataset + dataset card from output/cleaned/master.jsonl."""
    from dataset.build_dataset import main as build_main

    build_main()


@app.command()
def build_instructions() -> None:
    """Generate the instruction-tuning dataset from output/cleaned/master.jsonl."""
    from dataset.build_instruction_dataset import main as build_instr_main

    build_instr_main()


@app.command()
def stats(cleaned_path: Path = typer.Option(Path("output/cleaned/master.jsonl"))) -> None:
    """Print and save dataset statistics."""
    from dataset.statistics import main as stats_main

    stats_main(cleaned_path)


@app.command()
def upload_dataset(i_have_permission: bool = typer.Option(False, "--i-have-permission")) -> None:
    """Push the dataset to the Hugging Face Hub (permission-gated, see docs/LEGAL_NOTICE.md)."""
    from huggingface.upload_dataset import main as upload_main

    upload_main(i_have_permission=i_have_permission)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/training.yaml")),
    backend: str | None = typer.Option(
        None, "--backend", help="Force 'mlx' or 'unsloth'; default: auto-detect from platform"
    ),
) -> None:
    """Fine-tune. Auto-selects MLX (Apple Silicon) or Unsloth (CUDA); override with --backend."""
    from training.common import detect_platform_backend

    chosen = backend or detect_platform_backend()
    logger.info("training backend: %s", chosen)

    if chosen == "mlx":
        from training.train_mlx import train as train_main
    elif chosen == "unsloth":
        from training.train_unsloth import train as train_main
    else:
        fail_fast(f"Unknown backend '{chosen}'. Use 'mlx' or 'unsloth'.")
        return

    train_main(config)


@app.command()
def evaluate(
    model_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--model-dir"),
    config: Path = typer.Option(Path("configs/training.yaml"), "--config"),
    eval_file: Path = typer.Option(
        Path("output/datasets/bisaya-instruction-tuning/test.jsonl"), "--eval-file"
    ),
    out_dir: Path = typer.Option(Path("output/eval"), "--out-dir"),
    max_examples: int = typer.Option(200, "--max-examples"),
) -> None:
    """Evaluate a trained adapter: automatic metrics + samples + human-eval checklist."""
    from training.evaluate import main as evaluate_main

    evaluate_main(model_dir, config, eval_file, out_dir, max_examples)


@app.command()
def inference(
    instruction: str = typer.Argument(...),
    input_text: str = typer.Option("", "--input"),
    model_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--model-dir"),
    config: Path = typer.Option(Path("configs/training.yaml"), "--config"),
) -> None:
    """Run a single inference call against a trained adapter."""
    from training.inference import main as inference_main

    typer.echo(inference_main(instruction, input_text, model_dir, config))


@app.command()
def upload_model(
    adapter_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--adapter-dir"),
    repo_id: str | None = typer.Option(None, "--repo-id"),
    private: bool = typer.Option(True, "--private/--public"),
) -> None:
    """Push the trained adapter (+ auto-generated model card) to the Hugging Face Hub."""
    from huggingface.upload_model import main as upload_model_main

    upload_model_main(adapter_dir=adapter_dir, repo_id=repo_id, private=private)


if __name__ == "__main__":
    app()
