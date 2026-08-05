"""Evaluate the fine-tuned model against a held-out split.

Produces:
  - exact-match / token-overlap "translation accuracy" proxy metrics
  - a sample of generations for qualitative review
  - a human evaluation checklist (markdown) for native-speaker review
  - a simple inference latency/throughput benchmark

This is intentionally lightweight (no external MT-metric dependency like
sacrebleu is assumed installed); it is meant as a first-pass automatic
signal, not a substitute for native-speaker review.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from common import load_yaml_config, setup_logging
from training.inference import generate, load_model

logger = setup_logging("training")
app = typer.Typer(add_completion=False)


def token_overlap_score(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def evaluate_split(
    backend: str, model, tokenizer, eval_examples: list[dict], max_examples: int = 200
) -> dict:
    results = []
    exact_matches = 0
    overlap_scores = []

    for example in eval_examples[:max_examples]:
        prediction = generate(
            (backend, model), tokenizer, example["instruction"], example.get("input", "")
        )
        reference = example["output"]
        is_exact = prediction.strip().lower() == reference.strip().lower()
        overlap = token_overlap_score(prediction, reference)

        exact_matches += int(is_exact)
        overlap_scores.append(overlap)
        results.append(
            {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "reference": reference,
                "prediction": prediction,
                "exact_match": is_exact,
                "token_overlap": round(overlap, 3),
            }
        )

    n = len(results)
    summary = {
        "n_examples": n,
        "exact_match_rate": round(exact_matches / n, 4) if n else 0.0,
        "mean_token_overlap": round(sum(overlap_scores) / n, 4) if n else 0.0,
    }
    return {"summary": summary, "samples": results}


def benchmark_inference(
    backend: str, model, tokenizer, prompt: str = "Translate 'house' into Bisaya.", n: int = 10
) -> dict:
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        generate((backend, model), tokenizer, prompt, max_new_tokens=32)
        latencies.append(time.perf_counter() - start)
    return {
        "n_runs": n,
        "mean_latency_seconds": round(sum(latencies) / n, 3),
        "min_latency_seconds": round(min(latencies), 3),
        "max_latency_seconds": round(max(latencies), 3),
    }


HUMAN_EVAL_CHECKLIST = """# Human Evaluation Checklist -- Bisaya LLM

For each sampled generation in `eval_samples.json`, a native Cebuano/Bisaya
speaker should rate:

- [ ] **Fluency** (1-5): Does the output read as natural Bisaya/Cebuano?
- [ ] **Accuracy** (1-5): Does the translation/definition match intended meaning?
- [ ] **Orthography**: Are diacritics/spelling consistent with common usage?
- [ ] **Register**: Is formality/dialect appropriate to the prompt?
- [ ] **Hallucination check**: Does the model invent a word/meaning not in
      the source dictionary data?
- [ ] **Cultural appropriateness**: Any culturally insensitive or incorrect content?

Aggregate notes:
- Common failure patterns observed:
- Recommended dataset/training fixes:
"""


def main(
    model_dir: Path = Path("output/checkpoints/lora_adapter"),
    config: Path = Path("configs/training.yaml"),
    eval_file: Path = Path("output/datasets/bisaya-instruction-tuning/test.jsonl"),
    out_dir: Path = Path("output/eval"),
    max_examples: int = 200,
) -> None:
    """Plain function with real defaults -- safe to call directly from
    cli.py, not just from the Typer CLI below."""
    cfg = load_yaml_config(config.name)
    backend, model, tokenizer = load_model(model_dir, cfg["model"]["max_seq_length"])

    if not eval_file.exists():
        raise FileNotFoundError(
            f"Eval file not found: {eval_file}. Run `python cli.py build-instructions` first."
        )
    eval_examples = [
        json.loads(line)
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_results = evaluate_split(backend, model, tokenizer, eval_examples, max_examples)
    (out_dir / "eval_samples.json").write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    logger.info("evaluation summary: %s", eval_results["summary"])

    bench = benchmark_inference(backend, model, tokenizer)
    (out_dir / "inference_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    logger.info("inference benchmark: %s", bench)

    (out_dir / "human_eval_checklist.md").write_text(HUMAN_EVAL_CHECKLIST, encoding="utf-8")
    logger.info("wrote evaluation artifacts to %s", out_dir)


@app.command(name="main")
def _cli(
    model_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--model-dir"),
    config: Path = typer.Option(Path("configs/training.yaml"), "--config"),
    eval_file: Path = typer.Option(
        Path("output/datasets/bisaya-instruction-tuning/test.jsonl"), "--eval-file"
    ),
    out_dir: Path = typer.Option(Path("output/eval"), "--out-dir"),
    max_examples: int = typer.Option(200, "--max-examples"),
) -> None:
    main(model_dir, config, eval_file, out_dir, max_examples)


if __name__ == "__main__":
    app()
