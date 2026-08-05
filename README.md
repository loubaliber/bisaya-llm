# bisaya-llm

An open, reproducible pipeline for building a **Cebuano (Bisaya)** NLP
dataset from publicly accessible dictionary sources and fine-tuning an
open-weight LLM on it with **[Unsloth](https://github.com/unslothai/unsloth)**.

> **Read `docs/LEGAL_NOTICE.md` before you scrape or publish anything.**
> Two of the three candidate dictionary sources are disabled by default
> because their dictionary text derives from a third-party copyrighted work
> (John U. Wolff's *A Dictionary of Cebuano Visayan*, 1972). Publishing a
> public dataset is gated behind an explicit permission flag. This is not
> boilerplate — the code enforces it.

---

## Pipeline overview

```
                 ┌────────────┐
                 │ robots.txt │  (checked live, every request)
                 └─────┬──────┘
                        │
 configs/scraper.yaml   ▼
        │        ┌─────────────┐        output/raw/<source>.jsonl
        └───────► │  scraper/   │ ─────────────────────────────►
                 │  (crawler,  │        {word, url, html, source}
                 │  spiders)   │
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐        output/cleaned/<source>.parsed.jsonl
                 │  parser/    │ ─────────────────────────────►
                 │ HTML→fields │        {word, translation, pos, ...}
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐        output/cleaned/master.jsonl
                 │  cleaning/  │ ─────────────────────────────►
                 │ normalize → │        validated CleanedEntry records
                 │ dedupe →    │
                 │ validate    │
                 └─────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
     ┌─────────────────┐  ┌───────────────────────┐
     │ dataset/         │  │ dataset/               │
     │ build_dataset.py │  │ build_instruction_     │
     │ → HF dataset +   │  │ dataset.py             │
     │   dataset card   │  │ → Alpaca-style SFT data│
     └────────┬─────────┘  └───────────┬────────────┘
              │                        │
              ▼                        ▼
     huggingface/upload_dataset.py   cli.py train  (auto-picks backend)
     (permission-gated push)          │
                        ┌─────────────┴─────────────┐
                        ▼                            ▼
              training/train_mlx.py        training/train_unsloth.py
              (Apple Silicon, mlx_lm.lora)  (CUDA, Unsloth + TRL SFTTrainer)
                        └─────────────┬─────────────┘
                                      ▼
                         writes backend.json marker
                                      │
                                      ▼
                    training/evaluate.py, training/inference.py
                    (read backend.json, dispatch automatically)
                                      │
                                      ▼
                          huggingface/upload_model.py
```

## Repository layout

```text
bisaya-llm/
  README.md               this file
  docs/LEGAL_NOTICE.md     mandatory reading before scraping/publishing
  docs/ARCHITECTURE.md     component-level notes
  pyproject.toml, requirements*.txt, .env.example
  configs/                 scraper.yaml, dataset.yaml, training.yaml
  scraper/                 crawler.py, robots.py, rate_limiter.py, talkbisaya.py, binisaya.py
  parser/                  parser.py (HTML -> structured fields)
  cleaning/                normalize.py, deduplicate.py, validate.py
  dataset/                 build_dataset.py, build_instruction_dataset.py, statistics.py, splitting.py
  training/                common.py, train_mlx.py, train_unsloth.py, inference.py, evaluate.py
  huggingface/             upload_dataset.py, upload_model.py
  demo/                    Gradio demo + FastAPI inference server (stretch goals)
  tests/                   pytest suite
  cli.py                   single Typer CLI wrapping every stage
  common.py, schemas.py    shared config/logging + Pydantic data contracts
  Dockerfile, .github/workflows/ci.yml
```

## Installation

```bash
git clone <this-repo> && cd bisaya-llm
python3.12 -m venv .venv && source .venv/bin/activate

# Core: scraping / dataset tooling (CPU-only, fast)
pip install -r requirements.txt

# Optional, GPU (CUDA) box only: unsloth, torch, trl, peft, bitsandbytes
pip install -r requirements-train.txt

# Optional, Apple Silicon only: MLX training backend
pip install -r requirements-mlx.txt

# Copy the env template, then fill in HF_TOKEN etc.
cp .env.example .env

# Optional but recommended
pre-commit install
```

> **Note for zsh users:** if you copy commands with trailing `# comment`
> text that contains parentheses (as some older docs did), zsh can throw
> `zsh: bad pattern`. The commands above put comments on their own line to
> avoid that entirely — always safe to copy-paste as-is.

## Usage

Everything goes through `cli.py` (or `python -m cli` / the `bisaya` console
script once installed):

```bash
# 1. Scrape (talkbisaya only, by default — see LEGAL_NOTICE.md)
python cli.py scrape --source talkbisaya --limit 200

# 2. Parse raw HTML into structured fields
python cli.py parse --source talkbisaya

# 3. Normalize + dedupe + validate into output/cleaned/master.jsonl
python cli.py clean

# 4. Inspect quality
python cli.py stats

# 5. Build the published dataset (JSONL/Parquet/Arrow + dataset card)
python cli.py build-dataset

# 6. Build the instruction-tuning dataset
python cli.py build-instructions

# 7. Push to Hugging Face (stays private unless you've documented permission)
# Plain call -> private dataset repo:
python cli.py upload-dataset
# With documented permission -> can go public (see docs/LEGAL_NOTICE.md):
python cli.py upload-dataset --i-have-permission

# 8. Fine-tune. Auto-selects MLX (Apple Silicon) or Unsloth (CUDA); override with --backend
python cli.py train --config configs/training.yaml
python cli.py train --backend mlx        # force MLX
python cli.py train --backend unsloth    # force CUDA/Unsloth

# 9. Evaluate + run inference
python cli.py evaluate --model-dir output/checkpoints/lora_adapter
python cli.py inference "Unsay meaning sa 'balay'?"

# 10. Push the trained adapter/model
python cli.py upload-model
```

### Bypassing the scraper entirely

Drop hand-compiled or permissively-licensed (e.g. CC-BY-SA Wiktionary)
entries as JSONL into `output/raw/manual/`, matching the `RawEntry` schema
with a JSON payload in the `html` field (see `parser/parser.py:parse_manual`
and `tests/test_parser.py` for the exact shape). This is the recommended
path for anyone who wants to exercise the full pipeline without touching
the licensing questions in `docs/LEGAL_NOTICE.md` at all.

## Dataset schema

| field | type | description |
|---|---|---|
| `id` | string | stable per-record identifier |
| `word` | string | Cebuano (Bisaya) headword |
| `translation` | string | English gloss/translation |
| `part_of_speech` | string? | grammatical category, if known |
| `pronunciation` | string? | pronunciation guide, if known |
| `example_sentences` | list[string] | example usage sentences |
| `synonyms` / `antonyms` / `related_words` | list[string] | lexical relations |
| `definition` | string? | fuller definition text |
| `source_url` / `source` | string | provenance |

## Fine-tuning notes

Two independent training backends share one config file and one CLI command
(`python cli.py train`), which auto-detects which to use (override with
`--backend mlx` / `--backend unsloth`):

| | `training/train_unsloth.py` | `training/train_mlx.py` |
|---|---|---|
| Hardware | CUDA/NVIDIA | Apple Silicon |
| Stack | Unsloth + PEFT + TRL `SFTTrainer` + bitsandbytes | `mlx_lm.lora` (MLX's own trainer) |
| Install | `pip install -r requirements-train.txt` | `pip install mlx mlx-lm` |

They never mix: the MLX path never touches PEFT/TRL/bitsandbytes and never
hands an MLX tokenizer to a Hugging Face trainer (that mismatch —
`mlx_lm.tokenizer_utils.TokenizerWrapper` vs the `PreTrainedTokenizerBase`
TRL's `SFTTrainer` expects — was the original bug this split fixes).

- `configs/training.yaml` takes any Unsloth-compatible base model via
  `model.base_model_id` for the CUDA path — nothing is hardcoded.
- LoRA/QLoRA, 4-bit quantization, gradient checkpointing (`use_gradient_checkpointing:
  "unsloth"`), and BF16/FP16 are all config-driven for the CUDA path.
  `lora.mlx_num_layers` and `training_args.mlx_iters` configure the MLX path
  (which is step-driven, not epoch-driven).
- `training_args.resume_from_checkpoint` supports resuming an interrupted
  CUDA run; MLX resumes via `resume_adapter_file`, wired from the same field.
- Both backends write a `backend.json` marker into the adapter output
  directory. `evaluate`, `inference`, and `upload-model` read it and
  dispatch automatically — you never need to tell them which backend
  trained a given adapter.
- The Unsloth script additionally saves a merged 16-bit model and can push
  either artifact to the Hub once `output.push_*_to_hub` is `true`.

## Evaluation

`training/evaluate.py` produces:
- automatic exact-match / token-overlap proxy metrics on a held-out split,
- a JSON file of sample generations for qualitative review,
- an inference latency/throughput benchmark,
- `output/eval/human_eval_checklist.md` — a structured rubric intended for
  a native Cebuano/Bisaya speaker to fill in (fluency, accuracy,
  orthography, register, hallucination, cultural appropriateness). Automatic
  metrics are a first-pass signal only; they are not a substitute for this.

## Code quality

- Type hints + docstrings throughout; `mypy --strict` and `ruff` configured
  in `pyproject.toml`.
- `pytest` suite covers normalization, deduplication, validation, parsing,
  and instruction-template generation (`tests/`).
- `.pre-commit-config.yaml` runs ruff + mypy + basic hygiene hooks.
- `.github/workflows/ci.yml` runs lint + tests on every push/PR.

## Stretch goals included

- `demo/gradio_app.py` — a minimal Gradio demo for the fine-tuned model.
- `demo/api_server.py` — a FastAPI inference server (`/generate` endpoint).
- `Dockerfile` — CPU image for the scraping/dataset tooling (training image
  needs a CUDA base image + `requirements-train.txt`, noted inline).
- `.github/workflows/ci.yml` — lint + test CI.

Not yet implemented (left as documented future work in
`docs/ARCHITECTURE.md`): morphological analysis, synonym graph generation,
embedding-based near-duplicate detection, and a RAG-ready vector export.

## Future improvements

- Swap the token-overlap eval proxy for a proper MT metric (chrF++ tends to
  work better than BLEU for morphologically rich, low-resource languages).
- Add embedding-based fuzzy dedup (current dedup is exact-match on the
  normalized headword).
- Expand `configs/scraper.yaml` with additional *permissively licensed*
  sources (Wiktionary dumps, community-contributed corpora) rather than
  more dictionary sites of unclear provenance.
- Automatic dataset versioning via HF Hub dataset tags/releases.

## License

Code in this repository: MIT (see `pyproject.toml`). **Dataset content is
NOT covered by this license** — see `docs/LEGAL_NOTICE.md`.
