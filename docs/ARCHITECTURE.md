# Architecture Notes

## Design principles

1. **Legal compliance is a hard gate, not a comment.** robots.txt is
   checked live at request time (`scraper/robots.py`); publishing to a
   public HF dataset requires an explicit `--i-have-permission` flag *and*
   a documented `PERMISSION_NOTES` entry (`huggingface/upload_dataset.py`).
   See `docs/LEGAL_NOTICE.md`.
2. **Every stage is a pure function over JSONL files on disk**, so any
   stage can be re-run independently, inspected, or swapped out (e.g. drop
   in `output/raw/manual/*.jsonl` to skip scraping entirely).
3. **Config over code.** Base model, LoRA hyperparameters, quality filters,
   and rate limits all live in `configs/*.yaml`, not hardcoded in scripts.
4. **Site-specific logic is isolated.** `scraper/crawler.py` owns all
   politeness/retry/checkpoint logic; `scraper/talkbisaya.py` and
   `scraper/binisaya.py` only implement URL discovery. Parsers follow the
   same pattern in `parser/parser.py` via a dispatch table keyed by source.

## Data flow contracts (schemas.py)

```
RawEntry        -- {word, url, html, source, scraped_at}
    │  parser/parser.py
    ▼
ParsedEntry     -- structured fields, still unvalidated/uncleaned
    │  cleaning/normalize.py -> cleaning/deduplicate.py -> cleaning/validate.py
    ▼
CleanedEntry    -- final, validated, has stable `id`
    │  dataset/build_dataset.py            dataset/build_instruction_dataset.py
    ▼                                              ▼
HF DatasetDict (dictionary)          InstructionExample -> HF DatasetDict (SFT)
```

## Why a generic `BaseSpider`

New sources should not be able to accidentally skip robots.txt checks,
rate limiting, retries, or checkpointing. `scraper/crawler.py:BaseSpider`
owns the crawl loop; a new spider only implements
`discover_entry_urls() -> list[tuple[word, url]]`. `scraper/binisaya.py`
additionally demonstrates how to hard-gate a source behind an explicit
`allow_disabled_source=True` for cases with unclear redistribution rights.

## Known limitations / future work

- **Dedup is exact-match** on the lowercased headword. A near-duplicate
  entry with different spelling/diacritics is not currently merged.
  Planned: embedding-based fuzzy dedup (`sentence-transformers` + cosine
  threshold) as a stretch goal.
- **No morphological analysis.** Cebuano's rich affixation (mu-/mi-/nag-
  etc.) is not modeled; the dataset treats each headword as an opaque
  string. A morphological analyzer would let the instruction-dataset
  generator produce conjugation/affixation exercises.
- **No synonym graph.** `synonyms`/`antonyms`/`related_words` are flat
  lists per entry; a graph structure (e.g. NetworkX export) would enable
  richer downstream tasks (synonym-chain QA, graph-based data augmentation).
- **No RAG-ready vector export.** Adding a `dataset/build_embeddings.py`
  step (sentence-transformers -> FAISS/HF datasets with an embedding
  column) is straightforward future work using the same `CleanedEntry`
  records.

## Training backend split (MLX vs Unsloth/CUDA)

Originally a single `training/train_unsloth.py` was extended ad hoc to also
handle Apple Silicon, which meant an MLX model/tokenizer got handed to
`trl.SFTTrainer` -- a Hugging Face-only API that requires a
`PreTrainedTokenizerBase`, not MLX's `TokenizerWrapper`. The fix was to
treat MLX and CUDA as two fully separate implementations that never share
model/tokenizer objects:

- `training/common.py` -- the only shared surface: prompt formatting
  (`format_prompt`), and the `backend.json` marker convention so
  `evaluate`/`inference`/`upload-model` can tell which backend produced a
  given adapter without inspecting its internals.
- `training/train_unsloth.py` -- CUDA-only; hard-asserts `torch.cuda.is_available()`
  at the top of `train()` so it fails fast with an actionable message
  instead of a confusing crash deep inside TRL.
- `training/train_mlx.py` -- shells out to `python -m mlx_lm.lora` (a
  stable CLI) rather than calling `mlx_lm.tuner` internals directly, since
  those internals have changed shape across mlx-lm releases and the CLI is
  the part the maintainers keep backwards-compatible.
- `cli.py train` picks a backend via `training.common.detect_platform_backend()`
  (Darwin/arm64 + mlx_lm importable -> mlx, else unsloth), overridable with
  `--backend`.

## Dead code / duplication removed in the MLX refactor pass

- `dataset/build_dataset.py` and `dataset/build_instruction_dataset.py`
  each had their own copy of the train/validation/test split logic;
  consolidated into `dataset/splitting.py:records_to_split_dataset()`.
- `huggingface/upload_dataset.py` and `huggingface/upload_model.py` had
  `@app.command() def main(...)` with `typer.Option(...)` as the *default
  value* for parameters, then were called directly as plain Python
  functions from `cli.py`. Typer only resolves `Option(...)` defaults when
  it does its own CLI parsing, so a direct call leaked a raw `OptionInfo`
  object into path-handling code (`ValueError: Provided path: '<typer...
  OptionInfo object>' is not a directory`). Fixed by separating each into
  a plain function with real Python defaults (used by both `cli.py` and
  direct imports) and a thin `_cli` Typer wrapper that only exists for
  `python -m huggingface.upload_model` standalone invocation.
- `training/inference.py` and `training/evaluate.py` had the same
  Typer-default-leak pattern; fixed the same way.
- Duplicate prompt-formatting logic between the (then-mixed) training path
  and any future MLX path was pre-empted by centralizing it in
  `training/common.py:format_prompt()` before either backend was written
  against it -- `tests/test_training_common.py` explicitly asserts both
  backends produce byte-identical prompts for the same example, so this
  can't silently drift apart again.
