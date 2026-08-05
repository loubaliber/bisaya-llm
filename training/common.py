"""Shared helpers used by both training/train_mlx.py and
training/train_unsloth.py, plus inference.py and evaluate.py.

This module owns the one thing that must never diverge between backends:
the prompt format. It also owns the `backend.json` marker convention that
lets downstream stages (evaluate/inference/upload) know how a given
adapter directory was produced, without guessing from file contents.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Literal

Backend = Literal["mlx", "unsloth"]

BACKEND_MARKER_FILE = "backend.json"


def format_prompt(instruction: str, input_text: str, output_text: str = "") -> str:
    """Single source of truth for the Alpaca-style prompt template. Both
    backends and both training/inference paths call this so a prompt
    format change only ever needs to happen in one place."""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt + output_text


def detect_platform_backend() -> Backend:
    """Best-effort default backend for this machine. Callers should still
    allow an explicit --backend override (see cli.py)."""
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon and shutil.which("python3") is not None:
        try:
            import mlx_lm  # noqa: F401,PLC0415

            return "mlx"
        except ImportError:
            pass
    return "unsloth"


def write_backend_marker(adapter_dir: Path, backend: Backend, base_model_id: str) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    marker = {"backend": backend, "base_model_id": base_model_id}
    (adapter_dir / BACKEND_MARKER_FILE).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def read_backend_marker(adapter_dir: Path) -> dict:
    marker_path = adapter_dir / BACKEND_MARKER_FILE
    if not marker_path.exists():
        raise FileNotFoundError(
            f"No {BACKEND_MARKER_FILE} found in {adapter_dir}. This directory was not "
            "produced by training/train_mlx.py or training/train_unsloth.py (or training "
            "has not completed yet)."
        )
    return json.loads(marker_path.read_text(encoding="utf-8"))


def require_trained_adapter(adapter_dir: Path) -> dict:
    """Hard check used by evaluate/inference/upload before touching the
    adapter directory. Raises a clear, actionable error instead of letting
    a downstream library (e.g. huggingface_hub) misinterpret a missing
    local path as a remote repo id."""
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        raise FileNotFoundError(
            f"No trained adapter found at '{adapter_dir}'.\n"
            "Training has not completed successfully yet. Run:\n"
            "    python cli.py train\n"
            "and check the logs for errors before retrying this step."
        )
    return read_backend_marker(adapter_dir)
