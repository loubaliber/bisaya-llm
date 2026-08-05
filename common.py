"""Shared utilities: YAML config loading, logging setup, and small helpers
used across scraper/, parser/, cleaning/, dataset/, training/, huggingface/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "output"


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from configs/ by name, e.g. 'scraper.yaml'."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")
    return data


def setup_logging(
    name: str, level: int = logging.INFO, log_dir: Path | None = None
) -> logging.Logger:
    """Configure a rich console logger plus an optional rotating file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on re-import)

    logger.setLevel(level)
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def ensure_output_dirs() -> None:
    for sub in ("raw", "cleaned", "datasets", "logs", "checkpoints"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def fail_fast(message: str) -> None:
    """Print an error and exit non-zero. Used for hard legal/config gates."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)
