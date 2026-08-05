"""Push the built dataset to the Hugging Face Hub.

HARD GATE: this script will not create/update a *public* dataset repo
unless BOTH of the following are true:
  1. --i-have-permission is passed on the CLI, AND
  2. configs/dataset.yaml `dataset.PERMISSION_NOTES` is non-empty.

Without both, the repo is created/updated as private=True regardless of
`private_by_default`. See docs/LEGAL_NOTICE.md for why this exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

from common import fail_fast, load_yaml_config, setup_logging

load_dotenv()
logger = setup_logging("huggingface")
app = typer.Typer(add_completion=False)


def resolve_privacy(cfg: dict, i_have_permission: bool) -> bool:
    """Return True if the dataset repo should be private."""
    permission_notes = cfg["dataset"].get("PERMISSION_NOTES", "").strip()
    env_flag = os.getenv("I_HAVE_REDISTRIBUTION_PERMISSION", "false").lower() == "true"

    if cfg["dataset"].get("private_by_default", True):
        # Stay private unless permission has been explicitly documented.
        return not (i_have_permission and env_flag and permission_notes)

    # Config says private_by_default: false -- still require the gate.
    if not (i_have_permission and env_flag and permission_notes):
        fail_fast(
            "Refusing to publish a public dataset without documented permission. "
            "Pass --i-have-permission, set I_HAVE_REDISTRIBUTION_PERMISSION=true in .env, "
            "and fill in dataset.PERMISSION_NOTES in configs/dataset.yaml. "
            "See docs/LEGAL_NOTICE.md."
        )
    return False


def upload_dataset(
    dataset_dir: Path,
    repo_id: str,
    private: bool,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["arrow/**", "*.checkpoint.json"],
    )
    logger.info(
        "uploaded %s to https://huggingface.co/datasets/%s (private=%s)",
        dataset_dir,
        repo_id,
        private,
    )


def main(
    dataset_dir: Path = Path("output/datasets/bisaya-cebuano-dictionary"),
    i_have_permission: bool = False,
) -> None:
    """Plain function with real (non-Typer) defaults -- safe to call directly
    from cli.py or other Python code, not just from the Typer CLI below."""
    cfg = load_yaml_config("dataset.yaml")
    token = os.getenv("HF_TOKEN")
    if not token:
        fail_fast("HF_TOKEN not set. Copy .env.example to .env and fill it in.")

    private = resolve_privacy(cfg, i_have_permission)
    if private:
        logger.warning(
            "Publishing as PRIVATE. To go public you need --i-have-permission, "
            "I_HAVE_REDISTRIBUTION_PERMISSION=true in .env, and a filled-in "
            "PERMISSION_NOTES field in configs/dataset.yaml."
        )

    upload_dataset(dataset_dir, cfg["dataset"]["hf_repo_id"], private, token)


@app.command(name="main")
def _cli(
    dataset_dir: Path = typer.Option(
        Path("output/datasets/bisaya-cebuano-dictionary"), "--dataset-dir"
    ),
    i_have_permission: bool = typer.Option(
        False, "--i-have-permission", help="Confirm you have redistribution permission"
    ),
) -> None:
    main(dataset_dir, i_have_permission)


if __name__ == "__main__":
    app()
