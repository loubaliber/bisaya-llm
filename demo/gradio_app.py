"""Minimal Gradio demo for the fine-tuned Bisaya model.

Run (on a GPU box, after training):
    pip install gradio
    python demo/gradio_app.py --model-dir output/checkpoints/lora_adapter
"""

from __future__ import annotations

from pathlib import Path

import typer

from common import load_yaml_config
from training.inference import generate, load_model

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_dir: Path = typer.Option(Path("output/checkpoints/lora_adapter"), "--model-dir"),
    config: Path = typer.Option(Path("configs/training.yaml"), "--config"),
    share: bool = typer.Option(False, "--share"),
) -> None:
    import gradio as gr  # noqa: PLC0415 - optional dependency, only needed for the demo

    cfg = load_yaml_config(config.name)
    backend, model, tokenizer = load_model(model_dir, cfg["model"]["max_seq_length"])

    def respond(instruction: str, input_text: str) -> str:
        if not instruction.strip():
            return "Please enter an instruction, e.g. \"Translate 'house' into Bisaya.\""
        return generate((backend, model), tokenizer, instruction, input_text)

    demo = gr.Interface(
        fn=respond,
        inputs=[
            gr.Textbox(label="Instruction", placeholder="Unsay meaning sa 'kaon'?"),
            gr.Textbox(label="Input (optional)", placeholder=""),
        ],
        outputs=gr.Textbox(label="Response"),
        title="Bisaya (Cebuano) LLM Demo",
        description=(
            "Fine-tuned with Unsloth on a Cebuano/Bisaya dictionary-derived "
            "instruction dataset. See docs/LEGAL_NOTICE.md in the repo for "
            "data provenance and licensing notes."
        ),
        examples=[
            ["Translate 'house' into Bisaya.", ""],
            ["Unsay meaning sa 'kaon'?", ""],
            ["Use 'balay' in a sentence.", ""],
        ],
    )
    demo.launch(share=share)


if __name__ == "__main__":
    app()
