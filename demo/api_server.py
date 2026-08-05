"""FastAPI inference server for the fine-tuned Bisaya model.

Run (on a GPU box, after training):
    pip install fastapi uvicorn
    uvicorn demo.api_server:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST localhost:8000/generate \\
      -H 'Content-Type: application/json' \\
      -d '{"instruction": "Translate \\"house\\" into Bisaya."}'
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from common import load_yaml_config
from training.inference import generate, load_model

MODEL_DIR = Path(os.getenv("BISAYA_MODEL_DIR", "output/checkpoints/lora_adapter"))
CONFIG_PATH = Path(os.getenv("BISAYA_TRAINING_CONFIG", "configs/training.yaml"))

app = FastAPI(
    title="Bisaya LLM Inference API",
    description="Fine-tuned Cebuano/Bisaya instruction-following model. "
    "See docs/LEGAL_NOTICE.md for data provenance/licensing notes.",
)

_cfg = load_yaml_config(CONFIG_PATH.name)
_backend, _model, _tokenizer = load_model(MODEL_DIR, _cfg["model"]["max_seq_length"])


class GenerateRequest(BaseModel):
    instruction: str
    input: str = ""
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    output: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": str(MODEL_DIR)}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
    output = generate(
        (_backend, _model), _tokenizer, request.instruction, request.input, request.max_new_tokens
    )
    return GenerateResponse(output=output)
