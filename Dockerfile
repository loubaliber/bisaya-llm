# CPU image for scraping / parsing / cleaning / dataset-building / serving.
# For GPU fine-tuning, use a CUDA base image and add requirements-train.txt
# (e.g. FROM nvidia/cuda:12.1.1-devel-ubuntu22.04, then install Python 3.12 +
# `pip install -r requirements.txt -r requirements-train.txt`).

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
