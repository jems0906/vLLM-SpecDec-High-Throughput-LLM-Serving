FROM vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace/vllm-specdec
COPY pyproject.toml README.md RUNBOOK.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY benchmarks ./benchmarks
COPY patches ./patches

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir --no-deps -e .

EXPOSE 8000
CMD ["python3", "scripts/launch_tp_server.py", "--config", "configs/llama3_8b_specdec.json"]
