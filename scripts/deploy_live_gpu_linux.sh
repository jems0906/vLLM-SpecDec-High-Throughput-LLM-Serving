#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker/docker-compose.yml"
ENV_FILE="$ROOT/docker/.env"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

fail() {
  printf '\n[ERROR] %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed on this host."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is missing. Use a Linux host with NVIDIA drivers installed."

docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable. Start Docker before running this script."

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/docker/.env.example" "$ENV_FILE"
  log "Created docker/.env from docker/.env.example"
fi

if grep -Eq '^HF_TOKEN=(your_huggingface_token)?$' "$ENV_FILE"; then
  fail "Set a real HF_TOKEN in docker/.env before starting the stack."
fi

cd "$ROOT"

log "Host GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "Pulling images"
docker compose -f "$COMPOSE_FILE" pull

log "Starting live GPU stack"
docker compose -f "$COMPOSE_FILE" up -d

log "Service status"
docker compose -f "$COMPOSE_FILE" ps

log "Recent vllm-specdec logs"
docker compose -f "$COMPOSE_FILE" logs --tail=60 vllm-specdec || true

if command -v curl >/dev/null 2>&1; then
  log "Health probe"
  curl -fsS http://127.0.0.1:8000/health || true
fi

log "Live GPU deployment command completed"
