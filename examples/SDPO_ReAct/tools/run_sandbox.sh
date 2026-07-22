#!/bin/bash
# One-click launcher for the code_interpreter sandbox sidecar.
#
# Starts (or reuses) exactly ONE Docker container exposing exactly ONE port for
# the whole training job -- the host this runs on has a hard limit on how many
# ports a job may open, so we do NOT spawn a container/port per rollout worker
# or per tool call. Every rollout coroutine (across every SGLang engine, on
# every GPU) shares this single sidecar over the same fixed port; see
# tools/tool_client.py for the caller side.
#
# Idempotent: if the container is already running and healthy, this is a no-op.
#
#   PORT      (default 8420)                port the sandbox listens on (127.0.0.1 only)
#   IMAGE_TAG (default sdpo-react-sandbox)   local image tag
#   CONTAINER (default sdpo-react-sandbox)   container name
#   SANDBOX_CPUS   (default 32)              CPU core budget for the sidecar
#   SANDBOX_MEMORY (default 32g)             memory budget for the sidecar
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8420}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-sandbox}"
CONTAINER="${CONTAINER:-sdpo-react-sandbox}"
# Each code_interpreter call spawns its own `python3 -c` subprocess (real OS
# process isolation, see sandbox_server.py); at --generate-max-turns 5-20 x
# --n-samples-per-prompt 8 x --rollout-batch-size 32, several hundred of these
# can be in flight per rollout batch. The original --cpus=4 was measured
# pegged at 384% (of its own 4-core budget) DURING an otherwise-idle 96-core
# host (84% idle overall) -- a self-inflicted bottleneck, not a real resource
# constraint. 32 cores / 32g leaves the training/rollout GPU processes' own
# host-CPU needs untouched (they are GPU-bound, not CPU-bound) while giving
# the sandbox real headroom.
SANDBOX_CPUS="${SANDBOX_CPUS:-32}"
SANDBOX_MEMORY="${SANDBOX_MEMORY:-32g}"

# The training job itself (e.g. inside an enroot session, see
# ../enroot-run-sdpo-react.sh) has no `docker` binary and no need for one: the
# sandbox is a HOST-level singleton, started once by the launcher BEFORE the
# training container starts, and reached over the shared network namespace.
# When docker isn't on PATH, this script degrades to a pure health check --
# if the sidecar isn't already up in that case, we can't start it from here,
# so fail loudly with a pointer at the real fix instead of a confusing
# "docker: command not found".
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND sandbox sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_sandbox.sh (see ../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox sidecar already running and healthy: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

# Bind to 127.0.0.1 only: this sidecar is reached exclusively by rollout code
# on the same host (miles.utils.http_utils.post -> http://127.0.0.1:$PORT), it
# is never meant to be reachable off-host.
docker run -d --rm \
    --name "$CONTAINER" \
    --network=bridge \
    --memory="${SANDBOX_MEMORY}" \
    --cpus="${SANDBOX_CPUS}" \
    -p "127.0.0.1:${PORT}:8420" \
    "$IMAGE_TAG"

for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox sidecar up: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    sleep 1
done

echo "Sandbox sidecar failed to become healthy within 30s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
