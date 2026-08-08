#!/bin/bash
# One-click launcher for the alfworld_step sidecar. Built on the `alfworld`
# pip package's TextWorld env (AlfredTWEnv) -- docker/server.py wraps its own
# step()/init_env() directly, no tag-parsing loop reimplemented.
#
# Same "one long-lived container, one fixed port for the whole job" pattern
# as ../run_sandbox.sh / ../search/run_search_sidecar.sh; see docker/
# Dockerfile's docstring for why this needs its OWN container (alfworld's
# gymnasium/stable-baselines3 pins, separate from the training image).
#
# Idempotent: if the container is already running and healthy, this is a
# no-op.
#
#   PORT      (default 8423)                port this sidecar listens on (127.0.0.1 only)
#   IMAGE_TAG (default sdpo-react-alfworld) local image tag
#   CONTAINER (default sdpo-react-alfworld) container name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8423}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-alfworld}"
CONTAINER="${CONTAINER:-sdpo-react-alfworld}"

# Same host-vs-enroot split as ../run_sandbox.sh: the training job itself has
# no `docker` binary and doesn't need one -- this sidecar is a host-level
# singleton, started BEFORE the training/enroot session, reached over HTTP.
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "ALFWorld sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND alfworld sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_alfworld_sidecar.sh (see ../../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "ALFWorld sidecar already running and healthy: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

docker run -d --rm \
    --name "$CONTAINER" \
    --network=bridge \
    -p "127.0.0.1:${PORT}:8423" \
    "$IMAGE_TAG"

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "ALFWorld sidecar up: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    sleep 1
done

echo "ALFWorld sidecar failed to become healthy within 60s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
