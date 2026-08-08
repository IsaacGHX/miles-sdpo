#!/bin/bash
# One-click launcher for the webshop_step sidecar. Built on WebShop's own
# ``WebAgentTextEnv-v0`` gym env (github.com/princeton-nlp/WebShop, vendored
# verbatim at image build time) -- docker/server.py wraps its own step()/
# reset() directly, no tag-parsing loop reimplemented.
#
# Same "one long-lived container, one fixed port for the whole job" pattern
# as ../run_sandbox.sh / ../search/run_search_sidecar.sh / ../alfworld/
# run_alfworld_sidecar.sh; see docker/Dockerfile's docstring for why this
# needs its OWN container (WebShop's Python<=3.10 + old gym/pyserini/torch
# pins, separate from the training image).
#
# NOTE: first build is slow (~10-20 min) -- clones WebShop, downloads the
# small product dataset via gdown, builds a Lucene search index. All baked
# into the image, so subsequent starts are instant (docker layer cache).
#
# Idempotent: if the container is already running and healthy, this is a
# no-op.
#
#   PORT      (default 8422)                port this sidecar listens on (127.0.0.1 only)
#   IMAGE_TAG (default sdpo-react-webshop)  local image tag
#   CONTAINER (default sdpo-react-webshop)  container name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8422}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-webshop}"
CONTAINER="${CONTAINER:-sdpo-react-webshop}"

# Same host-vs-enroot split as ../run_sandbox.sh: the training job itself has
# no `docker` binary and doesn't need one -- this sidecar is a host-level
# singleton, started BEFORE the training/enroot session, reached over HTTP.
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "WebShop sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND webshop sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_webshop_sidecar.sh (see ../../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "WebShop sidecar already running and healthy: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

docker run -d --rm \
    --name "$CONTAINER" \
    --network=bridge \
    -p "127.0.0.1:${PORT}:8422" \
    "$IMAGE_TAG"

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "WebShop sidecar up: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    sleep 1
done

echo "WebShop sidecar failed to become healthy within 60s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
