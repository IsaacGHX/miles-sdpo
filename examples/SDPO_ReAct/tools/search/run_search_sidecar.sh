#!/bin/bash
# One-click launcher for the search/open/find sidecar. Built on i-DeepSearch's
# own BrowserPool/BrowserTool (docker/browser.py, copied verbatim), which
# wraps gpt-oss's simple_browser tool; docker/retrieval_adapter.py is the only
# repo-specific piece -- a thin shim exposing the existing wiki-18 retrieval
# index as the /search+/get_content contract BrowserPool's 'local' backend
# expects (see that file's docstring for why).
#
# Same "one long-lived container, one fixed port for the whole job" pattern as
# ../run_sandbox.sh (the code_interpreter sidecar); see docker/Dockerfile's
# docstring for why this needs its OWN container (gpt-oss requires Python
# >=3.12, separate from both the training image and the code sandbox image).
#
# Requires the wiki-18 retrieval server (../run_retrieval.sh) to already be up
# -- this sidecar routes search/open/find onto it, it does not embed a
# retriever of its own.
#
# Idempotent: if the container is already running and healthy, this is a no-op.
#
#   PORT              (default 8421)                   port this sidecar listens on (127.0.0.1 only)
#   RETRIEVAL_PORT     (default 8000)                  port the wiki-18 retrieval server listens on
#   IMAGE_TAG         (default sdpo-react-search)       local image tag
#   CONTAINER         (default sdpo-react-search)       container name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8421}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-search}"
CONTAINER="${CONTAINER:-sdpo-react-search}"

# Same host-vs-enroot split as ../run_sandbox.sh: the training job itself has
# no `docker` binary and doesn't need one -- this sidecar is a host-level
# singleton, started BEFORE the training/enroot session, reached over HTTP.
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Search sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND search sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_search_sidecar.sh (see ../../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Search sidecar already running and healthy: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

# host.docker.internal resolves on Docker Desktop; on plain Linux dockerd it
# does not by default -- --add-host maps it to the bridge gateway explicitly,
# so retrieval_adapter.py's SDPO_REACT_SEARCH_URL (127.0.0.1:$RETRIEVAL_PORT
# from the HOST's perspective) is reachable the same way from inside this
# container without hardcoding a docker-internal IP.
docker run -d --rm \
    --name "$CONTAINER" \
    --network=bridge \
    --add-host=host.docker.internal:host-gateway \
    -e SDPO_REACT_SEARCH_URL="http://host.docker.internal:${RETRIEVAL_PORT}/retrieve" \
    -p "127.0.0.1:${PORT}:8421" \
    "$IMAGE_TAG"

for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Search sidecar up: ${CONTAINER} (port ${PORT}, retrieval -> host:${RETRIEVAL_PORT})"
        exit 0
    fi
    sleep 1
done

echo "Search sidecar failed to become healthy within 30s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
