#!/bin/bash
# One-click launcher for the tau2 orchestration sidecar. Built on the `tau2`
# pip package's own Orchestrator/LLMAgent/UserSimulator/evaluate_simulation
# -- docker/server.py wires those together directly, no protocol re-
# implemented.
#
# Same "one long-lived container, one fixed port for the whole job" pattern
# as ../webshop/run_webshop_sidecar.sh / ../alfworld/run_alfworld_sidecar.sh;
# see docker/Dockerfile's docstring for why this needs its OWN container
# (tau2-bench's Python>=3.12 requirement, vs the training image's 3.10).
#
# Idempotent: if the container is already running and healthy, this is a
# no-op.
#
# UNLIKE webshop/alfworld (which are only ever CALLED INTO from the training
# process), this sidecar also calls OUT to whatever `base_url` a /run request
# supplies -- the session-server proxy sitting in front of the live rollout
# engine (miles/rollout/session/server.py), which this repo's single-node
# colocate setup always binds to 127.0.0.1 on the HOST (see MASTER_ADDR in
# every run script here). A bridge-networked container's own 127.0.0.1 is
# its OWN loopback, not the host's -- it could never reach that proxy. Using
# --network=host instead makes this container share the host's network
# namespace outright, so 127.0.0.1 means the same thing on both sides with
# zero address-translation logic (the more complex alternative
# swe-agent-v2's MILES_ROUTER_EXTERNAL_HOST env var exists to work around,
# for a genuinely multi-host setup this repo's tau2 integration doesn't need).
#
#   PORT      (default 8424)          port this sidecar listens on (127.0.0.1 only)
#   IMAGE_TAG (default sdpo-react-tau2) local image tag
#   CONTAINER (default sdpo-react-tau2) container name
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8424}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-tau2}"
CONTAINER="${CONTAINER:-sdpo-react-tau2}"

# Same host-vs-enroot split as ../run_sandbox.sh: the training job itself has
# no `docker` binary and doesn't need one -- this sidecar is a host-level
# singleton, started BEFORE the training/enroot session, reached over HTTP.
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "tau2 sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND tau2 sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_tau2_sidecar.sh (see ../../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "tau2 sidecar already running and healthy: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

# --network=host means the container's own uvicorn must bind directly to
# whatever PORT this launcher was given -- no -p remapping exists in this
# mode. TAU2_SIDECAR_PORT (Dockerfile's ENV, read by its shell-form CMD)
# carries the override in.
docker run -d --rm \
    --name "$CONTAINER" \
    --network=host \
    -e "TAU2_SIDECAR_PORT=${PORT}" \
    "$IMAGE_TAG"

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "tau2 sidecar up: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    sleep 1
done

echo "tau2 sidecar failed to become healthy within 60s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
