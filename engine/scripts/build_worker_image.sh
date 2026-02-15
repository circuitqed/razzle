#!/bin/bash
set -euo pipefail

IMAGE="ghcr.io/circuitqed/razzle-worker:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ENGINE_DIR"

echo "Building worker image: $IMAGE"
docker build -f Dockerfile.worker -t "$IMAGE" .

echo ""
echo "Build complete. To push:"
echo "  docker push $IMAGE"
echo ""
echo "To test locally:"
echo "  docker run --rm $IMAGE python -c \"from razzle.ai.network import RazzleNet; print('ok')\""
