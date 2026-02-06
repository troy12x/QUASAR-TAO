#!/bin/bash
# Load DOCKER_USERNAME from project .env and write docker_config.bzl
# Run this before 'bazel run //:push_miner_image' (or use push_miner.sh)

cd "$(dirname "$0")/.."
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DOCKER_USERNAME=${DOCKER_USERNAME:-your_dockerhub_username}

cat > docker-build/docker_config.bzl << EOF
# Auto-generated from .env - do not edit manually
DOCKER_USERNAME = "$DOCKER_USERNAME"
EOF

echo "✅ docker_config.bzl updated with DOCKER_USERNAME=$DOCKER_USERNAME"
