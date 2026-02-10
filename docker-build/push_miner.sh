#!/bin/bash
# Push miner images to Docker Hub using Bazel
# Supports both GPU and CPU images

cd "$(dirname "$0")"

# Load Docker username from .env
cd ..
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
cd docker-build

# Update docker_config.bzl
./load_config_from_env.sh

echo ""
echo "Which image to build and push?"
echo "1) GPU image (for Runpod)"
echo "2) CPU image (for Render)"
echo "3) Both"
read -p "Choice [1-3]: " choice

case $choice in
    1)
        echo "Building and pushing GPU image..."
        bazel run //:push_miner_image_gpu
        ;;
    2)
        echo "Building and pushing CPU image..."
        bazel run //:push_miner_image_cpu
        ;;
    3)
        echo "Building and pushing GPU image..."
        bazel run //:push_miner_image_gpu
        echo ""
        echo "Building and pushing CPU image..."
        bazel run //:push_miner_image_cpu
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Done! Images pushed to Docker Hub"
