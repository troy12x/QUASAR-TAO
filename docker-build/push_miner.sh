#!/bin/bash
# Push miner image to Docker Hub
# Loads DOCKER_USERNAME from .env, updates docker_config.bzl, then runs bazel push

cd "$(dirname "$0")"
./load_config_from_env.sh
bazel run //:push_miner_image
