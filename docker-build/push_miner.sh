cd "$(dirname "$0")"
./load_config_from_env.sh
bazel run //:push_miner_image
