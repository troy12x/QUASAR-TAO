<div align="center">

## QUASAR-SUBNET

**Long-context kernel optimization & inference verification subnet on Bittensor**

[![QUASAR](./banner.png)](https://github.com/SILX-LABS/QUASAR-SUBNET)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

  - [Overview](#overview)
- [Architecture](#architecture)
  - [Validator API](#validator-api)
  - [Miner Neuron](#miner-neuron)
  - [Validator Neuron](#validator-neuron)
  - [Miner Inference Server](#miner-inference-server)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Local: Run Full Stack](#local-run-full-stack)
  - [Ports & Conflicts](#ports--conflicts)
- [Testing Guide](#testing-guide)
- [Docker & Image Publishing](#docker--image-publishing)
- [Key Environment Variables](#key-environment-variables)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

QUASAR-SUBNET is a Bittensor subnet that:

- Optimizes **flash-linear attention kernels** via miner submissions
- Uses a centralized **Validator API** to orchestrate rounds, commit–reveal, and scoring
- Verifies miner behavior via a **logit-level inference verification** protocol
- Supports both **local miners** and **Docker-based miner inference servers**

Miners:
- Fork and optimize a target repository (e.g. `troy12x/flash-linear-attention`)
- Run kernel benchmarks and report results to the Validator API
- Expose an HTTP inference server (locally or via Docker) for logit verification

Validators:
- Coordinate benchmarking rounds through the Validator API
- Download and evaluate miner submissions (including verification of logits)
- Submit final weights to the Bittensor network.

---

## Architecture

### Validator API

Location: `validator_api/`  
Entry point: `validator_api/app.py`

- Central FastAPI service used by miners and validators
- Stores benchmark results, commit–reveal state, and miner metadata
- Backed by PostgreSQL or SQLite via `DATABASE_URL`
- Deployed in production using Docker/Render (see `render.yaml` and `Dockerfile`)

Local startup:

```bash
./START_SERVER.sh
# or explicitly
source .venv/bin/activate
PORT=8000 HOST=0.0.0.0 uvicorn validator_api.app:app --host "$HOST" --port "$PORT" --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

### Miner Neuron

Location: `neurons/miner.py`  
Role: Bittensor neuron that:

- Connects to the Bittensor network (`--subtensor.network`, `--netuid`)
- Forks the target GitHub repo using `GITHUB_TOKEN` and `GITHUB_USERNAME`
- Runs optimization loops on `chunk.py` and related kernels
- Submits results to the Validator API
- Optionally runs in **test mode** for local development

Typical local run (using `.env`):

```bash
./START_MINER.sh
```

Under the hood this is equivalent to:

```bash
python -m neurons.miner \
  --wallet.name "$WALLET_MINER_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY" \
  --subtensor.network "$SUBTENSOR_NETWORK" \
  --netuid "$NETUID"
```

> **Note**: The miner requires `GITHUB_TOKEN` (with `repo` scope) and `GITHUB_USERNAME` in `.env`.

### Validator Neuron

Location: `neurons/validator.py`  
Role: Bittensor neuron that:

- Polls the Validator API for new submissions
- Validates miner code, imports, and kernel behavior
- Runs benchmark tasks and computes rewards
- Verifies miner outputs using **logit verification** against a reference model

Local run (using `.env`):

```bash
./START_VALIDATOR.sh
```

This configures:

- `NETUID`
- `SUBTENSOR_NETWORK`
- `WALLET_VALIDATOR_NAME`
- `WALLET_HOTKEY`
- Inference verification parameters (`ENABLE_LOGIT_VERIFICATION`, `REFERENCE_MODEL`, thresholds)
- Commit–reveal timings (`BLOCKS_UNTIL_REVEAL`, `BLOCK_TIME_SECONDS`)

### Miner Inference Server

Location (Python implementation): `miner/inference_server.py`  
Location (OCI/Docker build): `docker-build/` and `miner/Dockerfile.inference`

This server exposes the **logit verification** interface:

- `POST /inference` – runs generation & returns captured logits
- `GET /health` – health check
- `GET /model_info` – basic metadata

Local dev startup (bound to port 8001 to avoid conflicts):

```bash
./START_MINER_INFERENCE.sh
```

Containerized startup (built image):

```bash
docker run -d \
  --name quasar-miner-inference \
  --gpus all \
  -p 8001:8000 \
  dockerhub_username/quasar-miner:latest
```

---

## Quick Start

### Environment Setup

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Set up environment variables**:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and fill in at least:
# - GITHUB_TOKEN (required for miners)
# - GITHUB_USERNAME (required for miners)
# - DOCKER_USERNAME (if building Docker images)
```

Required variables for miners:

```bash
GITHUB_TOKEN=ghp_...          # GitHub PAT with repo scope
GITHUB_USERNAME=your_username
DOCKER_USERNAME=your_dockerhub_username
```

Key Bittensor settings:

```bash
SUBTENSOR_NETWORK=test        # or finney for mainnet
NETUID=383
WALLET_MINER_NAME=Your_miner_wallet
WALLET_VALIDATOR_NAME=Your_validator_wallet
WALLET_HOTKEY=default
```

### Local: Run 

Use separate terminals:

1. **Validator API**

   ```bash
   ./START_SERVER.sh
   ```

2. **Miner Neuron**

   ```bash
   ./START_MINER.sh
   ```

3. **Validator Neuron**

   ```bash
   ./START_VALIDATOR.sh
   ```

4. (Optional) **Miner Inference Server for verification testing**

   ```bash
   ./START_MINER_INFERENCE.sh
   ```

### Ports & Conflicts

To avoid crashes due to reused ports, we standardize on:

| Service                 | Default Port (host) | Notes                              |
|-------------------------|---------------------|------------------------------------|
| Validator API           | `8000`              | Set via `PORT` env var             |
| Miner Inference Server  | `8001`              | Set via `PORT` env var             |
| Challenge container     | `8080`              | From `docker-compose.yml`          |
| Miner neuron axon       | auto / `8091`       | Can be set via `--axon.port`       |
| Validator neuron axon   | auto / `8092`       | Can be set via `--axon.port`       |

If you see `address already in use`:

- Check which process is using the port (`lsof -i :8000` / `:8001`)
- Either stop that process or override the port:

```bash
export PORT=8002
uvicorn validator_api.app:app --host 0.0.0.0 --port "$PORT"

export PORT=8003
python miner/inference_server.py
```

---

## Docker & Image Publishing

The miner inference container is defined using **rules_oci** in `docker-build/BUILD.bazel`.

### Deployment Options

- **Runpod (GPU)**: Use Bazel to build and push images (no Docker daemon required)
- **Render (CPU)**: Deploy via `render.yaml` using Dockerfile
- **Local**: Use Docker directly with Dockerfiles


### Building & Pushing with Bazel (Runpod)

1. Ensure `DOCKER_USERNAME` is set in `.env`:

   ```bash
   DOCKER_USERNAME=your_dockerhub_username
   ```

2. From `docker-build/`, update the Bazel config and push:

   ```bash
   cd docker-build
   ./load_config_from_env.sh
   
   # GPU image (for Runpod)
   bazel run //:push_miner_image_gpu
   
   # CPU image (for Render)
   bazel run //:push_miner_image_cpu
   
   # Or use interactive script
   ./push_miner.sh
   ```

This will push:
- `index.docker.io/$DOCKER_USERNAME/quasar-miner-gpu:latest` (CUDA)
- `index.docker.io/$DOCKER_USERNAME/quasar-miner-cpu:latest` (CPU)

### Manual Docker Build (Local)

**GPU Image**:
```bash
cd miner
docker build -f Dockerfile.inference -t quasar-miner-gpu:latest .

docker run -d --gpus all -p 8001:8000 \
  -e MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
  quasar-miner-gpu:latest
```

**CPU Image**:
```bash
docker build -f miner/Dockerfile.render -t quasar-miner-cpu:latest .

docker run -d -p 8001:8000 \
  -e MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
  -e DEVICE=cpu \
  quasar-miner-cpu:latest
```

---

## Key Environment Variables

Defined in `.env`:

| Variable                | Default / Example                             | Used By                 |
|-------------------------|-----------------------------------------------|-------------------------|
| `DATABASE_URL`          | `postgresql://...` or `sqlite:///...`        | Validator API           |
| `VALIDATOR_API_URL`     | `http://localhost:8000`                       | Miners, Validators      |
| `GITHUB_TOKEN`          | `ghp_...`                                     | Miner                   |
| `GITHUB_USERNAME`       | `gh_username`                                 | Miner                   |
| `GITHUB_FORK_NAME`      | `flash-linear-attention`                      | Miner                   |
| `TARGET_SEQUENCE_LENGTH`| `100000`                                      | Miner                   |
| `AGENT_ITERATIONS`      | `100`                                         | Miner                   |
| `OPTIMIZATION_INTERVAL` | `300` (seconds)                               | Miner                   |
| `NETUID`                | `24`                                          | All neurons             |
| `SUBTENSOR_NETWORK`     | `test` / `finney`                             | All neurons             |
| `WALLET_MINER_NAME`     | `quasar_miner`                                | Miner                   |
| `WALLET_VALIDATOR_NAME` | `quasar_validator`                            | Validator               |
| `WALLET_HOTKEY`         | `default`                                     | All neurons             |
| `HOST`                  | `0.0.0.0`                                     | Validator API, inference|
| `PORT`                  | `8000`                                        | Validator API           |
| `MINER_INFERENCE_PORT`  | `8000` (container), `8001` (host recommended) | Inference server        |
| `ENABLE_LOGIT_VERIFICATION` | `true`                                   | Validator neuron        |
| `REFERENCE_MODEL`       | `Qwen/Qwen2.5-0.5B-Instruct`                  | Validator & inference   |
| `COSINE_SIM_THRESHOLD`  | `0.99`                                        | Validator neuron        |
| `MAX_ABS_DIFF_THRESHOLD`| `0.1`                                         | Validator neuron        |
| `BLOCKS_UNTIL_REVEAL`   | `100`                                         | Commit–reveal           |
| `BLOCK_TIME_SECONDS`    | `12`                                          | Commit–reveal           |
| `DOCKER_USERNAME`       | `dokerhub_username`                                 | Bazel / oci_push        |

---

## Contributing

We welcome contributions of all kinds:

- Improving kernel optimization strategies
- Enhancing validator scoring and metrics
- Extending the inference verification pipeline
- Docker and deployment improvements
- Documentation and testing

To contribute:

1. Fork the repo and create a feature branch
2. Make your changes with tests where applicable
3. Open a PR with a clear description and rationale

You can also join the Bittensor community on Discord to discuss ideas and get support.

---

## License

QUASAR-SUBNET is released under the [MIT License](./LICENSE).

