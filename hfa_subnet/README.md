# QUASAR Long Context Subnet

[![Discord](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Professional long-context evaluation subnet on Bittensor**

---

## Overview

QUASAR is a Bittensor subnet that evaluates miners' ability to process and understand ultra-long context sequences (32k to 2M tokens). Validators test miners using real-world benchmarks and reward them based on accuracy.

### Key Features

- Real Benchmarks: LongBench tasks (NarrativeQA, Qasper, GovReport, etc.)
- Context Scaling: Tests from 32k to 2M tokens
- Accuracy-Based Rewards: Rewards directly proportional to performance
- Mock Mode: Test locally with real model inference
- WandB Integration: Track accuracy and rewards over time

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Bittensor wallet

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/QUASAR-TAO
cd QUASAR-TAO/hfa_subnet

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Validator

**Mock Mode (Local Testing)**:
```bash
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --mock \
  --logging.debug
```

**Testnet**:
```bash
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network test \
  --netuid 439
```

**Mainnet**:
```bash
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439
```

### Running Miner

**Choose Your Model**:
```bash
# Default (lightweight)
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091

# High performance (Kimi)
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091 \
  --miner.model_name "moonshotai/Kimi-Linear-48B-A3B-Instruct"

# Advanced reasoning (Qwen3)
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091 \
  --miner.model_name "Qwen/Qwen3-Next-80B-A3B-Thinking"
```

**Supported Models**:
- `silx-ai/Quasar-2M-Base` (default, 2M context specialist)
- `moonshotai/Kimi-Linear-48B-A3B-Instruct` (high performance, 48B params)
- `Qwen/Qwen3-Next-80B-A3B-Thinking` (advanced reasoning, 80B params)

---

## 📋 How It Works

### Validator Flow

1. **Task Selection**: Randomly select benchmark task from LongBench
2. **Miner Query**: Send context + question to miners
3. **Response Evaluation**: Calculate accuracy using dataset-specific metrics
4. **Reward Calculation**: Apply context-length multipliers
5. **Weight Update**: Update miner scores based on performance

### Reward System

Rewards are **directly proportional to accuracy**:

```python
# 1. Calculate accuracy (0.0 to 1.0)
accuracy = metric_fn(response, expected_answer)

# 2. Apply context-length multiplier
multiplier = {
    "32k": 1.0,    # Baseline
    "124k": 1.2,   # +20%
    "512k": 1.5,   # +50%
    "1.5m": 1.8,   # +80%
    "2m": 2.0      # +100%
}[bucket]

# 3. Final reward
reward = min(accuracy * multiplier, 1.0)
```

**Examples**:
- 80% accuracy on 32k context → 0.80 reward
- 50% accuracy on 2M context → 1.00 reward (capped)
- 10% accuracy on 32k context → 0.10 reward

---

## 🧪 Mock Mode

Test your validator locally with real model inference:

```bash
python neurons/validator.py --mock --logging.debug
```

**Features**:
- Loads configurable language model for inference
- Generates real responses for testing
- Calculates actual accuracy metrics
- Logs performance to WandB
- No blockchain connection required

**Expected Output**:
```
🔄 Loading Qwen/Qwen2.5-0.5B-Instruct for mock inference...
✅ Model loaded successfully on cuda:0
📋 Task: narrativeqa_7389 [32k] | Len: 13619
💰 Rewards: 0.5234 | 🎯 Accuracy: 0.5234 | Top5: ['0.523', '0.612', ...]
```

---

## 📊 Monitoring

### WandB Metrics

The validator logs the following metrics to Weights & Biases:

- `accuracy/{bucket}`: Per-bucket accuracy (32k, 124k, 512k, 1.5m, 2m)
- `avg_accuracy`: Overall accuracy across all buckets
- `rewards/{bucket}`: Normalized rewards per bucket
- `context_length`: Task context length
- `global_difficulty`: Current difficulty level

### Local Logs

Monitor validator progress:
```bash
tail -f ~/.bittensor/miners/validator/default/netuid439/validator/logs/validator.log
```

---

## 🛠️ Configuration

### Validator Config

Edit `hfa_config.json` and `subnet_config.json` to customize:

- Enabled benchmarks
- Context length tests
- Scoring weights
- Evaluation cycle duration

### Key Parameters

```python
# neurons/validator.py
REWARD_MULTIPLIERS = {
    "32k": 1.0,
    "124k": 1.2,
    "512k": 1.5,
    "1.5m": 1.8,
    "2m": 2.0
}

PENALTY_NO_RESPONSE = 0.0
PENALTY_FAKE = -0.5
```

---

## 📁 Project Structure

```
hfa_subnet/
├── neurons/
│   ├── validator.py          # Validator logic
│   └── miner.py               # Miner logic
├── template/
│   ├── protocol.py            # Synapse definitions
│   ├── mock.py                # Mock mode implementation
│   ├── benchmarks/
│   │   ├── benchmark_loader.py
│   │   └── metrics.py
│   └── base/
│       ├── validator.py       # Base validator class
│       └── neuron.py          # Base neuron class
├── hfa_config.json            # HFA architecture config
├── subnet_config.json         # Subnet parameters
└── requirements.txt           # Python dependencies
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- [Bittensor Documentation](https://docs.bittensor.com/)
- [Discord](https://discord.gg/bittensor)
- [Taostats](https://taostats.io/)

---

## 💡 Support

For questions or issues:
1. Check existing [GitHub Issues](https://github.com/your-org/QUASAR-TAO/issues)
2. Join our [Discord](https://discord.gg/bittensor)
3. Create a new issue with detailed information
