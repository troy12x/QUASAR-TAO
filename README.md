<div align="center">

# QUASAR: Long Context Foundation Models & Evaluation Subnet

[![QUASAR](./banner.png)](https://github.com/your-org/QUASAR-TAO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## Shaping the Future of Long-Context Understanding

</div>

---

- [Introduction](#introduction)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
  - [Miners](#miners)
  - [Validators](#validators)
- [Evaluation Process](#evaluation-process)
  - [Benchmark Tasks](#benchmark-tasks)
  - [Scoring System](#scoring-system)
  - [Reward Calculation](#reward-calculation)
- [Supported Models](#supported-models)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Welcome to QUASAR, a Bittensor subnet dedicated to advancing long-context language models through comprehensive evaluation and incentivization. As AI systems tackle increasingly complex tasks requiring understanding of extensive documents, code repositories, and conversations, the ability to process ultra-long contexts (32k to 2M tokens) becomes critical.

QUASAR provides a decentralized evaluation framework where miners compete by running state-of-the-art long-context models, and validators assess their performance using real-world benchmarks. Our main product is the QUASAR foundation models—built specifically to process millions of tokens—which serve as the benchmark for evaluating all models on long-context capabilities. By harnessing the Bittensor network, we're building the most comprehensive long-context evaluation system in existence while extending model capabilities for handling increasingly large contexts.

## Key Features

- **Real-World Benchmarks**: LongBench tasks including NarrativeQA, Qasper, GovReport, and more
- **Context Scaling**: Evaluations from 32k to 2M tokens
- **Fair Rewards**: Accuracy-based scoring directly proportional to performance
- **Model Diversity**: Support for multiple architectures (Qwen, Kimi, Llama)
- **Transparent Metrics**: WandB integration for real-time performance tracking
- **Mock Mode**: Local testing with real model inference

## How It Works

### Miners

Miners run long-context language models and respond to benchmark evaluation requests from validators. Key responsibilities:

- Load and maintain a long-context capable model
- Process evaluation requests with context lengths up to 2M tokens
- Generate accurate responses to questions based on provided context
- Optimize for both accuracy and inference speed

Supported models include:
- silx-ai/Quasar-2M-Base (2M context specialist)
- moonshotai/Kimi-Linear-48B-A3B-Instruct (48B parameters)
- Qwen/Qwen3-Next-80B-A3B-Thinking (80B parameters with advanced reasoning)

### Validators

Validators evaluate miner performance using standardized benchmarks:

- Select random tasks from LongBench dataset
- Send context + question to miners
- Calculate accuracy using dataset-specific metrics (F1, EM, ROUGE)
- Apply context-length multipliers to reward harder tasks
- Update miner scores based on performance

## Evaluation Process

### Benchmark Tasks

QUASAR uses tasks from the LongBench suite:

1. **Question Answering**
   - NarrativeQA: Story comprehension
   - Qasper: Scientific paper QA
   - MultiFieldQA: Multi-domain questions

2. **Summarization**
   - GovReport: Government document summaries
   - QMSum: Meeting summarization
   - MultiNews: Multi-document news summaries

3. **Classification**
   - TREC: Question classification
   - TriviaQA: Trivia questions

### Scoring System

```mermaid
flowchart LR
    A[Miner Response] --> B[Calculate Accuracy]
    B --> C[Apply Context Multiplier]
    C --> D[Clip to 0-1 Range]
    D --> E[Final Reward]
    
    style A fill:#5b9aa0
    style B fill:#e06377
    style C fill:#f0932b
    style D fill:#6ab04c
    style E fill:#30336b
```

### Reward Calculation

Rewards are calculated to be directly proportional to accuracy:

```python
# 1. Calculate raw accuracy (0.0 to 1.0)
accuracy = metric_fn(response, expected_answer)

# 2. Apply context-length multiplier
multipliers = {
    "32k": 1.0,    # Baseline
    "124k": 1.2,   # +20% bonus
    "512k": 1.5,   # +50% bonus
    "1.5m": 1.8,   # +80% bonus
    "2m": 2.0      # +100% bonus
}

# 3. Final reward (capped at 1.0)
reward = min(accuracy * multiplier, 1.0)
```

**Example**: A miner achieving 60% accuracy on a 2M token task receives:
- Raw accuracy: 0.60
- Multiplier: 2.0
- Final reward: min(0.60 × 2.0, 1.0) = 1.0

This incentivizes both accuracy and tackling longer contexts.

## Supported Models

Current supported models for miners:

| Model | Parameters | Context Length | Specialty |
|-------|-----------|----------------|-----------|
| silx-ai/Quasar-2M-Base | 26B | 2M tokens | Long context specialist |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 48B | 1M+ tokens | High performance |
| Qwen/Qwen3-Next-80B-A3B-Thinking | 80B | 128k tokens | Advanced reasoning |

## Running Miners and Validators

### Running a Miner

**Requirements**:
- Python 3.9+
- GPU with sufficient VRAM (varies by model)
- Bittensor wallet

**Setup**:
```bash
# Clone repository
git clone https://github.com/your-org/QUASAR-TAO
cd QUASAR-TAO/hfa_subnet

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run miner
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091 \
  --miner.model_name "silx-ai/Quasar-2M-Base"
```

**Tips for Better Performance**:
1. Use models optimized for long contexts
2. Ensure sufficient GPU memory for your chosen model
3. Monitor your scores via WandB
4. Optimize inference speed without sacrificing accuracy

### Running a Validator

**Requirements**:
- Python 3.9+
- CUDA-capable GPU
- Bittensor wallet with sufficient TAO for registration

**Setup**:
```bash
# Clone and install
git clone https://github.com/your-org/QUASAR-TAO
cd QUASAR-TAO/hfa_subnet
pip install -r requirements.txt
pip install -e .

# Test in mock mode first
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --mock \
  --logging.debug

# Run on mainnet
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439
```

**Recommended**:
- Set up WandB for monitoring: `export WANDB_API_KEY=<your key>`
- Use PM2 for process management
- Monitor validator logs regularly

## Roadmap

### Phase 1: Foundation (Q4 2024)
- [x] Launch QUASAR subnet on Bittensor testnet
- [x] Implement LongBench evaluation framework
- [x] Deploy mock mode for local testing
- [x] Integrate WandB monitoring

### Phase 2: Expansion (Q1 2025)
- [ ] Add support for additional long-context benchmarks
- [ ] Implement dynamic difficulty adjustment
- [ ] Expand supported model architectures
- [ ] Publish research paper on decentralized long-context evaluation

### Phase 3: Advanced Features (Q2 2025)
- [ ] Multi-modal long-context evaluation (text + images)
- [ ] Custom benchmark submission system
- [ ] Real-time leaderboard and analytics dashboard
- [ ] Integration with external AI research labs

### Phase 4: Ecosystem Growth (Q3 2025)
- [ ] Developer API for programmatic access
- [ ] Benchmark marketplace for custom evaluations
- [ ] Cross-subnet collaboration features
- [ ] Mobile and edge device support

## Contributing

We welcome contributions from the community! Whether you're a researcher, developer, or AI enthusiast, there are many ways to contribute:

- Submit new benchmark tasks
- Improve evaluation metrics
- Optimize miner implementations
- Enhance documentation
- Report bugs and suggest features

See our [contribution guidelines](./hfa_subnet/contrib/CONTRIBUTING.md) for more details.

Join our community on [Discord](https://discord.gg/bittensor) to connect with other contributors.

## License

QUASAR is released under the [MIT License](./LICENSE).

---

<div align="center">

**Building the future of long-context AI evaluation, together.**

</div>
