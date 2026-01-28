# QUASAR Deployment Guide

## Quick Deployment

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/your-org/QUASAR-TAO
cd QUASAR-TAO/hfa_subnet

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Create Wallet

```bash
# Create validator wallet
btcli wallet create --wallet.name validator --wallet.hotkey default

# Get TAO for registration (testnet)
# Visit: https://faucet.bittensor.com/

# Register on subnet
btcli subnet register --netuid 439 --wallet.name validator --wallet.hotkey default
```

### 3. Test Locally (Mock Mode)

```bash
# Run validator in mock mode
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --mock \
  --logging.debug
```

**Expected Output**:
```
🔄 Loading Qwen/Qwen2.5-0.5B-Instruct for mock inference...
✅ Model loaded successfully on cuda:0
📋 Task: narrativeqa_7389 [32k] | Len: 13619
💰 Rewards: 0.5234 | 🎯 Accuracy: 0.5234
```

### 4. Deploy to Testnet

```bash
# Run validator on testnet
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network test \
  --netuid 439 \
  --logging.info
```

### 5. Deploy to Mainnet

```bash
# IMPORTANT: Test thoroughly on testnet first!

# Run validator on mainnet
python neurons/validator.py \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --logging.info
```

---

## Miner Deployment

### 1. Setup

```bash
# Create miner wallet
btcli wallet create --wallet.name miner --wallet.hotkey default

# Register on subnet
btcli subnet register --netuid 439 --wallet.name miner --wallet.hotkey default
```

### 2. Run Miner

```bash
python neurons/miner.py \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network finney \
  --netuid 439 \
  --axon.port 8091 \
  --logging.info
```

---

## Configuration

### WandB Setup

```bash
# Login to WandB
wandb login

# Your API key will be auto-detected
# Entity will be auto-detected from your account
```

### Custom Config

Edit `hfa_config.json` and `subnet_config.json` to customize:
- Enabled benchmarks
- Context length tests
- Scoring weights
- Evaluation parameters

---

## Monitoring

### Check Validator Status

```bash
# View logs
tail -f ~/.bittensor/miners/validator/default/netuid439/validator/logs/validator.log

# Check WandB dashboard
# Visit: https://wandb.ai/your-entity/quasar-long-context-subnet
```

### Check Miner Status

```bash
# View logs
tail -f ~/.bittensor/miners/miner/default/netuid439/miner/logs/miner.log
```

---

## Troubleshooting

### Common Issues

**Issue**: Model fails to load in mock mode
```bash
# Solution: Ensure you have enough GPU memory
# Qwen2.5-0.5B requires ~2GB VRAM
```

**Issue**: WandB not logging
```bash
# Solution: Check WandB login
wandb login
```

**Issue**: Validator not finding miners
```bash
# Solution: Ensure you're on the correct network
btcli subnet list --netuid 439
```

---

## Production Checklist

- [ ] Tested in mock mode locally
- [ ] Tested on testnet for 24+ hours
- [ ] Verified WandB logging works
- [ ] Confirmed wallet has sufficient TAO
- [ ] Set up monitoring/alerts
- [ ] Documented any custom configurations
- [ ] Ready for mainnet deployment

---

## Support

For issues:
1. Check logs first
2. Review [README.md](README.md)
3. Join [Discord](https://discord.gg/bittensor)
4. Create GitHub issue with logs
