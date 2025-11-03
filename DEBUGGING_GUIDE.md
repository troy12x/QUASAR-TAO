# Miner Initialization Debugging Guide

## The Issue
Your miner hangs after showing:
```
2025-11-03 21:09:52.094 | WARNING | You are allowing non-validators to send requests to your miner. This is a security risk.
/home/ubuntu/bt/lib/python3.12/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True...
```

## What Happens Now

With the enhanced logging, you'll see **exactly where it gets stuck**. The initialization follows this sequence:

### 1. **HFAMiner Initialization**
```
================================================================================
🚀 Initializing HFAMiner...
================================================================================
📞 Calling BaseMinerNeuron.__init__...
```

### 2. **BaseMinerNeuron Initialization**
```
================================================================================
🏗️ Initializing BaseMinerNeuron...
================================================================================
📞 Calling BaseNeuron.__init__...
```

### 3. **Wallet & Network Setup** (LIKELY WHERE IT HANGS)
```
================================================================================
🔧 Setting up bittensor objects...
================================================================================
💼 Creating wallet...
✅ Wallet created: Wallet(...)
🌐 Connecting to subtensor (network: test)...
   This may take 10-30 seconds...
✅ Subtensor connected in X.XXs
📊 Downloading metagraph for netuid 439...
   This may take 30-60 seconds on first run...
✅ Metagraph downloaded in X.XXs
   Metagraph has XXX neurons
```

### 4. **Registration Check**
```
🔍 Checking if hotkey is registered on network...
   Hotkey: 5...
   Netuid: 439
✅ Hotkey is registered on network
```

### 5. **Axon Creation**
```
⚙️ Checking security settings...
🔌 Creating axon for handling requests...
✅ Axon created
🔗 Attaching forward function to miner axon...
✅ Axon configured
```

### 6. **Device & Model Setup**
```
🖥️ Detecting compute device...
✅ Using device: cuda
   GPU: Tesla T4
   GPU Memory: 15.00 GB
🏭 Initializing model factory...
✅ Model factory initialized
📊 Initializing performance tracking...
✅ Performance tracking initialized
```

### 7. **Model Loading** (if not skipped)
```
================================================================================
🔄 STARTING MODEL LOADING PROCESS
================================================================================
📋 Step 1/5: Getting model configurations...
[... detailed model loading logs ...]
```

## Most Likely Hang Points

### **Point 1: Metagraph Download (90% probability)**
```
📊 Downloading metagraph for netuid 439...
   This may take 30-60 seconds on first run...
```
**Why:** First time downloading the full metagraph from the blockchain
**Solution:** Just **wait 60-120 seconds**. It's normal on first run.

### **Point 2: Subtensor Connection (5% probability)**
```
🌐 Connecting to subtensor (network: test)...
   This may take 10-30 seconds...
```
**Why:** Network issues, firewall, or testnet unavailable
**Solution:** Check internet connection, try mainnet

### **Point 3: Model Loading (5% probability)**
```
[ModelFactory] Instantiating hfa model...
```
**Why:** Large model being created, GPU memory allocation
**Solution:** Use `--model.skip_loading` flag

## What To Do

### Run the miner again:
```bash
python scripts/run_miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439
```

### Watch the logs and note the LAST message you see

The last message will tell you exactly where it's stuck:

- **Last message: "Downloading metagraph..."** → Wait 2 minutes, it's downloading
- **Last message: "Connecting to subtensor..."** → Network/connection issue
- **Last message: "Instantiating model..."** → Model creation is slow
- **Last message: "Moving model to device..."** → GPU issue

## Quick Solutions

### If stuck on metagraph download:
**Just wait.** First run can take 1-2 minutes. Subsequent runs will be instant.

### If you want to skip models and just test connectivity:
```bash
python scripts/run_miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439 \
  --model.skip_loading
```

This will start the miner in seconds without loading any models.

### If it's truly frozen (>5 minutes):
1. Press Ctrl+C
2. Share the **last log message** you saw
3. We can diagnose the specific issue

## Expected Timeline

- **Wallet creation:** < 1 second
- **Subtensor connection:** 5-15 seconds
- **Metagraph download (first run):** 30-120 seconds ⚠️
- **Metagraph download (subsequent):** 1-5 seconds
- **Registration check:** 2-5 seconds
- **Axon creation:** < 1 second
- **Model loading (if enabled):** 30-60 seconds per model

**Total first run:** ~2-3 minutes
**Total subsequent runs:** ~1-2 minutes (or 10 seconds with --model.skip_loading)
