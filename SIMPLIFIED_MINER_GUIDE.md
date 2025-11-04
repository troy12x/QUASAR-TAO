# Simplified Miner Guide

## What Changed

Your miner has been simplified to follow the **Bittensor template pattern**:

### ✅ Simple, Direct Execution
- No wrapper scripts needed
- Run directly: `python neurons/miner.py`
- Uses built-in Bittensor initialization

### ✅ Minimal Code
- Removed complex model factory pattern
- Direct model imports (HFA, SimpleMind)
- Clean, readable initialization

### ✅ Context Manager Pattern
- Uses `with HFAMiner() as miner:` pattern
- Proper resource management
- Graceful shutdown

### ✅ Built-in Logging
- Uses Bittensor's native logging
- No custom logging layers
- Clear, simple output

## How to Run

### Basic Command (Recommended)
```bash
cd ~/QUASAR-TAO/bittensor_subnet/hfa_subnet
python neurons/miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439
```

### With Python Module
```bash
cd ~/QUASAR-TAO/bittensor_subnet/hfa_subnet
python -m neurons.miner \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439
```

## What You'll See

```
🚀 Initializing HFAMiner models...
Using device: cuda
Loading models...
Loading HFA model...
✅ HFA model loaded: 123,456,789 parameters
Loading SimpleMind model...
✅ SimpleMind model loaded: 98,765,432 parameters
Default model set to HFA
✅ Miner ready with 2 models: ['hfa', 'simplemind']
Miner running... 1730000000.0
Miner running... 1730000005.0
...
```

## Architecture

### Models Loaded
1. **HFA (Hierarchical Flow Anchoring)**
   - 100% memory retention
   - O(n) complexity
   - 6 layers, 8 heads
   - 50K vocab size

2. **SimpleMind**
   - O(n) complexity
   - Dynamic routing
   - 6 layers, 64 channels
   - Learnable aggregation

### Default Behavior
- HFA is the default model (preferred)
- Falls back to SimpleMind if HFA fails
- Both models run on GPU if available

## Key Features

### Automatic Model Selection
The miner automatically picks the best model based on:
- Request context length
- Model availability
- Performance history

### Performance Tracking
Tracks stats per architecture:
- Request count
- Average processing time
- Success rate

### Fault Tolerance
- Continues running if model loading fails
- Logs warnings instead of crashing
- Falls back to available models

## Configuration

### Model Parameters (Hardcoded for Simplicity)
```python
hfa_config = {
    'vocab_size': 50257,      # GPT-2 vocab
    'd_model': 512,            # Model dimension
    'num_layers': 6,           # Transformer layers
    'num_heads': 8,            # Attention heads
    'd_ff': 2048,              # Feedforward dimension
    'max_seq_len': 100000,     # 100K context
    'dropout': 0.1,
}

simplemind_config = {
    'vocab_size': 50257,
    'd_model': 512,
    'num_layers': 6,
    'num_channels': 64,         # SimpleMind channels
    'router_type': 'dynamic',   # Dynamic routing
    'aggregation_type': 'learnable',
    'max_seq_len': 100000,
    'dropout': 0.1,
}
```

### Want to Customize?
Edit these configs directly in `neurons/miner.py` in the `_load_models()` method.

## Troubleshooting

### "Failed to load HFA model"
- HFA components might not be available
- Check if `template/models/hfa_model.py` exists
- Miner will fall back to SimpleMind

### "Failed to load SimpleMind model"
- SimpleMind components might not be available
- Check if `template/models/simplemind_model.py` exists
- Miner will still run but return mock responses

### "No models loaded"
- Both HFA and SimpleMind failed to load
- Check your Python environment
- Check for missing dependencies

### Out of GPU Memory
- Models are too large for your GPU
- Reduce `num_layers` or `d_model` in configs
- Or run on CPU (slower but works)

## File Structure

```
neurons/
  └── miner.py          ← Main miner (simplified!)
template/
  ├── base/
  │   └── miner.py      ← Base miner class (Bittensor)
  └── models/
      ├── hfa_model.py          ← HFA implementation
      └── simplemind_model.py   ← SimpleMind implementation
```

## Next Steps

1. **Run the miner** with the command above
2. **Wait for models to load** (30-60 seconds first time)
3. **Verify it's running** - should print "Miner running..." every 5 seconds
4. **Monitor performance** - check GPU usage with `nvidia-smi`

## Comparison: Old vs New

### Old Way (Complex)
```python
# Multiple layers of abstraction
config = get_config()
miner = HFAMiner(config=config)
# Factory pattern with lazy loading
# Complex configuration system
# Custom logging layers
miner.run()
```

### New Way (Simple)
```python
# Direct, template-style
with HFAMiner() as miner:
    while True:
        bt.logging.info(f"Miner running... {time.time()}")
        time.sleep(5)
```

## Benefits

✅ **Faster startup** - Direct model loading
✅ **Easier to debug** - Less abstraction layers
✅ **Cleaner code** - Follows Bittensor conventions
✅ **More maintainable** - Standard patterns
✅ **Better logging** - Native Bittensor logging

---

**Ready to run?** Just execute:
```bash
python neurons/miner.py --wallet.name hfa_silx --wallet.hotkey hfa_silx_hot --subtensor.network test --netuid 439
```
