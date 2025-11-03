# Enhanced Logging Guide

## What Was Added

Comprehensive logging has been added throughout the model loading pipeline to help debug where the miner gets stuck.

### Logging Locations

#### 1. **Miner Initialization** (`neurons/miner.py`)
- Progress indicators for each step (Step 1/5, 2/5, etc.)
- Timestamps showing duration of each operation
- Model configuration details
- Device placement logging
- Clear separators (=== and ---) for readability

#### 2. **Model Factory** (`template/model_factory.py`)
- Import timing for each model class
- Configuration validation steps
- Model instantiation timing
- Parameter counts
- Error tracebacks for failures

#### 3. **Lazy Loading** (`template/model_factory.py`)
- Individual import timing for each architecture
- Module loading progress
- Registry status

## Log Output Format

When you run the miner, you'll now see detailed output like:

```
================================================================================
🔄 STARTING MODEL LOADING PROCESS
================================================================================
📋 Step 1/5: Getting model configurations...
✓ Got configurations in 0.05s for: ['hfa', 'simplemind']
   Config details: 2 architectures
--------------------------------------------------------------------------------
📦 Step 2/5: Loading model 1/2: hfa
   Config keys: ['vocab_size', 'd_model', 'num_layers', ...]
🔧 Step 3/5: Calling model factory for hfa...
   Factory class: ModelArchitectureFactory
[ModelFactory] create_model called for: hfa
[ModelFactory] Lazy loading model classes...
[ModelFactory]   Importing hfa from .models.hfa_model...
[ModelFactory]   Module imported in 0.12s
[ModelFactory]   Getting class HFAModel...
[ModelFactory]   ✅ Loaded hfa model class
[ModelFactory] Model registry has 4 entries
[ModelFactory] Validating config for hfa...
[ModelFactory] Config validated successfully
[ModelFactory] Getting model class for hfa...
[ModelFactory] Model class: HFAModel
[ModelFactory] Instantiating hfa model...
   🏗️ Creating hfa model instance...
[ModelFactory] Model instantiated in 2.34s
✓ Model created in 2.45s
✅ hfa model object created successfully
📍 Step 4/5: Moving hfa model to device: cuda
   ✓ Moved to device in 0.89s
⚙️ Step 5/5: Setting hfa model to eval mode...
   ✓ Model ready for inference
```

## What to Look For

### If the miner hangs, the last log message will tell you where:

1. **"Getting model configurations..."** → Config loading issue
2. **"Importing X from..."** → Module import hanging (missing dependency?)
3. **"Instantiating X model..."** → Model __init__ hanging (large model creation)
4. **"Moving X model to device..."** → GPU/memory issue
5. **"Setting X model to eval mode..."** → Model.eval() issue

### Common Issues

**Hangs at "Instantiating model"**
- Large model with many parameters being initialized
- Memory allocation issue
- Solution: Use `--model.skip_loading` flag to skip models

**Hangs at "Moving to device"**
- GPU memory full
- CUDA initialization issue
- Solution: Check GPU with `nvidia-smi`, free memory, or use CPU

**Hangs at "Importing from"**
- Missing dependency
- Circular import
- Solution: Check the specific model file for import errors

## Running the Miner

### With Enhanced Logging (Normal Mode)
```bash
python scripts/run_miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439
```

You'll see detailed step-by-step progress. **Wait 30-60 seconds** for model loading.

### Quick Start (Skip Models)
```bash
python scripts/run_miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439 \
  --model.skip_loading
```

Starts immediately, uses mock responses.

### With Debug Level Logging
```bash
python scripts/run_miner.py \
  --wallet.name hfa_silx \
  --wallet.hotkey hfa_silx_hot \
  --subtensor.network test \
  --netuid 439 \
  --logging.debug
```

Even more detailed output including debug messages.

## Timeout Protection

A 30-second timeout has been added for each model creation on Linux/Mac systems. If a model takes longer than 30 seconds to create, it will be skipped and logged as a timeout error.

**Note:** Timeout doesn't work on Windows. If on Windows and it hangs, use Ctrl+C to stop.

## Next Steps

1. Run the miner and watch the logs
2. Note the last log message before it hangs (if it does)
3. Share that information to diagnose the specific issue
4. Use `--model.skip_loading` if you just want to test the miner's connectivity
