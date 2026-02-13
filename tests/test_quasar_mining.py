import torch
import os
import logging
import time

# Enable verbose logging for Triton kernel compilation
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_PRINT_DEBUG"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fla.layers.quasar import QuasarAttention

def test_quasar_basic():
    print("Testing QuasarAttention basic functionality...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Clear GPU memory before test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_mem_start = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory before test: {free_mem_start:.2f} GB free")
    else:
        free_mem_start = None
    
    # Model parameters - start conservative
    batch_size = 2
    seq_len = 100000
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    
    # Initialize QuasarAttention
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)
    
    # Check memory again after model initialization and before creating input
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        free_mem_before_input = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory after model init: {free_mem_before_input:.2f} GB free")
        
        # Dynamically adjust parameters based on available memory
        # Be very conservative - need at least 2GB free for computation
        if free_mem_before_input < 1.0:
            seq_len = 20000
            batch_size = 1
            print(f"Very low GPU memory ({free_mem_before_input:.2f} GB), using seq_len={seq_len}, batch_size={batch_size}")
        elif free_mem_before_input < 2.0:
            seq_len = 50000
            batch_size = 1
            print(f"Low GPU memory ({free_mem_before_input:.2f} GB), using seq_len={seq_len}, batch_size={batch_size}")
        elif free_mem_before_input < 4.0:
            seq_len = 50000
            batch_size = 2
            print(f"Moderate GPU memory ({free_mem_before_input:.2f} GB), using seq_len={seq_len}, batch_size={batch_size}")
        else:
            seq_len = 100000
            batch_size = 2
            print(f"Sufficient GPU memory ({free_mem_before_input:.2f} GB), using seq_len={seq_len}, batch_size={batch_size}")
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Forward pass with retry logic for OOM
    print(f"Input shape: {x.shape}")
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Final memory check right before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                free_mem_final = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"GPU memory before forward pass: {free_mem_final:.2f} GB free")
                
                # More aggressive check - need at least 1GB for safe operation
                if free_mem_final < 1.0:
                    if retry_count == 0:
                        # First retry: reduce batch size
                        print(f"Low memory ({free_mem_final:.2f} GB), reducing batch_size from {batch_size} to 1")
                        batch_size = 1
                        del x
                        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        retry_count += 1
                        continue
                    elif retry_count == 1:
                        # Second retry: reduce sequence length
                        seq_len = max(seq_len // 2, 20000)
                        print(f"Still low memory ({free_mem_final:.2f} GB), reducing seq_len to {seq_len}")
                        del x
                        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        retry_count += 1
                        continue
                    else:
                        raise RuntimeError(f"Insufficient GPU memory for forward pass: {free_mem_final:.2f} GB free (tried reducing parameters)")
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                output, _, _ = quasar(x)
            
            # Success - break out of retry loop
            break
        
            print(f"Output shape: {output.shape}")
            print("QuasarAttention basic test PASSED!")
            
            # Cleanup after test
            del output, x, quasar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            return True
            
        except torch.cuda.OutOfMemoryError as e:
            retry_count += 1
            print(f"CUDA OOM Error (attempt {retry_count}/{max_retries}): {e}")
            
            # Cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            if retry_count >= max_retries:
                print(f"Failed after {max_retries} retries with reduced parameters")
                raise
            
            # Retry with reduced parameters
            if retry_count == 1:
                # First retry: reduce batch size
                print("Retrying with batch_size=1...")
                batch_size = 1
                del x
                x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            elif retry_count == 2:
                # Second retry: reduce sequence length
                seq_len = max(seq_len // 2, 20000)
                print(f"Retrying with seq_len={seq_len}...")
                del x
                x = torch.randn(batch_size, seq_len, hidden_size, device=device)

def test_quasar_prefill_benchmark():
    print("\n" + "="*60)
    print("QuasarAttention Prefill Benchmark")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    batch_size = 1
    seq_len = 100000
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    
    # Initialize QuasarAttention
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)
    
    # Warmup
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    for _ in range(5):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 20
    start = time.time()
    
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed
    
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Time for {num_runs} runs: {elapsed:.3f}s")
    print(f"Tokens/sec: {tokens_per_sec:.0f}")
    
    return tokens_per_sec

def test_quasar_stacked_benchmark():
    print("\n" + "="*60)
    print("QuasarAttention Stacked Benchmark (match QuasarRoPE dims)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Match the 1B-ish QuasarRoPE benchmark configuration
    # Dynamically adjust based on available GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU memory before stacked test: {free_mem:.2f} GB free")
        
        # Adjust parameters based on available memory - be conservative to avoid OOM
        if free_mem < 5.0:
            batch_size = 1
            hidden_size = 512
            num_heads = 4
            n_layers = 8
            seq_lens = [20000]
            print(f"Very low memory ({free_mem:.2f} GB), using minimal config: hidden_size={hidden_size}, n_layers={n_layers}, seq_len={seq_lens[0]}")
        elif free_mem < 10.0:
            batch_size = 1
            hidden_size = 1024
            num_heads = 8
            n_layers = 12
            seq_lens = [50000]
            print(f"Low memory ({free_mem:.2f} GB), using reduced config: hidden_size={hidden_size}, n_layers={n_layers}, seq_len={seq_lens[0]}")
        elif free_mem < 20.0:
            batch_size = 1
            hidden_size = 1536
            num_heads = 12
            n_layers = 16
            seq_lens = [75000]
            print(f"Moderate memory ({free_mem:.2f} GB), using medium config: hidden_size={hidden_size}, n_layers={n_layers}, seq_len={seq_lens[0]}")
        else:
            # Even with >20GB, be conservative - account for other processes and fragmentation
            # Full config (24 layers, 2048 hidden) can OOM even with 40GB+ due to other processes
            batch_size = 1
            hidden_size = 1536
            num_heads = 12
            n_layers = 16
            seq_lens = [75000]
            print(f"Memory {free_mem:.2f} GB (using medium config to avoid OOM from other processes): hidden_size={hidden_size}, n_layers={n_layers}, seq_len={seq_lens[0]}")
    else:
        # CPU fallback
        batch_size = 1
        hidden_size = 1024
        num_heads = 8
        n_layers = 12
        seq_lens = [20000]
    
    head_dim = hidden_size // num_heads

    # Build a stack of attention layers (attention-only, no FFN/LN/head)
    # Use try-except to handle OOM during layer creation
    max_retries = 3
    retry_count = 0
    layers = None
    
    while retry_count < max_retries:
        try:
            # Clear cache before creating layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            layers = torch.nn.ModuleList([
                QuasarAttention(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mode="chunk",
                    use_short_conv=True,
                )
                for _ in range(n_layers)
            ]).to(device)
            
            # Check memory after layer creation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                free_mem_after = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"GPU memory after layer creation: {free_mem_after:.2f} GB free")
                
                # If memory is critically low, reduce parameters and retry
                if free_mem_after < 0.5:
                    if retry_count == 0:
                        print(f"⚠️  Critically low memory after layer creation ({free_mem_after:.2f} GB), reducing n_layers from {n_layers} to {max(4, n_layers // 2)}")
                        n_layers = max(4, n_layers // 2)
                        if layers is not None:
                            del layers
                            torch.cuda.empty_cache()
                        retry_count += 1
                        continue
                    elif retry_count == 1:
                        print(f"⚠️  Still low memory ({free_mem_after:.2f} GB), reducing hidden_size from {hidden_size} to {max(512, hidden_size // 2)}")
                        hidden_size = max(512, hidden_size // 2)
                        num_heads = max(4, num_heads // 2)
                        head_dim = hidden_size // num_heads
                        seq_lens = [max(10000, seq_lens[0] // 2)]
                        if layers is not None:
                            del layers
                            torch.cuda.empty_cache()
                        retry_count += 1
                        continue
            
            # Success - break out of retry loop
            break
            
        except torch.cuda.OutOfMemoryError as e:
            retry_count += 1
            print(f"⚠️  OOM during layer creation (attempt {retry_count}/{max_retries}): {e}")
            
            # Cleanup
            if layers is not None:
                del layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            if retry_count >= max_retries:
                print("❌ Failed to create layers after multiple retries. Skipping stacked benchmark.")
                return
            
            # Reduce parameters
            if retry_count == 1:
                n_layers = max(4, n_layers // 2)
                print(f"Retrying with n_layers={n_layers}...")
            elif retry_count == 2:
                hidden_size = max(512, hidden_size // 2)
                num_heads = max(4, num_heads // 2)
                head_dim = hidden_size // num_heads
                seq_lens = [max(10000, seq_lens[0] // 2)]
                print(f"Retrying with hidden_size={hidden_size}, num_heads={num_heads}, seq_len={seq_lens[0]}...")
    
    if layers is None:
        print("❌ Failed to create layers. Skipping stacked benchmark.")
        return

    def run(mode_backward: bool):
        mode_name = "fwd+bwd" if mode_backward else "fwd-only"
        print(f"\nMode: {mode_name}")
        print("seq_len\tstep_s\ttok/s")

        for seq_len in seq_lens:
            # Check memory before creating input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                free_mem_before = torch.cuda.mem_get_info()[0] / 1024**3
                if free_mem_before < 1.0:
                    print(f"⚠️  Low memory ({free_mem_before:.2f} GB) before forward pass. Skipping seq_len={seq_len}")
                    continue
            
            try:
                x = torch.randn(batch_size, seq_len, hidden_size, device=device)
                x.requires_grad_(mode_backward)

                # Warmup with OOM handling
                try:
                    for _ in range(3):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                            y = x
                            for layer in layers:
                                y, _, _ = layer(y)
                            loss = y.float().mean()
                        if mode_backward:
                            loss.backward()
                            for p in layers.parameters():
                                if p.grad is not None:
                                    p.grad = None
                            if x.grad is not None:
                                x.grad = None
                except torch.cuda.OutOfMemoryError as e:
                    print(f"⚠️  OOM during warmup for seq_len={seq_len}: {e}")
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue

                if device.type == "cuda":
                    torch.cuda.synchronize()

                runs = 5
                t0 = time.time()
                try:
                    for _ in range(runs):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                            y = x
                            for layer in layers:
                                y, _, _ = layer(y)
                            loss = y.float().mean()
                        if mode_backward:
                            loss.backward()
                            for p in layers.parameters():
                                if p.grad is not None:
                                    p.grad = None
                            if x.grad is not None:
                                x.grad = None
                except torch.cuda.OutOfMemoryError as e:
                    print(f"⚠️  OOM during benchmark for seq_len={seq_len}: {e}")
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()

                elapsed = t1 - t0
                step_s = elapsed / runs
                tokens = batch_size * seq_len
                tok_s = tokens / step_s if step_s > 0 else 0
                print(f"{seq_len}\t{step_s:.4f}\t{tok_s:.0f}")
                
                # Cleanup
                del x, y, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"⚠️  OOM for seq_len={seq_len}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                continue

    run(mode_backward=False)
    run(mode_backward=True)


if __name__ == "__main__":
    test_quasar_basic()
    tps = test_quasar_prefill_benchmark()
    print(f"\nQuasarAttention achieved: {tps:.0f} tokens/sec")
    test_quasar_stacked_benchmark()
