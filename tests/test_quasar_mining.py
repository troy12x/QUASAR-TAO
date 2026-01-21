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
    
    # Model parameters
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
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
        output, _, _ = quasar(x)
    
    print(f"Output shape: {output.shape}")
    print("QuasarAttention basic test PASSED!")
    return True

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
    batch_size = 1
    hidden_size = 2048
    num_heads = 16
    head_dim = hidden_size // num_heads
    n_layers = 24
    seq_lens = [100000]

    # Build a stack of attention layers (attention-only, no FFN/LN/head)
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

    def run(mode_backward: bool):
        mode_name = "fwd+bwd" if mode_backward else "fwd-only"
        print(f"\nMode: {mode_name}")
        print("seq_len\tstep_s\ttok/s")

        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            x.requires_grad_(mode_backward)

            # Warmup
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

            if device.type == "cuda":
                torch.cuda.synchronize()

            runs = 5
            t0 = time.time()
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

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            elapsed = t1 - t0
            step_s = elapsed / runs
            tokens = batch_size * seq_len
            tok_s = tokens / step_s if step_s > 0 else 0
            print(f"{seq_len}\t{step_s:.4f}\t{tok_s:.0f}")

    run(mode_backward=False)
    run(mode_backward=True)


if __name__ == "__main__":
    test_quasar_basic()
    tps = test_quasar_prefill_benchmark()
    print(f"\nQuasarAttention achieved: {tps:.0f} tokens/sec")
    test_quasar_stacked_benchmark()
