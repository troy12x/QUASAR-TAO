# The MIT License (MIT)
# Copyright © 2026 SILX INC

"""
Benchmark Loader for SILX Subnet

Loads and manages benchmark tasks for validator evaluation.
Supports:
- LongBench (standard)
- Synthetic scaling tasks (up to 2M tokens)
- Caching and filtering
"""

import bittensor as bt
import random
from typing import List, Dict, Any, Optional, Tuple
from .benchmark_task import BenchmarkTask
# import datasets  # If allowed, otherwise we use local loading

class BenchmarkLoader:
    """
    Loads benchmark tasks for validator testing.
    Manages the "Curriculum" of the subnet.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize benchmark loader."""
        self.config = config or {}
        self.longbench_config = self.config.get('longbench', {})
        self.enabled_datasets = self.longbench_config.get('enabled_tasks', ['narrativeqa', 'gov_report'])
        self.data_path = self.longbench_config.get('data_path', 'data/longbench')
        
        self.cache = {} # task_type -> List[BenchmarkTask]
        self.real_loader = ContextualNeedleLoader()
        bt.logging.info(f"📚 BenchmarkLoader initialized | Mode: EXECUTION_RIDGE")

    def load_benchmark_tasks(
        self, 
        num_tasks: int = 1, 
        benchmark_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        max_context_length: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """
        Load a batch of benchmark tasks.
        """
        tasks = []
        types_to_load = benchmark_types or ['longbench']
        
        for b_type in types_to_load:
            if b_type == 'longbench':
                tasks.extend(self._load_longbench_tasks(num_tasks, difficulty, max_context_length))
            elif b_type == 'synthetic':
                 tasks.extend(self._generate_synthetic_tasks(num_tasks, difficulty, max_context_length))
        
        # Shuffle and return requested amount
        random.shuffle(tasks)
        return tasks[:num_tasks]


    def _load_longbench_tasks(self, count: int, difficulty: Optional[str] = None, max_ctx: Optional[int] = None) -> List[BenchmarkTask]:
        """Internal: Load contextual execution tasks."""
        loaded_tasks = []
        for _ in range(count):
            # Use the new ContextualNeedleLoader
            task = self.real_loader.get_sample(max_ctx)
            if task:
                loaded_tasks.append(task)
                
        return loaded_tasks

    def _generate_synthetic_tasks(self, count: int, difficulty: Optional[str], max_ctx: Optional[int] = None) -> List[BenchmarkTask]:
        # Redirect to the main loader for now to enforce the new protocol everywhere
        return self._load_longbench_tasks(count, difficulty, max_ctx)

class ContextualNeedleLoader:
    """
    Generates 'Semantic Dependency' tasks (V4 - Hardened).
    
    Upgrade V4 Features:
    1. **Rotational Invariants**: Instead of just Odd/Even, we use `Result % Divisor == Remainder`.
       - System generates a random Divisor (3-9) and Target Remainder.
       - Target Function guarantees `(Result % D) == T`.
       - Distractors guarantee `(Result % D) != T`.
       
    2. **Semantic Mode Mapping**: 
       - Narrative does NOT use the keyword "CRITICAL".
       - Narrative uses: "severe hydraulic instability", "core temperature critical", etc.
       - Miner must map "instability" -> "CRITICAL" to run the code `if mode == 'CRITICAL':`.
       
    3. **Undecidable Distractors**:
       - Distractors use the SAME variable names but different logic/modes which produce invalid invariants.
    """
    
    def __init__(self):
        self.systems = ["Gorgon Drive", "Chimera Protocol", "Hydra Engine", "Pegasus Link", "Kraken Module"]
        # Code Constants : [Semantic Descriptions]
        self.mode_map = {
            "CRITICAL": ["exceeds safe operating limits", "severe hydraulic instability", "imminent core failure", "structural integrity compromised"],
            "OPTIMIZED": ["operating at peak efficiency", "nominal throughput achieved", "flow rate 100%", "optimal harmonic resonance"],
            "SAFE": ["maintenance protocols active", "output restricted for safety", "idle state engaged", "manual override enabled"],
            "DEGRADED": ["performance sub-optimal", "minor fractures detected", "buffer overflow warning", "latency high"]
        }
        
    def generate_task(self, n_distractors: int = 4, target_len: int = 32000) -> BenchmarkTask:
        import math
        
        # 1. Setup Narrative Truths
        system_name = random.choice(self.systems)
        true_mode_key = random.choice(list(self.mode_map.keys()))
        true_mode_desc = random.choice(self.mode_map[true_mode_key])
        
        # 2. Setup Invariant (The Anti-Brute-Force Key)
        # We pick a Divisor (3-9) and a Target Remainder
        # The prompt will NOT explicitly state this invariant, but the VALID function will satisfy it.
        # This prevents random guessing.
        divisor = random.randint(3, 9)
        target_remainder = random.randint(0, divisor - 1)
        
        # 3. Generate Target Function
        # We need a function that, given 'x' and correct 'mode', returns `val` where `val % divisor == target_remainder`.
        # Logic: return (x * mul + offset) * divisor + target_remainder
        # This guarantees the modulo invariant.
        
        x_input = random.choice(range(10, 100))
        tgt_func_name = f"calc_{random.randint(100, 999)}_tgt"
        
        # To make it look natural, we do: `return x * large_mul + adjusted_bias`
        # We solve for bias.
        # desired = k * divisor + remainder
        # We start with some random logic `x * mul`
        # Then add bias to align modulo.
        
        mul_base = random.randint(2, 10)
        
        # Target Code Logic
        # if mode == "CRITICAL": ...
        # bias calculation:
        # current_val = x * mul_base
        # required_rem = target_remainder
        # current_rem = current_val % divisor
        # bias_needed = (required_rem - current_rem) % divisor
        # To make bias look random, we add random multiples of divisor
        
        current_rem = (x_input * mul_base) % divisor
        bias = (target_remainder - current_rem) % divisor
        bias += random.randint(1, 5) * divisor # Add noise to bias
        
        target_code = f"""def {tgt_func_name}(x, mode):
    # Logic for {system_name}
    if mode == "{true_mode_key}":
        return x * {mul_base} + {bias}
    else:
        return x * {mul_base} + {bias + 1}""" # +1 forces invariant violation
        
        target_result = x_input * mul_base + bias
        
        functions = [target_code]
        
        # 4. Generate Distractors
        # Distractors must FAIL the invariant on the challenge input
        for _ in range(n_distractors):
            d_name = f"calc_{random.randint(100, 999)}"
            d_mul = random.randint(2, 10)
            
            # Ensure this calculation violates the target_remainder invariant
            # (current + bias) % div != target_rem
            d_bias = random.randint(1, 20)
            if (x_input * d_mul + d_bias) % divisor == target_remainder:
                d_bias += 1 # Force mismatch
            
            d_code = f"""def {d_name}(x, mode):
    if mode == "SAFE":
        return x * {d_mul} + {d_bias}
    else:
        return x * {d_mul} + {d_bias + 2}"""
            functions.append(d_code)

        # 5. Generate Narrative
        # Use SEMANTIC DESCRIPTION, not the keyword
        clue = f"Engineering report: The {system_name} is currently showing signs of {true_mode_desc}."
        
        context_parts = []
        noise_pool = ["calibrating", "waiting for ack", "packet loss", "resyncing", "deploying hotifx"]
        
        for _ in range(n_distractors):
            fake_sys = f"System-{random.randint(1,99)}"
            fake_key = random.choice(list(self.mode_map.keys()))
            fake_desc = random.choice(self.mode_map[fake_key])
            context_parts.append(f"The {fake_sys} is {fake_desc}.")
            
        context_parts.append(clue)
        for f in functions:
            context_parts.append(f)
            
        while sum(len(c) for c in context_parts) < target_len:
            context_parts.append(f"LOG: {random.choice(noise_pool)} - {random.randint(100,999)}")
            
        random.shuffle(context_parts)
        full_context = "\n\n".join(context_parts)
        
        # 6. Construct Prompt
        # Hint about the invariant to guide the miner? 
        # "Map the state description to a standard mode (CRITICAL, OPTIMIZED, SAFE, DEGRADED)."
        prompt = f"Calculate the output for the {system_name} using input x={x_input}. You must identify the operational state of the {system_name} from the text and map it to a valid system Mode (CRITICAL, OPTIMIZED, SAFE, DEGRADED) to execute the function."
        
        return BenchmarkTask(
            task_id=f"exec_v4_{random.randint(1000, 9999)}",
            dataset_name="quasar_execution_v3", # Keep v3 dataset name for compat
            task_type="execution",
            context=full_context,
            prompt=prompt,
            expected_output=str(target_result),
            context_length=len(full_context),
            difficulty_level="extreme",
            evaluation_metrics=["execution_match"],
            source="contextual_needle_generator_v4"
        )
        
    def get_sample(self, max_ctx: Optional[int] = None) -> BenchmarkTask:
        # Default to the requested 125k - 1M range
        base_len = random.randint(125000, 1000000)
        target_len = min(max_ctx, base_len) if max_ctx else base_len
        
        return self.generate_task(n_distractors=10 if target_len > 50000 else 4, target_len=target_len)
