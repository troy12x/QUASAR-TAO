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
        min_context_length: Optional[int] = None,
        max_context_length: Optional[int] = None
    ) -> List[BenchmarkTask]:
        """
        Load a batch of benchmark tasks.
        """
        tasks = []
        types_to_load = benchmark_types or ['longbench']
        
        for b_type in types_to_load:
            if b_type == 'longbench':
                tasks.extend(self._load_longbench_tasks(num_tasks, difficulty, min_context_length, max_context_length))
            elif b_type == 'synthetic':
                 tasks.extend(self._generate_synthetic_tasks(num_tasks, difficulty, min_context_length, max_context_length))
        
        # Shuffle and return requested amount
        random.shuffle(tasks)
        return tasks[:num_tasks]


    def _load_longbench_tasks(self, count: int, difficulty: Optional[str] = None, min_ctx: Optional[int] = None, max_ctx: Optional[int] = None) -> List[BenchmarkTask]:
        """Internal: Load contextual execution tasks."""
        loaded_tasks = []
        for _ in range(count):
            # Use the new ContextualNeedleLoader
            task = self.real_loader.get_sample(min_ctx, max_ctx)
            if task:
                loaded_tasks.append(task)
                
        return loaded_tasks

    def _generate_synthetic_tasks(self, count: int, difficulty: Optional[str], min_ctx: Optional[int] = None, max_ctx: Optional[int] = None) -> List[BenchmarkTask]:
        # Redirect to the main loader for now to enforce the new protocol everywhere
        return self._load_longbench_tasks(count, difficulty, min_ctx, max_ctx)

class ContextualNeedleLoader:
    """
    Generates 'Semantic Dependency' tasks (V5 - Long-Context Reasoning).
    
    V5 Design: Forces genuine long-context semantic reasoning
    
    Key Changes from V4:
    1. **No function logic visible** - Only signatures shown
    2. **Configuration tables scattered** - Must find and lookup multiple tables
    3. **Semantic parameter mapping** - Mode descriptions map to numeric parameters
    4. **Brute-force impossible** - Too many parameter combinations (100+)
    5. **Multi-step reasoning** - Requires 4+ information lookups
    
    Task Structure:
    - Context contains:
      * Function signatures (no implementation)
      * System → Function mapping table
      * Mode description → Parameter A table  
      * Mode description → Parameter B table
      * Noise logs
      * Semantic clue about system state
    
    - To solve, must:
      1. Find which function corresponds to system (table lookup)
      2. Find system's state description (semantic clue)
      3. Map state to Parameter A (table lookup)
      4. Map state to Parameter B (table lookup)
      5. Calculate: result = (x * param_A) + param_B
    """
    
    def __init__(self):
        self.systems = ["Gorgon Drive", "Chimera Protocol", "Hydra Engine", "Pegasus Link", "Kraken Module", "Titan Core", "Vortex Array", "Nexus Bridge"]
        
        # Semantic mode descriptions (10 per mode for variety)
        self.mode_descriptions = {
            "CRITICAL": [
                "exceeds safe operating limits", "severe hydraulic instability", "imminent core failure",
                "structural integrity compromised", "catastrophic pressure buildup", "thermal runaway detected",
                "containment breach imminent", "critical resonance failure", "emergency shutdown required",
                "system at failure threshold"
            ],
            "OPTIMIZED": [
                "operating at peak efficiency", "nominal throughput achieved", "flow rate 100%",
                "optimal harmonic resonance", "maximum performance mode", "efficiency at theoretical maximum",
                "power output optimized", "thermal regulation perfect", "load balancing ideal",
                "performance metrics nominal"
            ],
            "SAFE": [
                "maintenance protocols active", "output restricted for safety", "idle state engaged",
                "manual override enabled", "safety interlocks engaged", "reduced capacity mode",
                "standby operation", "conservative power settings", "safety margin maintained",
                "operation within safe parameters"
            ],
            "DEGRADED": [
                "performance sub-optimal", "minor fractures detected", "buffer overflow warning",
                "latency high", "efficiency reduced", "partial system failure",
                "degraded performance mode", "subsystem malfunction", "throughput limited",
                "operating with reduced capability"
            ]
        }
        
    def generate_task(self, n_distractors: int = 6, target_len: int = 32000) -> BenchmarkTask:
        """
        Generate a task requiring genuine long-context semantic reasoning.
        """
        
        # 1. Setup task parameters
        system_name = random.choice(self.systems)
        true_mode = random.choice(list(self.mode_descriptions.keys()))
        true_mode_desc = random.choice(self.mode_descriptions[true_mode])
        
        x_input = random.randint(10, 99)
        
        # 2. Generate function signatures (no implementation - just signatures)
        # Target function + distractors
        all_functions = []
        target_func_name = f"calc_{random.randint(100, 999)}"
        all_functions.append(target_func_name)
        
        for _ in range(n_distractors):
            all_functions.append(f"calc_{random.randint(100, 999)}")
        
        random.shuffle(all_functions)
        
        # 3. Generate parameter values for each mode description
        # Create unique parameter mappings to prevent brute-force
        param_a_map = {}  # mode_desc -> multiplier
        param_b_map = {}  # mode_desc -> offset
        
        for mode, descs in self.mode_descriptions.items():
            for desc in descs:
                # Random parameters (2-20 range for multiplier, 1-50 for offset)
                param_a_map[desc] = random.randint(2, 20)
                param_b_map[desc] = random.randint(1, 50)
        
        # Calculate expected result
        multiplier = param_a_map[true_mode_desc]
        offset = param_b_map[true_mode_desc]
        expected_result = (x_input * multiplier) + offset
        
        # 4. Build context parts
        context_parts = []
        noise_pool = ["calibrating", "waiting for ack", "packet loss", "resyncing", "deploying hotifx", "handshake complete", "buffer flush", "checksum verified"]
        
        # Add function signatures (no logic visible!)
        func_signatures = "\n".join([f"def {f}(x, multiplier, offset):" for f in all_functions])
        context_parts.append(f"# System Calculation Functions\n{func_signatures}")
        
        # Add System → Function mapping table (scattered in context)
        sys_func_map = {}
        for func in all_functions:
            # Assign random systems to functions
            sys_func_map[random.choice(self.systems)] = func
        
        # Ensure target system is in the map
        sys_func_map[system_name] = target_func_name
        
        # Format as table
        map_table = "# System Function Mapping\n"
        for sys_name, func_name in sys_func_map.items():
            map_table += f"{sys_name} -> {func_name}\n"
        context_parts.append(map_table)
        
        # Add Mode → Parameter A mapping table
        param_a_table = "# Parameter A (Multiplier) Configuration\n"
        for desc, val in param_a_map.items():
            param_a_table += f'"{desc}" -> {val}\n'
        context_parts.append(param_a_table)
        
        # Add Mode → Parameter B mapping table
        param_b_table = "# Parameter B (Offset) Configuration\n"
        for desc, val in param_b_map.items():
            param_b_table += f'"{desc}" -> {val}\n'
        context_parts.append(param_b_table)
        
        # Add distractor system states (wrong systems)
        for _ in range(n_distractors):
            fake_sys = random.choice([s for s in self.systems if s != system_name])
            fake_mode = random.choice(list(self.mode_descriptions.keys()))
            fake_desc = random.choice(self.mode_descriptions[fake_mode])
            context_parts.append(f"Status report: {fake_sys} is {fake_desc}.")
        
        # Add target system state (THE KEY INFORMATION)
        clue = f"Status report: {system_name} is {true_mode_desc}."
        context_parts.append(clue)
        
        # Add noise to reach target length
        while sum(len(c) for c in context_parts) < target_len:
            context_parts.append(f"LOG: {random.choice(noise_pool)} - {random.randint(100, 999)}")
        
        # Shuffle context (but keep tables together for readability)
        random.shuffle(context_parts)
        full_context = "\n\n".join(context_parts)
        
        # Verify clue is present
        if clue not in full_context:
            full_context = clue + "\n\n" + full_context
            bt.logging.warning("Clue lost during shuffle, prepended")
        
        # 5. Construct prompt - forces multi-step reasoning
        prompt = f"""Calculate the output for {system_name} with input x={x_input}.

To solve this task, you must:
1. Find the function assigned to {system_name} in the System Function Mapping table
2. Find the current operational state of {system_name} in the status reports
3. Look up the Parameter A (multiplier) for this state in the Parameter A Configuration table
4. Look up the Parameter B (offset) for this state in the Parameter B Configuration table
5. Execute the function with: result = (x * multiplier) + offset

Provide only the numeric result."""
        
        # 6. Log task generation
        bt.logging.info(f"Task V5: {system_name} | Mode: {true_mode} | x={x_input} | Expected: {expected_result}")
        bt.logging.info(f"Clue: {clue}")
        bt.logging.info(f"Function: {target_func_name} | Multiplier: {multiplier} | Offset: {offset}")
        
        return BenchmarkTask(
            task_id=f"exec_v5_{random.randint(1000, 9999)}",
            dataset_name="quasar_execution_v5",
            task_type="execution",
            context=full_context,
            prompt=prompt,
            expected_output=str(expected_result),
            context_length=len(full_context),
            difficulty_level="extreme",
            evaluation_metrics=["execution_match"],
            source="contextual_needle_generator_v5",
            metadata={
                "system_name": system_name,
                "true_mode": true_mode,
                "true_mode_desc": true_mode_desc,
                "target_function": target_func_name,
                "multiplier": multiplier,
                "offset": offset,
                "x_input": x_input,
                "clue": clue
            }
        )
        
    def get_sample(self, min_ctx: Optional[int] = None, max_ctx: Optional[int] = None) -> BenchmarkTask:
        # Default to a reasonable range if not specified
        low = min_ctx if min_ctx is not None else 125000
        high = max_ctx if max_ctx is not None else 1000000
        
        # Ensure low doesn't exceed high
        if low > high:
            low = high // 2
            
        target_len = random.randint(low, high)
        
        return self.generate_task(n_distractors=10 if target_len > 50000 else 4, target_len=target_len)
