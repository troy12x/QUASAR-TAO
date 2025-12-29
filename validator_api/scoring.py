import math
import re
import sys
import os
from typing import Tuple, Union, Optional

# Add parent directory to path to import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from quasar.benchmarks.metrics import dataset2metric

# Context Buckets (Ported from validator.py)
BUCKETS = {
    '32k': (0, 32_000),      # < 32k
    '128k': (32_000, 128_000),
    '500k': (128_000, 500_000),
    '1M': (500_000, 1_000_000),
    '2M': (1_000_000, 2_000_000),
}

# Non-linear Reward Multipliers (Ported from validator.py)
REWARD_MULTIPLIERS = {
    '32k': 0.1,    # Heavy penalty for short context
    '128k': 0.1,   
    '500k': 1.0,   # Baseline for "Long Context"
    '1M': 1.5,     # Reward scaling
    '2M': 2.0      # Maximum reward
}

def extract_answer(text: str) -> Tuple[Union[float, None], str]:
    """Extract numeric answer from text, handling \boxed{} and CoT."""
    try:
        text = text.strip()
        # 1. Try \boxed{val}
        boxed_match = re.search(r"\\boxed\{([0-9\.]+)\}", text)
        if boxed_match:
            return float(boxed_match.group(1)), "boxed"
        
        # 2. Try finding the LAST number in the text
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            return float(nums[-1]), f"last_num({nums[-1]})"
            
        return None, "none"
    except Exception as e:
        return None, f"error({str(e)[:50]})"

def calculate_score(
    response_text: str, 
    expected_output: str, 
    dataset_name: str, 
    context_length: int,
    all_classes: Optional[list] = None
) -> float:
    """
    Calculates the score based on production metrics and context length multipliers.
    """
    if not response_text:
        return 0.0

    # 1. Determine base score using dataset-specific metrics
    score = 0.0
    try:
        if dataset_name in ["quasar_execution_v1", "quasar_execution_v3"]:
            # Execution Scoring: Exact numeric match (with small tolerance)
            miner_val, _ = extract_answer(response_text)
            target_val = float(expected_output)
            
            if miner_val is not None and math.isclose(miner_val, target_val, rel_tol=1e-3, abs_tol=0.01):
                score = 1.0
            else:
                score = 0.0
        else:
            # Standard metrics from quasar
            metric_fn = dataset2metric.get(dataset_name, dataset2metric.get('narrativeqa'))
            if all_classes:
                score = metric_fn(response_text, expected_output, all_classes=all_classes)
            else:
                score = metric_fn(response_text, expected_output)
    except Exception as e:
        print(f"Scoring error: {e}")
        score = 0.0

    # 2. Apply Multiplier based on context length
    bucket = "infinity"
    for name, (low, high) in BUCKETS.items():
        if low <= context_length < high:
            bucket = name
            break
    
    multiplier = REWARD_MULTIPLIERS.get(bucket, 1.0)
    return float(score * multiplier)
