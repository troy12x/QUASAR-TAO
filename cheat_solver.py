#!/usr/bin/env python3
"""
Cheat solver for V5 execution tasks - attempts to solve without LLM.
Tests if the task is vulnerable to script-based brute-force attacks.
"""

import re
import sys
import os

# Add QUASAR-SUBNET to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'QUASAR-SUBNET'))

from quasar.benchmarks.benchmark_loader import ContextualNeedleLoader

def solve_task_v5(task):
    """
    Attempt to solve V5 task programmatically.
    
    Returns: (answer, method_used, confidence)
    """
    context = task.context
    prompt = task.prompt
    
    print("=" * 80)
    print("🔍 CHEAT SOLVER - Attempting to solve without LLM")
    print("=" * 80)
    
    # Step 1: Extract input x from prompt
    x_match = re.search(r'input x=(\d+)', prompt)
    if not x_match:
        return None, "Failed to extract input x", 0
    x_input = int(x_match.group(1))
    print(f"\n✓ Extracted input: x = {x_input}")
    
    # Step 2: Extract system name from prompt
    system_match = re.search(r'Calculate the output for ([\w\s]+) with input', prompt)
    if not system_match:
        return None, "Failed to extract system name", 0
    system_name = system_match.group(1).strip()
    print(f"✓ Target system: {system_name}")
    
    # Step 3: Parse System Function Mapping table
    print(f"\n🔍 Step 1: Finding function for {system_name}...")
    
    # Look for the mapping table pattern
    mapping_table_match = re.search(r'# System Function Mapping\n(.*?)(?=\n#|\nLOG:|\nStatus report:|\n$)', context, re.DOTALL)
    if not mapping_table_match:
        return None, "Failed to find System Function Mapping table", 0
    
    mapping_lines = mapping_table_match.group(1).strip().split('\n')
    sys_func_map = {}
    for line in mapping_lines:
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                sys_name = parts[0].strip()
                func_name = parts[1].strip()
                sys_func_map[sys_name] = func_name
    
    if system_name not in sys_func_map:
        print(f"❌ System '{system_name}' not found in mapping table!")
        print(f"Available systems: {list(sys_func_map.keys())}")
        return None, "System not in mapping table", 0
    
    target_function = sys_func_map[system_name]
    print(f"✓ Found function: {target_function}")
    
    # Step 4: Find the system's operational state (the clue)
    print(f"\n🔍 Step 2: Finding operational state of {system_name}...")
    
    # Look for status report about the target system
    state_match = re.search(rf'(?:Status report|Engineering report):\s*{re.escape(system_name)}\s+is\s+([^.]+)\.', context)
    if not state_match:
        # Try alternative pattern
        state_match = re.search(rf'{re.escape(system_name)}\s+is\s+([^.]+)\.', context)
    
    if not state_match:
        print(f"❌ Could not find state description for {system_name}")
        return None, "State description not found", 0
    
    state_description = state_match.group(1).strip()
    print(f"✓ Found state: '{state_description}'")
    
    # Step 5: Parse Parameter A (Multiplier) Configuration table
    print(f"\n🔍 Step 3: Looking up Parameter A (multiplier)...")
    
    param_a_table_match = re.search(r'# Parameter A \(Multiplier\) Configuration\n(.*?)(?=\n#|\nLOG:|\nStatus report:|\n$)', context, re.DOTALL)
    if not param_a_table_match:
        return None, "Failed to find Parameter A table", 0
    
    param_a_map = {}
    param_a_lines = param_a_table_match.group(1).strip().split('\n')
    for line in param_a_lines:
        # Match pattern: "description" -> value
        match = re.match(r'"([^"]+)"\s*->\s*(\d+)', line)
        if match:
            desc = match.group(1)
            val = int(match.group(2))
            param_a_map[desc] = val
    
    if state_description not in param_a_map:
        print(f"❌ State '{state_description}' not in Parameter A table!")
        print(f"Available states: {list(param_a_map.keys())[:5]}...")
        return None, "State not in Parameter A table", 0
    
    multiplier = param_a_map[state_description]
    print(f"✓ Multiplier = {multiplier}")
    
    # Step 6: Parse Parameter B (Offset) Configuration table
    print(f"\n🔍 Step 4: Looking up Parameter B (offset)...")
    
    param_b_table_match = re.search(r'# Parameter B \(Offset\) Configuration\n(.*?)(?=\n#|\nLOG:|\nStatus report:|\n$)', context, re.DOTALL)
    if not param_b_table_match:
        return None, "Failed to find Parameter B table", 0
    
    param_b_map = {}
    param_b_lines = param_b_table_match.group(1).strip().split('\n')
    for line in param_b_lines:
        match = re.match(r'"([^"]+)"\s*->\s*(\d+)', line)
        if match:
            desc = match.group(1)
            val = int(match.group(2))
            param_b_map[desc] = val
    
    if state_description not in param_b_map:
        print(f"❌ State '{state_description}' not in Parameter B table!")
        return None, "State not in Parameter B table", 0
    
    offset = param_b_map[state_description]
    print(f"✓ Offset = {offset}")
    
    # Step 7: Calculate result
    print(f"\n🔍 Step 5: Calculating result...")
    result = (x_input * multiplier) + offset
    print(f"✓ Calculation: ({x_input} × {multiplier}) + {offset} = {result}")
    
    print("\n" + "=" * 80)
    print(f"🎯 CHEAT SOLVER RESULT: {result}")
    print("=" * 80)
    
    return str(result), "Full multi-step parsing", 1.0


def brute_force_attempt(task):
    """
    Attempt brute-force by trying all possible parameter combinations.
    This should fail if V5 design is correct.
    """
    print("\n" + "=" * 80)
    print("💀 BRUTE FORCE ATTEMPT - Trying all combinations")
    print("=" * 80)
    
    context = task.context
    prompt = task.prompt
    
    # Extract input x
    x_match = re.search(r'input x=(\d+)', prompt)
    if not x_match:
        return None, "Brute force: Failed to extract x", 0
    x_input = int(x_match.group(1))
    
    # Extract all possible multipliers and offsets from tables
    param_a_values = []
    param_b_values = []
    
    # Extract all numbers from Parameter A table
    param_a_match = re.search(r'# Parameter A \(Multiplier\) Configuration\n(.*?)(?=\n#|\nLOG:|\nStatus report:|\n$)', context, re.DOTALL)
    if param_a_match:
        numbers = re.findall(r'->\s*(\d+)', param_a_match.group(1))
        param_a_values = [int(n) for n in numbers]
    
    # Extract all numbers from Parameter B table
    param_b_match = re.search(r'# Parameter B \(Offset\) Configuration\n(.*?)(?=\n#|\nLOG:|\nStatus report:|\n$)', context, re.DOTALL)
    if param_b_match:
        numbers = re.findall(r'->\s*(\d+)', param_b_match.group(1))
        param_b_values = [int(n) for n in numbers]
    
    combinations = len(param_a_values) * len(param_b_values)
    print(f"Found {len(param_a_values)} multipliers, {len(param_b_values)} offsets")
    print(f"Total combinations to try: {combinations}")
    
    if combinations > 1000:
        print("❌ Too many combinations for brute force (> 1000)")
        return None, f"Brute force blocked: {combinations} combos", 0
    
    print(f"Trying all {combinations} combinations...")
    
    # Try all combinations
    for mult in param_a_values:
        for off in param_b_values:
            result = (x_input * mult) + off
            # We can't verify without knowing the correct answer
            # But we could generate a list of candidates
    
    print(f"⚠️ Brute force generated {combinations} candidates (cannot verify without ground truth)")
    return None, f"Brute force: {combinations} candidates", 0.1


def main():
    """Generate a task and try to cheat-solve it."""
    
    print("🎯 Generating V5 execution task...")
    loader = ContextualNeedleLoader()
    task = loader.generate_task(n_distractors=4, target_len=32000)
    
    print(f"\nTask ID: {task.task_id}")
    print(f"Dataset: {task.dataset_name}")
    print(f"Expected Answer: {task.expected_output}")
    
    # Try to solve
    answer, method, confidence = solve_task_v5(task)
    
    if answer:
        if answer == task.expected_output:
            print(f"\n✅ CHEAT SUCCESSFUL! Solved correctly: {answer}")
            print(f"⚠️  WARNING: Task is vulnerable to script-based attacks!")
            print(f"   Method: {method}")
            print(f"   Confidence: {confidence}")
        else:
            print(f"\n❌ CHEAT FAILED! Got {answer}, expected {task.expected_output}")
            print(f"   Method: {method}")
    else:
        print(f"\n❌ CHEAT FAILED: {method}")
        print(f"   Confidence: {confidence}")
    
    # Try brute force
    brute_force_attempt(task)
    
    # Save task for inspection
    with open("cheat_task.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TASK DETAILS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Task ID: {task.task_id}\n")
        f.write(f"Dataset: {task.dataset_name}\n")
        f.write(f"Expected Answer: {task.expected_output}\n\n")
        f.write("=" * 80 + "\n")
        f.write("FULL CONTEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(task.context)
    print(f"\n📄 Task saved to: cheat_task.txt")


if __name__ == "__main__":
    main()
