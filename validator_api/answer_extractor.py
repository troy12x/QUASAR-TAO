"""Answer Extraction Logic for QUASAR-SUBNET

Extracts answers from model outputs in the format:
"Therefore, the answer is X.XX"
"""

import re
from typing import Optional, List


def extract_answer(model_output: str) -> Optional[str]:
    """
    Extract answer from model output.
    Expected format: "Therefore, the answer is X.XX"
    
    Args:
        model_output: The raw output from the model
    
    Returns:
        Extracted answer string, or None if not found
    """
    if not model_output:
        return None
    
    # Pattern 1: "Therefore, the answer is X.XX" (most common)
    match = re.search(r"Therefore, the answer is\s+(.+?)(?:\.|$)", model_output)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "answer is X.XX"
    match = re.search(r"answer is\s+(.+?)(?:\.|$)", model_output)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: "The answer is X.XX"
    match = re.search(r"The answer is\s+(.+?)(?:\.|$)", model_output)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: Last number in output (fallback)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", model_output)
    if match:
        return match.group(0)
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    Removes extra whitespace, currency symbols, etc.
    
    Args:
        answer: The answer string to normalize
    
    Returns:
        Normalized answer string
    """
    if not answer:
        return ""
    
    # Remove currency symbols
    answer = re.sub(r'[$€£¥]', '', answer)
    
    # Remove commas from numbers
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
    
    # Strip whitespace
    answer = answer.strip()
    
    # Remove trailing period
    if answer.endswith('.'):
        answer = answer[:-1]
    
    return answer


def compare_answers(predicted: str, ground_truth: str, tolerance: float = 0.0) -> bool:
    """
    Compare predicted answer with ground truth.
    
    Args:
        predicted: The predicted answer string
        ground_truth: The ground truth answer string
        tolerance: Numerical tolerance for comparison
    
    Returns:
        True if answers match (within tolerance for numbers)
    """
    if not predicted or not ground_truth:
        return False
    
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)
    
    # Try exact match first
    if pred_norm == truth_norm:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_norm)
        truth_num = float(truth_norm)
        
        if tolerance > 0:
            return abs(pred_num - truth_num) <= tolerance
        else:
            return abs(pred_num - truth_num) < 1e-6  # Very small tolerance for floating point
    except (ValueError, TypeError):
        # Not numbers, do string comparison
        return pred_norm.lower() == truth_norm.lower()


def extract_and_compare(model_output: str, ground_truth: str, tolerance: float = 0.0) -> tuple[bool, Optional[str]]:
    """
    Extract answer from model output and compare with ground truth.
    
    Args:
        model_output: The raw output from the model
        ground_truth: The ground truth answer string
        tolerance: Numerical tolerance for comparison
    
    Returns:
        Tuple of (is_correct, extracted_answer)
    """
    extracted = extract_answer(model_output)
    if not extracted:
        return False, None
    
    is_correct = compare_answers(extracted, ground_truth, tolerance)
    return is_correct, extracted


def batch_extract_and_compare(model_outputs: List[str], ground_truths: List[str], tolerance: float = 0.0) -> List[dict]:
    """
    Batch process multiple model outputs.
    
    Args:
        model_outputs: List of model outputs
        ground_truths: List of ground truth answers
        tolerance: Numerical tolerance for comparison
    
    Returns:
        List of result dictionaries with keys: correct, predicted, ground_truth
    """
    if len(model_outputs) != len(ground_truths):
        raise ValueError("model_outputs and ground_truths must have same length")
    
    results = []
    for i, (output, truth) in enumerate(zip(model_outputs, ground_truths)):
        is_correct, predicted = extract_and_compare(output, truth, tolerance)
        results.append({
            "sample_id": i,
            "correct": is_correct,
            "predicted": predicted,
            "ground_truth": truth
        })
    
    return results
