"""Docmath Dataset Loader for QUASAR-SUBNET

Loads Tongyi-Zhiwen/docmath dataset format:
{
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {"style": "rule", "ground_truth": "Therefore, the answer is X.XX."}
}
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DocmathSample:
    """Single sample from docmath dataset"""
    sample_id: int
    prompt: List[Dict[str, str]]
    ground_truth: str
    expected_answer: str
    
    def get_prompt_text(self) -> str:
        """Get the prompt text for the model"""
        if self.prompt and len(self.prompt) > 0:
            return self.prompt[0].get("content", "")
        return ""
    
    def get_expected_answer(self) -> str:
        """Extract the expected answer from ground_truth"""
        # Format: "Therefore, the answer is X.XX."
        import re
        match = re.search(r"Therefore, the answer is\s+(.+?)(?:\.|$)", self.ground_truth)
        if match:
            return match.group(1).strip()
        return self.ground_truth


class DocmathDataset:
    """Docmath dataset loader"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to docmath.jsonl file. If None, uses default location.
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "data" / "docmath.jsonl"
        else:
            dataset_path = Path(dataset_path)
        
        self.dataset_path = dataset_path
        self.samples: List[DocmathSample] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from JSONL file"""
        if not self.dataset_path.exists():
            print(f"⚠️  Docmath dataset not found at {self.dataset_path}")
            print(f"   Creating sample dataset for testing...")
            self._create_sample_dataset()
            return
        
        print(f"📂 Loading docmath dataset from {self.dataset_path}...")
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        sample = self._parse_sample(data, line_num)
                        if sample:
                            self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse line {line_num}: {e}")
                        continue
            
            print(f"✅ Loaded {len(self.samples)} samples from docmath dataset")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self._create_sample_dataset()
    
    def _parse_sample(self, data: Dict[str, Any], sample_id: int) -> Optional[DocmathSample]:
        """Parse a single sample from JSON data"""
        try:
            prompt = data.get("prompt", [])
            reward_model = data.get("reward_model", {})
            
            if not prompt or not reward_model:
                return None
            
            ground_truth = reward_model.get("ground_truth", "")
            if not ground_truth:
                return None
            
            return DocmathSample(
                sample_id=sample_id,
                prompt=prompt,
                ground_truth=ground_truth,
                expected_answer=reward_model.get("expected_answer", "")
            )
        except Exception as e:
            print(f"⚠️  Error parsing sample {sample_id}: {e}")
            return None
    
    def _create_sample_dataset(self):
        """Create a sample dataset for testing when real dataset is not available"""
        print("📝 Creating sample docmath dataset...")
        
        sample_data = [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "\nPlease read the following text and answer the question below.\n\n<text>\nItem 1. Financial Statements.\n\nCONDENSED BALANCE SHEETS\n| | 2024 | 2023 |\n|---|---|---|\n| Cash | $100,000 | $80,000 |\n| Revenue | $200,000 | $180,000 |\n</text>\n\nWhat is the cash amount in 2024?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "Therefore, the answer is $100,000."
                }
            },
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "\nPlease read the following text and answer the question below.\n\n<text>\nItem 1. Financial Statements.\n\nCONDENSED STATEMENTS OF OPERATIONS\n| | 2024 |\n|---|---|\n| Revenue | $200,000 |\n| Expenses | $150,000 |\n| Net Income | $50,000 |\n</text>\n\nWhat is the net income?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "Therefore, the answer is $50,000."
                }
            },
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "\nPlease read the following text and answer the question below.\n\n<text>\nItem 1. Financial Statements.\n\nCONDENSED BALANCE SHEETS\n| | 2024 |\n|---|---|\n| Assets | $500,000 |\n| Liabilities | $300,000 |\n| Equity | $200,000 |\n</text>\n\nWhat is the total equity?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "Therefore, the answer is $200,000."
                }
            },
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "\nPlease read the following text and answer the question below.\n\n<text>\nItem 1. Financial Statements.\n\nCONDENSED STATEMENTS OF CASH FLOWS\n| | 2024 |\n|---|---|\n| Operating Cash Flow | $75,000 |\n| Investing Cash Flow | -$25,000 |\n| Financing Cash Flow | -$10,000 |\n| Net Change | $40,000 |\n</text>\n\nWhat is the net change in cash?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "Therefore, the answer is $40,000."
                }
            },
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "\nPlease read the following text and answer the question below.\n\n<text>\nItem 1. Financial Statements.\n\nCONDENSED STATEMENTS OF OPERATIONS\n| | 2024 |\n|---|---|\n| Gross Profit | $120,000 |\n| Operating Expenses | $80,000 |\n| Operating Income | $40,000 |\n</text>\n\nWhat is the operating income?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
                    }
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "Therefore, the answer is $40,000."
                }
            }
        ]
        
        for i, data in enumerate(sample_data):
            sample = self._parse_sample(data, i)
            if sample:
                self.samples.append(sample)
        
        print(f"✅ Created {len(self.samples)} sample dataset entries")
    
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[DocmathSample]:
        """
        Sample n random samples from dataset
        
        Args:
            n: Number of samples to return
            seed: Random seed for reproducibility
        
        Returns:
            List of DocmathSample objects
        """
        if seed is not None:
            random.seed(seed)
        
        if n >= len(self.samples):
            return self.samples.copy()
        
        return random.sample(self.samples, n)
    
    def get_sample_by_id(self, sample_id: int) -> Optional[DocmathSample]:
        """Get a specific sample by ID"""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.samples)
