"""Longcode Dataset Loader for QUASAR-SUBNET

Loads longcode benchmark format with code injection:
{
  "id": 0,
  "prompt": "...",
  "template_code": "def solve(): ...",
  "test_cases": [
    {"input_code": "...", "expected_output": [...]}
  ],
  "timeout": 30
}
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCase:
    """Single test case for code execution"""
    input_code: str
    expected_output: Any


@dataclass
class LongcodeSample:
    """Single sample from longcode benchmark with code injection"""
    sample_id: int
    prompt: str
    template_code: str
    test_cases: List[TestCase]
    context_length: str
    timeout: int = 30
    
    def get_prompt_text(self) -> str:
        """Get the prompt text for the model"""
        return self.prompt
    
    def get_template_code(self) -> str:
        """Get the template code that miner needs to complete"""
        return self.template_code


class LongcodeDataset:
    """Longcode benchmark dataset loader with test cases"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to longcode.json file. If None, uses default location.
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "data" / "longcode.json"
        else:
            dataset_path = Path(dataset_path)
        
        self.dataset_path = dataset_path
        self.samples: List[LongcodeSample] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from JSON file"""
        if not self.dataset_path.exists():
            print(f"⚠️  Longcode dataset not found at {self.dataset_path}")
            print(f"   Creating sample dataset for testing...")
            self._create_sample_dataset()
            return
        
        print(f"📂 Loading longcode dataset from {self.dataset_path}...")
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single object and array formats
            if isinstance(data, dict):
                data = [data]
            
            for i, item in enumerate(data):
                sample = self._parse_sample(item, i)
                if sample:
                    self.samples.append(sample)
            
            print(f"✅ Loaded {len(self.samples)} samples from longcode dataset")
            for sample in self.samples:
                print(f"   - Sample {sample.sample_id}: {len(sample.test_cases)} test cases, context {sample.context_length}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self._create_sample_dataset()
    
    def _parse_sample(self, data: Dict[str, Any], sample_id: int) -> Optional[LongcodeSample]:
        """Parse a single sample from JSON data"""
        try:
            prompt = data.get("prompt", "")
            template_code = data.get("template_code", "")
            test_cases_data = data.get("test_cases", [])
            context_length = data.get("context_length", "0~8K")
            timeout = data.get("timeout", 30)
            
            if not prompt or not template_code or not test_cases_data:
                return None
            
            # Parse test cases
            test_cases = []
            for tc_data in test_cases_data:
                tc = TestCase(
                    input_code=tc_data.get("input_code", ""),
                    expected_output=tc_data.get("expected_output")
                )
                test_cases.append(tc)
            
            return LongcodeSample(
                sample_id=sample_id,
                prompt=prompt,
                template_code=template_code,
                test_cases=test_cases,
                context_length=context_length,
                timeout=timeout
            )
        except Exception as e:
            print(f"⚠️  Error parsing sample {sample_id}: {e}")
            return None
    
    def _create_sample_dataset(self):
        """Create a sample dataset for testing when real dataset is not available"""
        print("📝 Creating sample longcode dataset...")
        
        sample_data = {
            "id": 0,
            "context_length": "0~8K",
            "prompt": "Write a function that adds two numbers.",
            "template_code": "def add(a, b):\n    # Your code here\n    pass",
            "test_cases": [
                {
                    "input_code": "",
                    "expected_output": 3
                }
            ],
            "timeout": 30
        }
        
        sample = self._parse_sample(sample_data, 0)
        if sample:
            self.samples.append(sample)
        
        print(f"✅ Created {len(self.samples)} sample dataset entries")
    
    def sample(self, n: int = 1, seed: Optional[int] = None) -> List[LongcodeSample]:
        """
        Sample n random samples from dataset
        
        Args:
            n: Number of samples to return
            seed: Random seed for reproducibility
        
        Returns:
            List of LongcodeSample objects
        """
        if seed is not None:
            random.seed(seed)
        
        if n >= len(self.samples):
            return self.samples.copy()
        
        return random.sample(self.samples, n)
    
    def get_sample_by_id(self, sample_id: int) -> Optional[LongcodeSample]:
        """Get a specific sample by ID"""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.samples)
