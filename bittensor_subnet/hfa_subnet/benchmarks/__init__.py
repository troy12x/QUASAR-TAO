# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
HFA Subnet Real-World Benchmark System

This module provides real-world benchmark integration for evaluating infinite context
language modeling capabilities using established datasets:

- LongBench: Comprehensive long-context evaluation suite
- HotpotQA (distractor): Multi-hop reasoning with distractors
- GovReport: Government report summarization tasks
- Needle-in-Haystack: Information retrieval in long contexts

The benchmark system integrates with the existing HFA subnet infrastructure
while providing standardized evaluation across multiple real-world tasks.
"""

from .benchmark_loader import BenchmarkLoader
from .benchmark_task import BenchmarkTask
from .longbench_loader import LongBenchLoader
from .hotpotqa_loader import HotpotQALoader
from .govreport_loader import GovReportLoader
from .needle_haystack_loader import NeedleHaystackLoader

__all__ = [
    'BenchmarkLoader',
    'BenchmarkTask', 
    'LongBenchLoader',
    'HotpotQALoader',
    'GovReportLoader',
    'NeedleHaystackLoader'
]