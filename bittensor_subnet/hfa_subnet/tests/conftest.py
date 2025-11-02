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

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add the subnet directory to Python path
subnet_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if subnet_dir not in sys.path:
    sys.path.insert(0, subnet_dir)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_bittensor():
    """Mock bittensor components for testing"""
    with patch('bittensor.logging') as mock_logging:
        mock_logging.info = Mock()
        mock_logging.warning = Mock()
        mock_logging.error = Mock()
        mock_logging.debug = Mock()
        
        yield mock_logging


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        "netuid": 999,
        "subtensor": {
            "network": "test",
            "chain_endpoint": "ws://127.0.0.1:9944"
        },
        "wallet": {
            "name": "test_wallet",
            "hotkey": "test_hotkey"
        },
        "model_config": {
            "architecture": "hfa",
            "model_name": "test_model",
            "max_context_length": 4096
        }
    }


@pytest.fixture
def mock_metagraph():
    """Mock metagraph for testing"""
    metagraph = Mock()
    metagraph.hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
    metagraph.validator_permit = [False, True, False]  # miner, validator, miner
    metagraph.axons = [Mock() for _ in range(3)]
    return metagraph


@pytest.fixture
def mock_subtensor():
    """Mock subtensor for testing"""
    subtensor = Mock()
    subtensor.network = "test"
    return subtensor


@pytest.fixture
def mock_wallet():
    """Mock wallet for testing"""
    wallet = Mock()
    wallet.hotkey.ss58_address = "test_hotkey_address"
    return wallet