# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import subprocess
import argparse
import bittensor as bt
from .logging import setup_events_logger
from .config_validator import ConfigValidator


def is_cuda_available():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT
        )
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)
    
    # Validate subnet configuration files
    validate_subnet_configs(config)


def validate_subnet_configs(config: "bt.Config"):
    """
    Validate subnet configuration files.
    
    Args:
        config: Bittensor configuration object
    """
    # Initialize subnet config attributes with empty dicts as fallback
    # Check both if attribute doesn't exist AND if it's None
    if not hasattr(config, 'subnet') or config.subnet is None:
        config.subnet = type('SubnetConfig', (), {})()
    
    config.subnet.hfa_config = {}
    config.subnet.subnet_config = {}
    
    try:
        # Get the directory containing configuration files
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(config_dir, '..', '..')  # Go up to subnet root
        config_dir = os.path.abspath(config_dir)
        
        # Validate all configuration files
        validated_configs = ConfigValidator.validate_all_configs(config_dir)
        
        # Update with validated configs if successful
        config.subnet.hfa_config = validated_configs.get('hfa', {})
        config.subnet.subnet_config = validated_configs.get('subnet', {})
        
        bt.logging.info("Subnet configuration validation completed successfully")
        
    except Exception as e:
        bt.logging.warning(f"Configuration validation failed: {e}")
        bt.logging.warning("Using empty configuration - miner will use default settings")
        # Config attributes are already initialized with empty dicts above


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=439)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )

    # Architecture-specific configuration arguments
    parser.add_argument(
        "--neuron.model_architecture",
        type=str,
        help="Model architecture to use (hfa, simplemind, hybrid, standard)",
        default="hfa",
        choices=["hfa", "simplemind", "hybrid", "standard"]
    )

    parser.add_argument(
        "--neuron.architecture_config_override",
        type=str,
        help="JSON string to override default architecture configuration",
        default=None
    )

    parser.add_argument(
        "--neuron.enable_architecture_switching",
        action="store_true",
        help="Enable dynamic architecture switching based on task requirements",
        default=False
    )

    parser.add_argument(
        "--neuron.hybrid_mixing_strategy",
        type=str,
        help="Mixing strategy for hybrid models (alternating, parallel, sequential)",
        default="alternating",
        choices=["alternating", "parallel", "sequential"]
    )

    parser.add_argument(
        "--neuron.max_context_length",
        type=int,
        help="Maximum context length to support",
        default=100000
    )

    parser.add_argument(
        "--neuron.enable_benchmark_evaluation",
        action="store_true",
        help="Enable real-world benchmark evaluation",
        default=True
    )

    parser.add_argument(
        "--neuron.benchmark_types",
        type=str,
        nargs="+",
        help="Types of benchmarks to enable",
        default=["longbench", "hotpotqa_distractor", "govreport", "needle_in_haystack"],
        choices=["longbench", "hotpotqa_distractor", "govreport", "needle_in_haystack"]
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )

    # Miner-specific architecture arguments
    parser.add_argument(
        "--miner.preferred_architecture",
        type=str,
        help="Preferred model architecture for this miner",
        default=None,
        choices=["hfa", "simplemind", "hybrid", "standard"]
    )

    parser.add_argument(
        "--miner.enable_model_switching",
        action="store_true",
        help="Enable switching between different model architectures based on task",
        default=False
    )

    parser.add_argument(
        "--miner.model_checkpoint_path",
        type=str,
        help="Path to model checkpoint file",
        default=None
    )

    parser.add_argument(
        "--miner.performance_tracking",
        action="store_true",
        help="Enable detailed performance tracking and logging",
        default=True
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=50,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="template-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="opentensor-dev",
    )

    # Validator-specific architecture arguments
    parser.add_argument(
        "--validator.enable_architecture_diversity_tracking",
        action="store_true",
        help="Enable tracking of architecture diversity among miners",
        default=True
    )

    parser.add_argument(
        "--validator.perturbation_testing_frequency",
        type=float,
        help="Frequency of perturbation testing (0.0 to 1.0)",
        default=0.2
    )

    parser.add_argument(
        "--validator.consensus_threshold",
        type=float,
        help="Threshold for consensus validation",
        default=0.9
    )

    parser.add_argument(
        "--validator.enable_audit_logging",
        action="store_true",
        help="Enable comprehensive audit logging",
        default=True
    )

    parser.add_argument(
        "--validator.benchmark_rotation_schedule",
        type=str,
        help="Schedule for rotating between different benchmarks",
        default="round_robin",
        choices=["round_robin", "random", "weighted"]
    )

    parser.add_argument(
        "--validator.diversity_bonus_weight",
        type=float,
        help="Weight for diversity bonus in scoring",
        default=0.1
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    cls.add_args(parser)
    return bt.Config(parser)
