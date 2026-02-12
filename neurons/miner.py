# The MIT License (MIT)
# Copyright 2026 SILX INC

import os
import time
import typing
import requests
import hashlib
import subprocess
import tempfile
import shutil
import json
import sys
from typing import Optional, List, Any, Dict, Tuple
from pydantic import Field
import torch
import bittensor as bt
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from cryptography.hazmat.primitives.asymmetric import ed25519

# Add the parent directory to path so we can import quasar
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.miner import BaseMinerNeuron
from quasar.utils.context_builder import (
    build_full_context,
    build_minimal_context,
    validate_repo_structure,
    estimate_context_tokens,
)

class Miner(BaseMinerNeuron):
    """
    QuasarSubnet Miner - Agent-Based Kernel Optimization
    
    This miner forks the flash-linear-attention repository, runs AI agents
    to optimize Quasar attention kernels, continuously tests performance,
    and submits improvements to the validator API.
    """

    TARGET_REPO = "https://github.com/troy12x/flash-linear-attention.git"
    TARGET_FILES = [
        "chunk.py",
        "chunk_intra_token_parallel.py",
        "forward_substitution.py",
        "fused_recurrent.py",
        "gate.py",
        "__init__.py"
    ]
    TEST_SEQUENCE_LENGTHS = [4096, 16384, 65536, 100000]
    REPORT_SEQUENCE_LENGTHS = [512, 1024, 2048]

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Set PyTorch CUDA memory allocation config to reduce fragmentation
        if torch.cuda.is_available():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Agent state
        self.active_agents = {}
        self.optimization_iterations = 0
        self.best_performance = {}
        
        # Mining statistics
        self.tasks_processed = 0
        self.tasks_succeeded = 0
        self.tasks_failed = 0
        self.start_time = time.time()
        
        bt.logging.info("Initializing QUASAR-SUBNET Miner...")
        print(f"\n [MINER] MY HOTKEY SS58: {self.wallet.hotkey.ss58_address} (COPY THIS FOR DASHBOARD)\n")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bt.logging.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            total_mem, free_mem = torch.cuda.mem_get_info()
            print(f"[MINER] GPU Memory: {free_mem/1024**3:.2f} GB free / {total_mem/1024**3:.2f} GB total", flush=True)
        
        # Get model name from config, env var, or use default (DeepSeek-V3.2 for better code understanding)
        self.model_name = os.getenv("MINER_MODEL_NAME", 
            getattr(self.config.miner, 'model_name', "Qwen/Qwen3-4B-Instruct-2507"))
        
        # Estimate model size and adjust parameters (Phase 2: Model-specific config)
        model_size = self._estimate_model_size(self.model_name)
        if model_size < 1.0:  # < 1B
            print(f"[MINER] ⚠️  WARNING: Model {self.model_name} is small (<1B). Consider using larger model (>1B) for better error fixing.")
            print(f"[MINER] ⚠️  Recommended: Qwen/Qwen3-4B-Instruct-2507 or set MINER_MODEL_NAME env var", flush=True)
            self.agent_max_new_tokens = 2048  # Reduce for small models
        elif model_size >= 4.0:  # >= 4B
            self.agent_max_new_tokens = 8192  # Increase for large models
            print(f"[MINER] Using large model ({self.model_name}, ~{model_size}B params). Increased max_new_tokens to {self.agent_max_new_tokens}", flush=True)
        else:  # 1B - 4B
            self.agent_max_new_tokens = 4096  # Default
            print(f"[MINER] Using medium model ({self.model_name}, ~{model_size}B params)", flush=True)
        
        # Agent generation parameters
        self.agent_max_length = 8192
        
        # Error tracking and success patterns (Phase 2 & 3)
        self.successful_fixes = {}  # Track successful error fixes by type
        self.failed_patterns = []   # Track patterns that consistently fail
        self.error_statistics = {}  # Track error frequency
        
        # Model will be loaded in load_model() method
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.model_loaded = False
        
        # GitHub configuration
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.github_username = os.getenv("GITHUB_USERNAME", "")
        self.fork_name = os.getenv("GITHUB_FORK_NAME", "")
        
        # Validator API URL
        self.validator_api_url = os.getenv("VALIDATOR_API_URL", "https://quasar-subnet.onrender.com")
        
        # Agent configuration
        self.agent_iterations = int(os.getenv("AGENT_ITERATIONS", "100"))
        self.target_sequence_length = int(os.getenv("TARGET_SEQUENCE_LENGTH", "100000"))
        self.optimization_interval = float(os.getenv("OPTIMIZATION_INTERVAL", "300"))  # 5 minutes
        
        # Context builder configuration (Phase 2: Full Repository Context)
        self.repo_path = os.getenv("REPO_PATH", None)  # Optional: local repo path for BYOC mode
        self.byoc_file_path = os.getenv("BYOC_FILE_PATH", None)  # Optional: expert's optimized code file
        self.use_full_context = os.getenv("USE_FULL_CONTEXT", "true").lower() == "true"
        self.context_max_files = int(os.getenv("CONTEXT_MAX_FILES", "50"))
        self.context_max_size = int(os.getenv("CONTEXT_MAX_SIZE", "200000"))
        self.repo_hash = None  # Will be calculated during optimization loop

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters (Phase 2)."""
        # Extract from name or use defaults
        name_lower = model_name.lower()
        if "0.5" in name_lower or "500m" in name_lower:
            return 0.5
        elif "1.5" in name_lower or "1.5b" in name_lower:
            return 1.5
        elif "2" in name_lower and ("2b" in name_lower or "2.5" in name_lower):
            return 2.5
        elif "3" in name_lower and ("3b" in name_lower or "3.5" in name_lower):
            return 3.5
        elif "4" in name_lower and ("4b" in name_lower or "4.5" in name_lower):
            return 4.0
        elif "8" in name_lower and ("8b" in name_lower or "8.5" in name_lower):
            return 8.0
        elif "1b" in name_lower or "1.0" in name_lower:
            return 1.0
        elif "7b" in name_lower:
            return 7.0
        elif "13b" in name_lower:
            return 13.0
        # DeepSeek models
        if "deepseek" in name_lower:
            if "v4" in name_lower or "v3.2" in name_lower:
                return 67.0  # DeepSeek-V3.2/V4 are ~67B parameters
            elif "v3" in name_lower or "v2" in name_lower:
                return 67.0  # DeepSeek-V3/V2 are ~67B parameters
            elif "coder" in name_lower:
                if "1.3" in name_lower or "1.5" in name_lower:
                    return 1.5
                elif "6.7" in name_lower or "7b" in name_lower:
                    return 7.0
                elif "33b" in name_lower:
                    return 33.0
            return 67.0  # Default DeepSeek to large model
        
        # Default assumption based on model name
        if "qwen3" in name_lower and "4b" in name_lower:
            return 4.0
        elif "qwen2.5" in name_lower and "0.5b" in name_lower:
            return 0.5
        return 1.0  # Default assumption

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model_loaded:
            return
        
        try:
            print(f" Loading tokenizer for {self.model_name}...")
            bt.logging.info(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            print(f" Loading model weights for {self.model_name}... (this can take several minutes)")
            # Don't use device_map="auto" if we need to move model to CPU for tests
            # Use explicit device placement instead
            use_device_map = os.getenv("MINER_USE_DEVICE_MAP", "false").lower() == "true"
            if use_device_map and torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            else:
                # Load to CPU first, then move to GPU (easier to move back to CPU later)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None  # Load to CPU first
                )
                if torch.cuda.is_available():
                    print(f"[MINER] Moving model to GPU...", flush=True)
                    self.model = self.model.to('cuda')
            
            self.max_length = getattr(self.config.miner, 'max_length', 32768)
            bt.logging.info(f"Miner max length set to: {self.max_length}")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f" Model Device Map: {self.model.hf_device_map}")
            else:
                print(f" Model Device: {self.model.device}")
                
            self.model.eval()
            
            # Ensure generation_config exists (required for generation)
            if not hasattr(self.model, 'generation_config') or self.model.generation_config is None:
                print("[MINER] Setting up generation_config...", flush=True)
                from transformers import GenerationConfig
                try:
                    self.model.generation_config = GenerationConfig.from_model_config(self.model.config)
                except Exception:
                    self.model.generation_config = GenerationConfig()
            
            print(f"\n [MINER] MY HOTKEY SS58: {self.wallet.hotkey.ss58_address} (COPY THIS FOR DASHBOARD)\n")
            bt.logging.info(f"Model loaded successfully: {self.model_name}")
            
            # Verify model is correct (Phase 3: Model verification)
            self._verify_model_loaded()
            
            if torch.cuda.is_available():
                total_mem, free_mem = torch.cuda.mem_get_info()
                print(f"[MINER] GPU Memory after model load: {free_mem/1024**3:.2f} GB free / {total_mem/1024**3:.2f} GB total", flush=True)
            
            self.model_loaded = True
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            raise e
    
    def _verify_model_loaded(self):
        """Verify that the correct model is loaded (Phase 3: Model verification)."""
        print(f"[MINER] Verifying model configuration...", flush=True)
        
        # Check model name matches
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path'):
            loaded_name = self.model.config.name_or_path
            if loaded_name != self.model_name:
                print(f"[MINER] ⚠️  WARNING: Model name mismatch!", flush=True)
                print(f"[MINER]   Expected: {self.model_name}", flush=True)
                print(f"[MINER]   Loaded: {loaded_name}", flush=True)
            else:
                print(f"[MINER] ✅ Model name verified: {loaded_name}", flush=True)
        
        # Check model device
        if torch.cuda.is_available():
            try:
                model_device = next(self.model.parameters()).device
                print(f"[MINER] ✅ Model device: {model_device}", flush=True)
            except StopIteration:
                print(f"[MINER] ⚠️  Could not determine model device", flush=True)
        
        # Check model size (parameter count)
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            params_b = total_params / 1e9
            print(f"[MINER] ✅ Model parameters: {params_b:.2f}B total, {trainable_params/1e9:.2f}B trainable", flush=True)
            
            # Verify model size matches expected
            expected_size = self._estimate_model_size(self.model_name)
            if abs(params_b - expected_size) > 0.5:  # Allow 0.5B difference
                print(f"[MINER] ⚠️  WARNING: Model size mismatch!", flush=True)
                print(f"[MINER]   Expected: ~{expected_size}B", flush=True)
                print(f"[MINER]   Actual: ~{params_b:.2f}B", flush=True)
        except Exception as e:
            print(f"[MINER] ⚠️  Could not count model parameters: {e}", flush=True)
        
        # Check tokenizer matches model
        if hasattr(self.tokenizer, 'model_max_length'):
            print(f"[MINER] ✅ Tokenizer max length: {self.tokenizer.model_max_length}", flush=True)
        
        # Print model configuration summary
        print(f"[MINER] Model Configuration Summary:", flush=True)
        print(f"  - Model Name: {self.model_name}", flush=True)
        print(f"  - Estimated Size: ~{self._estimate_model_size(self.model_name)}B parameters", flush=True)
        print(f"  - Max New Tokens: {self.agent_max_new_tokens}", flush=True)
        print(f"  - Max Retries: {int(os.getenv('MINER_MAX_RETRIES', '20'))}", flush=True)

    def _clean_gpu_memory(self):
        """Aggressively clean GPU memory - clear all caches and force garbage collection."""
        import gc
        if not torch.cuda.is_available():
            return
        
        # Clear any cached activations or intermediate states
        if self.model is not None:
            # Clear any cached states in the model (but NOT generation_config - it's required!)
            for attr in ['cache', 'past_key_values', '_past_key_values']:
                if hasattr(self.model, attr):
                    try:
                        cached = getattr(self.model, attr)
                        if cached is not None:
                            if isinstance(cached, dict):
                                cached.clear()
                            elif isinstance(cached, (list, tuple)):
                                for item in cached:
                                    if hasattr(item, 'cpu'):
                                        _ = item.cpu()
                                    del item
                            delattr(self.model, attr)
                    except Exception:
                        pass
            
            # Clear any module-level caches
            for module in self.model.modules():
                for attr in ['cache', 'past_key_values', '_past_key_values']:
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
        
        # Multiple rounds of aggressive cleanup
        for round_num in range(10):
            # Python garbage collection
            gc.collect()
            
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
            
            # Reset peak memory stats to clear tracking overhead
            if round_num == 0:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Small delay to let CUDA driver release memory
        time.sleep(0.5)

    def _move_model_to_cpu(self):
        """Temporarily move model to CPU to free GPU memory for tests."""
        import gc
        if self.model is None or not torch.cuda.is_available():
            return
        try:
            initial_mem = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[MINER] GPU memory before moving model to CPU: {initial_mem:.2f} GB", flush=True)

            # Step 1: Clean any cached states before moving
            print(f"[MINER] Cleaning cached states...", flush=True)
            self._clean_gpu_memory()

            # Step 2: Move model to CPU
            if hasattr(self.model, 'hf_device_map'):
                print(f"[MINER] Model uses device_map, using aggressive CPU offloading...", flush=True)
                moved_count = 0
                for name, param in self.model.named_parameters():
                    if param.is_cuda:
                        # Create new CPU tensor and replace
                        cpu_param = param.data.cpu()
                        param.data = cpu_param
                        moved_count += 1
                for name, buffer in self.model.named_buffers():
                    if buffer.is_cuda:
                        cpu_buffer = buffer.data.cpu()
                        buffer.data = cpu_buffer
                        moved_count += 1
                print(f"[MINER] Moved {moved_count} parameters/buffers to CPU", flush=True)
            else:
                print(f"[MINER] Moving standard model to CPU...", flush=True)
                
                # Clear any cached activations first
                if hasattr(self.model, 'cache'):
                    try:
                        del self.model.cache
                    except Exception:
                        pass
                
                # Move to CPU and explicitly delete old GPU model reference
                old_model = self.model
                self.model = old_model.to('cpu')
                
                # Explicitly delete old model to break GPU references
                del old_model
                gc.collect()
                
                # Force move any remaining GPU parameters/buffers
                for param in self.model.parameters():
                    if param.is_cuda:
                        cpu_param = param.data.cpu()
                        param.data = cpu_param
                        del cpu_param
                for name, buffer in self.model.named_buffers():
                    if buffer.is_cuda:
                        cpu_buffer = buffer.data.cpu()
                        buffer.data = cpu_buffer
                        del cpu_buffer

            # Step 3: Aggressive cleanup after moving
            print(f"[MINER] Performing aggressive GPU memory cleanup...", flush=True)
            self._clean_gpu_memory()

            # Step 4: Verify memory freed
            final_mem = torch.cuda.mem_get_info()[0] / 1024**3
            freed_mem = final_mem - initial_mem
            print(f"[MINER] Model moved to CPU. GPU memory: {initial_mem:.2f} GB -> {final_mem:.2f} GB (freed {freed_mem:.2f} GB)", flush=True)

            if freed_mem < 5.0:
                print(f"[MINER] ⚠️  WARNING: Only freed {freed_mem:.2f} GB. Model may not have moved completely.", flush=True)
                print(f"[MINER] ⚠️  Attempting emergency cleanup...", flush=True)
                
                # Emergency: try to force free by resetting CUDA context
                try:
                    # One more aggressive pass
                    for _ in range(20):
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    time.sleep(1.0)  # Longer delay
                    
                    final_mem_emergency = torch.cuda.mem_get_info()[0] / 1024**3
                    freed_mem_emergency = final_mem_emergency - initial_mem
                    print(f"[MINER] After emergency cleanup: {final_mem_emergency:.2f} GB free (freed {freed_mem_emergency:.2f} GB total)", flush=True)
                except Exception as e:
                    print(f"[MINER] Emergency cleanup failed: {e}", flush=True)

            # Step 5: Verify model is actually on CPU
            try:
                model_device = next(self.model.parameters()).device
                if model_device.type == 'cpu':
                    print(f"[MINER] ✅ Verified: Model is on CPU", flush=True)
                else:
                    print(f"[MINER] ⚠️  Warning: Model device is {model_device}, expected CPU", flush=True)
            except StopIteration:
                print(f"[MINER] ⚠️  Warning: Could not verify model device", flush=True)
                
        except Exception as e:
            print(f"[MINER] ⚠️  Error moving model to CPU: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def categorize_error(self, error_output: str) -> dict:
        """Categorize error and provide specific fix strategy (Phase 2: Error pattern recognition)."""
        error_type = "unknown"
        fix_strategy = ""
        severity = "medium"
        
        if "IndentationError" in error_output or "unexpected indent" in error_output:
            error_type = "indentation"
            fix_strategy = "Fix Python indentation - ensure top-level code at column 0, use 4 spaces per level"
            severity = "high"
        elif "size of tensor" in error_output or "must match" in error_output or "non-singleton dimension" in error_output:
            error_type = "tensor_shape"
            # Extract tensor sizes
            import re
            match = re.search(r"tensor a \((\d+)\) must match.*tensor b \((\d+)\)", error_output)
            if match:
                fix_strategy = f"Tensor shape mismatch: sizes {match.group(1)} vs {match.group(2)}. Check broadcasting dimensions and .view()/.expand() operations."
            else:
                fix_strategy = "Tensor shape mismatch - check all tensor operations for correct dimensions"
            severity = "high"
        elif "OutOfMemoryError" in error_output or "out of memory" in error_output.lower():
            error_type = "oom"
            fix_strategy = "Reduce memory usage - use smaller tensors, process in chunks, free intermediate tensors"
            severity = "critical"
        elif "AttributeError" in error_output:
            error_type = "attribute"
            fix_strategy = "Check object type and available methods - verify you're calling methods on correct object"
            severity = "medium"
        elif "NameError" in error_output:
            error_type = "name"
            fix_strategy = "Variable or function name not defined - check imports and variable names"
            severity = "medium"
        elif "TypeError" in error_output:
            error_type = "type"
            fix_strategy = "Type mismatch - check argument types and function signatures"
            severity = "medium"
        elif "SyntaxError" in error_output:
            error_type = "syntax"
            fix_strategy = "Python syntax error - check brackets, quotes, colons, indentation"
            severity = "high"
        elif "ImportError" in error_output or "ModuleNotFoundError" in error_output:
            error_type = "import"
            fix_strategy = "Import error - don't add new imports, use existing ones from fla.utils"
            severity = "high"
        
        return {
            'type': error_type,
            'strategy': fix_strategy,
            'severity': severity,
            'raw_error': error_output[:500]
        }

    def validate_extracted_code(self, code: str, filename: str) -> tuple[bool, str]:
        """Validate extracted code before writing (Phase 3: Code validation)."""
        issues = []
        
        if not code or not code.strip():
            return False, "Code is empty"
        
        # Check for basic syntax
        try:
            compile(code, filename, 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, "\n".join(issues)
        except Exception as e:
            issues.append(f"Compilation error: {e}")
            return False, "\n".join(issues)
        
        # Check for common issues
        if code.strip().startswith(' ') or code.strip().startswith('\t'):
            issues.append("Code starts with whitespace - will cause IndentationError")
        
        # Check for known problematic patterns
        if 'beta.view(-1, 1)' in code and 'beta.view(-1, 1, 1, 1, 1)' not in code:
            issues.append("Potential tensor shape error: beta.view(-1, 1) detected (should be beta.view(-1, 1, 1, 1, 1))")
        
        if 'alpha_expanded' in code and '.expand(' not in code and '.unsqueeze(' not in code:
            issues.append("alpha_expanded may not be expanded correctly - check expansion dimensions")
        
        # Check for forbidden imports
        forbidden_imports = ['fused_quasar_gate', 'ISAMD']  # Should be IS_AMD
        for forbidden in forbidden_imports:
            if f'import {forbidden}' in code or f'from.*{forbidden}' in code:
                issues.append(f"Forbidden import detected: {forbidden}")
        
        return len(issues) == 0, "\n".join(issues) if issues else "OK"

    def clean_code_content(self, code: str) -> str:
        """Clean extracted code content to fix indentation issues."""
        if not code:
            return code
        
        lines = code.split('\n')
        if not lines:
            return code
        
        # Find first non-empty line
        first_non_empty_idx = None
        for i, line in enumerate(lines):
            if line.strip():  # Non-empty line
                first_non_empty_idx = i
                break
        
        if first_non_empty_idx is None:
            return code  # All lines are empty
        
        # Check if first non-empty line has leading whitespace
        first_line = lines[first_non_empty_idx]
        if not first_line.startswith((' ', '\t')):
            return code  # No leading whitespace, return as-is
        
        # Calculate leading whitespace on first non-empty line
        leading_whitespace = len(first_line) - len(first_line.lstrip())
        
        # Remove that amount of leading whitespace from all lines
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                if line.startswith(' ' * leading_whitespace):
                    cleaned_lines.append(line[leading_whitespace:])
                elif line.startswith('\t' * (leading_whitespace // 4)):
                    # Handle tabs (assuming 4 spaces = 1 tab)
                    cleaned_lines.append(line[(leading_whitespace // 4):])
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)  # Keep empty lines as-is
        
        return '\n'.join(cleaned_lines)

    def _move_model_to_gpu(self):
        """Move model back to GPU after tests."""
        if self.model is not None and torch.cuda.is_available():
            try:
                # Ensure generation_config exists before moving (it's required for generation)
                if not hasattr(self.model, 'generation_config') or self.model.generation_config is None:
                    print("[MINER] Recreating generation_config...", flush=True)
                    from transformers import GenerationConfig
                    # Create a default generation config if missing
                    try:
                        self.model.generation_config = GenerationConfig.from_model_config(self.model.config)
                    except Exception:
                        # Fallback: create minimal config
                        self.model.generation_config = GenerationConfig()
                
                # Handle models loaded with device_map="auto"
                if hasattr(self.model, 'hf_device_map'):
                    # Use device_map="auto" to restore original device mapping
                    from transformers import AutoModelForCausalLM
                    # Reload with device_map if needed, or just move to cuda
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to('cuda')
                else:
                    # Standard model movement
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to('cuda')
                    elif hasattr(self.model, 'cuda'):
                        self.model = self.model.cuda()
                
                print("[MINER] Model moved back to GPU", flush=True)
            except Exception as e:
                print(f"[MINER] Warning: Could not move model back to GPU: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def _sign_message(self, message: str) -> str:
        """Sign a message with the wallet's private key."""
        signature = self.wallet.hotkey.sign(message.encode())
        return signature.hex()

    def _get_auth_headers(self) -> dict:
        """Get authentication headers for API requests."""
        signature = self._sign_message(self.wallet.hotkey.ss58_address)
        return {
            "Hotkey": self.wallet.hotkey.ss58_address,
            "Signature": signature,
        }

    def _api_request(self, method: str, path: str, *, headers: Optional[dict] = None, json: Optional[dict] = None, timeout: int = 120) -> requests.Response:
        """Make API request to validator."""
        url = f"{self.validator_api_url}{path}"
        try:
            print(f"[API] Request: {method} {url}", flush=True)
            print(f"[API] Headers: {headers}", flush=True)
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=timeout,
            )
            print(f"[API] Response status: {resp.status_code}", flush=True)
            return resp
        except Exception as e:
            bt.logging.error(f"API request failed: {e}")
            raise

    def create_github_fork(self) -> Tuple[str, str]:
        """Create a fork of the target repository on GitHub."""
        if not self.github_token or not self.github_username:
            raise ValueError("GITHUB_TOKEN and GITHUB_USERNAME environment variables required")
        
        # Extract owner and repo from TARGET_REPO
        repo_path = self.TARGET_REPO.replace("https://github.com/", "").replace(".git", "")
        owner, repo_name = repo_path.split("/")
        
        # Create fork via GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo_name}/forks"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        bt.logging.info(f"Creating fork of {owner}/{repo_name}...")
        print(f"[GITHUB] Creating fork of {owner}/{repo_name}...", flush=True)
        
        response = requests.post(api_url, headers=headers)
        response.raise_for_status()
        fork_data = response.json()
        
        fork_url = fork_data["html_url"]
        fork_owner = fork_data["owner"]["login"]
        
        # Wait for fork to be ready
        print(f"[GITHUB] Fork created: {fork_url}", flush=True)
        print(f"[GITHUB] Waiting for fork to be ready...", flush=True)
        time.sleep(5)
        
        return fork_url, fork_owner

    def clone_fork(self, fork_url: str, local_path: str = None) -> str:
        """Clone the fork locally."""
        if local_path is None:
            local_path = os.path.join(tempfile.gettempdir(), "flash-linear-attention-miner")
        
        # Remove existing repo if present
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        bt.logging.info(f"Cloning fork to {local_path}...")
        print(f"[GIT] Cloning {fork_url} to {local_path}...", flush=True)
        
        subprocess.run(
            ["git", "clone", fork_url, local_path],
            check=True,
            capture_output=True,
            text=True
        )

        # Install package in editable mode
        print(f"[PIP] Installing package in editable mode...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=local_path,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[PIP] Package installed successfully", flush=True)

        return local_path


    def get_commit_hash(self, repo_path: str) -> str:
        """Get the current commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()


    def submit_to_validator(self, fork_url: str, commit_hash: str, performance: float, benchmarks: Optional[Dict[int, Dict[str, float]]] = None) -> bool:
        """Submit optimization results to validator API."""
        try:
            if benchmarks is None:
                benchmarks = {}

            target_metrics = benchmarks.get(int(self.target_sequence_length), {"tokens_per_sec": performance, "vram_mb": 0.0})
            
            # Include repo_hash in signature for consistency
            signature_data = f"{fork_url}{commit_hash}{performance}{json.dumps(benchmarks, sort_keys=True)}"
            if self.repo_hash:
                signature_data += f"{self.repo_hash}"
            
            payload = {
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "fork_url": fork_url,
                "commit_hash": commit_hash,
                "repo_hash": self.repo_hash,  # Repository context hash for consistency
                "target_sequence_length": self.target_sequence_length,
                "tokens_per_sec": target_metrics.get("tokens_per_sec", performance),
                "vram_mb": float(target_metrics.get("vram_mb", 0.0)),
                "benchmarks": benchmarks,
                "signature": self._sign_message(signature_data)
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            bt.logging.info(f"Submitting to validator: {performance:.2f} tokens/sec")
            print(f"[API] Submitting to validator: {performance:.2f} tokens/sec", flush=True)

            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    response = self._api_request(
                        "POST",
                        "/submit_kernel",
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )

                    if response is None:
                        raise RuntimeError("Failed to create submission request")

                    print(f"[API] Response status: {response.status_code}", flush=True)
                    print(f"[API] Response text: {response.text[:500]}", flush=True)

                    if response.status_code == 422:
                        minimal_payload = {
                            "miner_hotkey": payload["miner_hotkey"],
                            "fork_url": payload["fork_url"],
                            "commit_hash": payload["commit_hash"],
                            "target_sequence_length": payload["target_sequence_length"],
                            "tokens_per_sec": payload["tokens_per_sec"],
                            "signature": self._sign_message(f"{fork_url}{commit_hash}{performance}"),
                        }
                        response = self._api_request(
                            "POST",
                            "/submit_optimization",
                            headers=headers,
                            json=minimal_payload,
                            timeout=120,
                        )

                        if response is None:
                            raise RuntimeError("Failed to create submission request")

                        response.raise_for_status()
                        result = response.json()
                        bt.logging.info(f"Submission successful: {result.get('submission_id')}")
                        print(f"[API] Submission successful: {result.get('submission_id')}", flush=True)
                        return True

                    response.raise_for_status()
                    result = response.json()
                    bt.logging.info(f"Submission successful: {result.get('submission_id')}")
                    print(f"[API] Submission successful: {result.get('submission_id')}", flush=True)
                    return True
                except Exception as e:
                    last_err = e
                    bt.logging.warning(f"Submission attempt {attempt + 1}/3 failed: {e}")
                    print(f"[API] Submission attempt {attempt + 1}/3 failed: {e}", flush=True)
                    time.sleep(2 * (attempt + 1))

            if last_err is not None:
                raise last_err
            
        except Exception as e:
            bt.logging.warning(f"Submission failed: {e}")
            print(f"[API] Submission failed: {e}", flush=True)
            return False


    def run_optimization_loop(self, fork_url: str, repo_path: str):
        """Run the main optimization loop."""
        import re
        from threading import Thread

        print(f"[MINER] Starting optimization loop...", flush=True)
        bt.logging.info("Starting optimization loop")
        
        # Store repo_path for context builder
        self.repo_path = repo_path
        
        # Validate repository structure
        is_valid, warnings = validate_repo_structure(repo_path)
        if warnings:
            print(f"[MINER] ⚠️  Repository validation warnings:", flush=True)
            for warning in warnings:
                print(f"  - {warning}", flush=True)
        
        # Build repository context (once, reused across iterations)
        repo_context = None
        repo_hash = None
        if self.use_full_context:
            try:
                print(f"[MINER] Building full repository context...", flush=True)
                repo_context = build_full_context(
                    repo_path=repo_path,
                    target_file="chunk.py",
                    include_tree=True,
                    max_files=self.context_max_files,
                    max_size=self.context_max_size,
                    byoc_mode=self.byoc_file_path is not None,
                    byoc_file_path=self.byoc_file_path
                )
                context_tokens = estimate_context_tokens(repo_context)
                
                # Calculate repo_hash for consistency tracking
                import hashlib
                repo_hash = hashlib.sha256(repo_context.encode()).hexdigest()[:16]
                
                print(f"[MINER] ✅ Repository context built: ~{context_tokens} tokens, hash: {repo_hash}", flush=True)
                bt.logging.info(f"Repository context built: ~{context_tokens} tokens, hash: {repo_hash}")
                
                # Store repo_hash for submission
                self.repo_hash = repo_hash
            except Exception as e:
                print(f"[MINER] ⚠️  Failed to build full context: {e}. Using minimal context.", flush=True)
                bt.logging.warning(f"Failed to build full context: {e}")
                repo_context = None
                self.repo_hash = None
        else:
            self.repo_hash = None

        for iteration in range(self.agent_iterations):
            print(f"\n[MINER] --- Iteration {iteration + 1}/{self.agent_iterations} ---", flush=True)
            
            # Read current files
            file_contents = {}
            for filename in self.TARGET_FILES:
                filepath = os.path.join(repo_path, "fla", "ops", "quasar", filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        file_contents[filename] = f.read()

            # Construct prompt with full context
            context_note = ""
            if repo_context:
                context_note = (
                    "\n\nIMPORTANT: You have been provided with FULL REPOSITORY CONTEXT including:\n"
                    "- Complete file tree\n"
                    "- All relevant source files (.py, .cu, .cuh, .h)\n"
                    "- Files that reference chunk_quasar, quasar, attention, and kernel functionality\n"
                    "- Use this context to understand the complete codebase contract and ensure compatibility.\n"
                )
            
            system_prompt = (
                "You are an expert AI kernel engineer optimizing Quasar Attention.\n"
                "You are part of an ITERATIVE ERROR-FIXING SYSTEM.\n"
                + context_note +
                "\n"
                "🔴 CRITICAL: INSTRUCTION FOLLOWING IS MANDATORY\n"
                "─────────────────────────────────────────────────────────────────────\n"
                "You MUST follow ALL instructions provided in the context EXACTLY.\n"
                "When expert code or specific instructions are provided:\n"
                "1. READ the instructions CAREFULLY before generating code\n"
                "2. UNDERSTAND what is being asked - do not skip or ignore instructions\n"
                "3. FOLLOW the step-by-step guidance provided\n"
                "4. USE expert code as your reference when provided\n"
                "5. VERIFY your output matches the requirements before submitting\n"
                "6. DO NOT deviate from explicit instructions - they are there for a reason\n"
                "\n"
                "When you see errors, you MUST:\n"
                "1. Read the error message CAREFULLY\n"
                "2. Identify the EXACT line and operation causing the error\n"
                "3. Understand WHY the error occurred (tensor shape mismatch, syntax error, etc.)\n"
                "4. Make a MINIMAL fix that addresses ONLY that specific error\n"
                "5. Keep ALL other code unchanged\n"
                "6. Verify your fix by checking tensor shapes match before operations\n\n"
                "COMMON ERROR PATTERNS YOU MUST FIX:\n"
                "1. Tensor shape mismatch: Check all .view(), .expand(), .unsqueeze() operations\n"
                "2. IndentationError: Ensure code starts at column 0, use 4 spaces for indentation\n"
                "3. AttributeError: Check that you're calling methods on the correct object type\n"
                "4. Import errors: Don't add new imports, use existing ones\n\n"
                "ERROR-FIXING WORKFLOW:\n"
                "1. Read the error message\n"
                "2. Find the file and line number mentioned\n"
                "3. Look at that specific line in the code\n"
                "4. Understand what's wrong (shape mismatch, syntax, etc.)\n"
                "5. Make the MINIMAL change needed to fix it\n"
                "6. Output the COMPLETE file with your fix\n\n"
                "CRITICAL OUTPUT FORMAT:\n"
                "You MUST wrap your code in markdown code blocks like this example:\n\n"
                "```python:chunk.py\n"
                "import torch\n"
                "import triton\n"
                "# ... rest of the code ...\n"
                "```\n\n"
                "CRITICAL IMPORT RULES:\n"
                "1. DO NOT add new imports\n"
                "2. DO NOT import fused_quasar_gate\n"
                "3. Keep ALL existing imports EXACTLY as they are\n"
                "4. MUST use correct import names from fla.utils:\n"
                "   - Use 'IS_AMD' (with underscore), NOT 'ISAMD'\n"
                "   - Use 'autocast_custom_bwd', 'autocast_custom_fwd'\n"
                "   - Use 'autotune_cache_kwargs', 'check_shared_mem', 'input_guard'\n"
                "5. MUST include: chunk_quasar_fwd, ChunkQuasarFunction, chunk_quasar\n"
                "6. The code fence line is NOT Python code - it's markdown formatting\n"
                "7. Start your actual Python code on the line AFTER the fence\n\n"
                "MEMORY EFFICIENCY RULES:\n"
                "1. Use memory-efficient operations - avoid large intermediate tensors\n"
                "2. Process data in chunks when possible to reduce memory footprint\n"
                "3. Use in-place operations (e.g., tensor.add_(), tensor.mul_()) where safe\n"
                "4. Free intermediate tensors with 'del tensor' after use\n"
                "5. Use tensor views (.view(), .reshape()) carefully - ensure tensors are contiguous\n"
                "6. Avoid creating copies unnecessarily - use .contiguous() only when needed\n"
            )

            # Build user prompt with context
            if repo_context:
                # Use full repository context
                user_prompt = repo_context + "\n\n"
                user_prompt += "# ════════════════════════════════════════════════════════════════════════\n"
                user_prompt += "# CURRENT TARGET FILES (for reference)\n"
                user_prompt += "# ════════════════════════════════════════════════════════════════════════\n\n"
                for fname, content in file_contents.items():
                    if fname in ["chunk.py", "fused_recurrent.py", "gate.py"]:
                        user_prompt += f"=== {fname} ===\n{content}\n\n"
                
                # Add instruction-following emphasis
                if self.byoc_file_path:
                    user_prompt += (
                        "# 🔴 CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
                        "# ────────────────────────────────────────────────────────────────────────\n"
                        "# Expert code has been provided above. You MUST:\n"
                        "# 1. USE the expert code as your PRIMARY reference\n"
                        "# 2. FOLLOW the implementation approach shown in expert code\n"
                        "# 3. ADAPT the expert code structure to match repository conventions\n"
                        "# 4. DO NOT ignore or deviate from the expert code\n"
                        "# 5. VERIFY all function signatures match before outputting\n"
                        "# ────────────────────────────────────────────────────────────────────────\n\n"
                    )
                
                user_prompt += (
                    "Based on the FULL REPOSITORY CONTEXT above, please rewrite `chunk.py` and `fused_recurrent.py` "
                    "to use the kernelized gate mechanism from `gate.py`. Remove the pure PyTorch alpha/beta computation.\n\n"
                    "CRITICAL: Ensure `chunk_quasar` function is exported correctly and matches the expected API from the codebase.\n\n"
                    "REMEMBER: Follow ALL instructions provided in the context above. Read the MANDATORY INSTRUCTIONS section carefully "
                    "and verify your output matches all requirements."
                )
            else:
                # Fallback to minimal context (original behavior)
                user_prompt = "Here are the current files:\n\n"
                for fname, content in file_contents.items():
                    if fname in ["chunk.py", "fused_recurrent.py", "gate.py"]:
                        user_prompt += f"=== {fname} ===\n{content}\n\n"
                
                user_prompt += (
                    "Please rewrite `chunk.py` and `fused_recurrent.py` to use the kernelized gate mechanism from `gate.py`. "
                    "Remove the pure PyTorch alpha/beta computation."
                )

            # Generate code with streaming
            # We wrap this in a retry loop to handle test failures
            # Increased retries for larger models that can handle more iterations
            max_retries = int(os.getenv("MINER_MAX_RETRIES", "20"))  # Increased for better error fixing
            success = False
            previous_error = None
            error_history = []  # Track error history for cumulative learning

            for attempt in range(max_retries):
                # Ensure model is on GPU for code generation (if it was moved to CPU)
                if self.model is not None and torch.cuda.is_available():
                    try:
                        model_device = next(self.model.parameters()).device
                        if model_device.type == 'cpu':
                            print("[MINER] Model is on CPU, moving to GPU for code generation...", flush=True)
                            self._move_model_to_gpu()
                    except (StopIteration, AttributeError):
                        pass
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

                if attempt > 0:
                    print(f"\n[MINER] --- Attempt {attempt + 1}/{max_retries} (Retry with feedback) ---", flush=True)
                    # Exponential backoff for thinking time (but cap at 10s)
                    wait_time = min(2 ** (attempt - 1), 10)
                    if wait_time > 0:
                        print(f"[MINER] Waiting {wait_time}s for model to process error feedback...", flush=True)
                        time.sleep(wait_time)

                # Construct messages afresh to save context window
                # We start with the base prompts
                current_system_prompt = system_prompt
                current_user_prompt = user_prompt
                
                # If we have a previous error, append it to the user prompt to give feedback
                # WITHOUT keeping the entire previous failed conversation history
                if previous_error:
                    # Add error to history
                    error_history.append({
                        'attempt': attempt,
                        'error': previous_error,
                        'timestamp': time.time()
                    })
                    
                    # Build cumulative error context (last 3 errors to avoid too much context)
                    recent_errors = error_history[-3:] if len(error_history) > 3 else error_history
                    error_context = "\n\n".join([
                        f"--- Attempt {e['attempt'] + 1} Error ---\n{e['error'][:500]}" 
                        for e in recent_errors
                    ])
                    
                    current_user_prompt = (
                        f"ITERATIVE ERROR FIXING - Attempt {attempt + 1}/{max_retries}\n\n"
                        f"PREVIOUS ERRORS (learn from these - don't repeat the same mistakes):\n"
                        f"{error_context}\n\n"
                        f"CURRENT BROKEN CODE:\n\n"
                        f"=== chunk.py (CURRENT - HAS ERRORS) ===\n"
                        f"{file_contents.get('chunk.py', '')}\n\n"
                        f"LATEST ERROR (this is what you need to fix NOW):\n{previous_error}\n\n"
                        f"CRITICAL INSTRUCTIONS:\n"
                        f"1. Analyze ALL previous errors above - don't repeat the same mistakes\n"
                        f"2. Look at the LATEST error trace carefully\n"
                        f"3. Find the EXACT line causing the error\n"
                        f"4. Understand WHY it's failing (tensor shape? syntax? attribute?)\n"
                        f"5. Fix ONLY that specific issue with MINIMAL changes\n"
                        f"6. Output the COMPLETE corrected file (don't skip any parts)\n"
                        f"7. Keep ALL imports and ALL functions intact\n"
                        f"8. Verify tensor shapes match before operations\n"
                        f"9. If you see the same error pattern from previous attempts, try a DIFFERENT approach\n"
                    )

                messages = [
                    {"role": "system", "content": current_system_prompt},
                    {"role": "user", "content": current_user_prompt}
                ]
                
                print(f"[MINER] Generating code...", flush=True)
                
                # Apply chat template - returns BatchEncoding (dict-like object)
                tokenized = self.tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
                
                # Extract input_ids from BatchEncoding
                if hasattr(tokenized, 'input_ids'):
                    input_ids = tokenized['input_ids']
                elif isinstance(tokenized, dict):
                    input_ids = tokenized.get('input_ids', tokenized)
                else:
                    input_ids = tokenized
                
                # Ensure it's a tensor and move to device
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                
                input_ids = input_ids.to(self.device)
                
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
                generation_kwargs = dict(
                    input_ids=input_ids,
                    streamer=streamer,
                    max_new_tokens=self.agent_max_new_tokens,
                    temperature=0.7,
                )
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                accumulated_response = ""
                for new_text in streamer:
                    print(new_text, end="", flush=True)
                    accumulated_response += new_text
                thread.join()
                print() 
                
                # Aggressively clear GPU memory after generation
                print("[MINER] Clearing GPU memory after code generation...", flush=True)
                del input_ids
                if torch.cuda.is_available():
                    # Use the aggressive cleanup method
                    self._clean_gpu_memory()
                print(f"[MINER] GPU memory cleared. Free: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB", flush=True)

 

                # Parse and apply changes
                # Strategy 1: Look for explicit filename in code block tag
                # Updated regex to handle trailing chars after filename on the code fence line
                pattern_strict = r"```python:([a-zA-Z0-9_]+\.py).*?\n(.*?)```"
                matches_strict = list(re.finditer(pattern_strict, accumulated_response, re.DOTALL))
                
                # Strategy 2: Look for standard python blocks and infer filename from content
                pattern_lax = r"```python\n(.*?)```"
                matches_lax = list(re.finditer(pattern_lax, accumulated_response, re.DOTALL))
                
                modified_files = set()
                import difflib

                def apply_update_with_diff(fname, new_content):
                    f_path = os.path.join(repo_path, "fla", "ops", "quasar", fname)
                    old_code = ""
                    if os.path.exists(f_path):
                        with open(f_path, 'r') as f:
                            old_code = f.read()
                    
                    # Clean the extracted content to fix indentation issues
                    new_content = self.clean_code_content(new_content)
                    
                    # Validate code before writing (Phase 3: Code validation)
                    is_valid, validation_issues = self.validate_extracted_code(new_content, fname)
                    if not is_valid:
                        print(f"[MINER] ⚠️  Code validation warnings for {fname}:", flush=True)
                        for issue in validation_issues.split('\n'):
                            if issue:
                                print(f"[MINER]   - {issue}", flush=True)
                        # Still write it, but log the issues - model can fix them
                    
                    print(f"\n[MINER] Diff for {fname}:", flush=True)
                    diff_generator = difflib.unified_diff(
                        old_code.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"current/{fname}",
                        tofile=f"new/{fname}",
                        n=3
                    )
                    diff_text = "".join(diff_generator)
                    if diff_text:
                        print(diff_text, flush=True)
                    else:
                        print("(No changes)", flush=True)
                    
                    line_count = len(new_content.splitlines())
                    print(f"[MINER] New file has {line_count} lines", flush=True)

                    with open(f_path, 'w') as f:
                        f.write(new_content)

                # Process strict matches first
                for match in matches_strict:
                    filename = match.group(1)
                    content = match.group(2)
                    if filename in self.TARGET_FILES:
                         apply_update_with_diff(filename, content)
                         print(f"[MINER] Updated {filename} (strict match)", flush=True)
                         modified_files.add(filename)

                # Process lax matches if we haven't found everything
                for match in matches_lax:
                    content = match.group(1)
                    filename = None
                    
                    # Heuristic content matching
                    if "def chunk_quasar_fwd" in content or "class ChunkQuasarFunction" in content:
                        filename = "chunk.py"
                    elif "def fused_recurrent_quasar_fwd" in content or "class FusedRecurrentQuasarFunction" in content:
                        filename = "fused_recurrent.py"
                    elif "def fused_quasar_gate" in content or "def quasar_gate_fwd" in content:
                        filename = "gate.py"
                    elif "def forward_substitution" in content:
                        filename = "forward_substitution.py"
                    
                    if filename and filename in self.TARGET_FILES and filename not in modified_files:
                         apply_update_with_diff(filename, content)
                         print(f"[MINER] Updated {filename} (inferred from content)", flush=True)
                         modified_files.add(filename)
                
                if not modified_files:
                    print("[MINER] No valid files generated. Adding feedback and retrying...", flush=True)
                    previous_error = "You did not generate any valid code blocks. Please output the full code for `chunk.py` and `fused_recurrent.py` using ```python:filename.py``` blocks. Ensure you provide the COMPLETE file content."
                    continue

                # Install and Test
                try:
                    print("[MINER] Installing package...", flush=True)
                    # Ensure we are in the repo path
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-e", "."],
                        cwd=repo_path,
                        check=True,
                        capture_output=True
                    )
                    
                    # Move model to CPU to free GPU memory for tests
                    print("[MINER] Moving model to CPU to free GPU memory for tests...", flush=True)
                    self._move_model_to_cpu()
                    
                    # Verify model is on CPU and get final memory state
                    if torch.cuda.is_available():
                        # Additional aggressive cleanup before tests
                        print("[MINER] Final GPU memory cleanup before tests...", flush=True)
                        self._clean_gpu_memory()
                        
                        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                        total_mem = torch.cuda.mem_get_info()[1] / 1024**3
                        print(f"[MINER] GPU memory before tests: {free_mem:.2f} GB free / {total_mem:.2f} GB total", flush=True)
                        
                        # Warn if memory is still low
                        if free_mem < 5.0:
                            print(f"[MINER] ⚠️  WARNING: Low GPU memory ({free_mem:.2f} GB). Tests may fail with OOM.", flush=True)
                            print(f"[MINER] ⚠️  The test script will automatically reduce parameters, but this may still fail.", flush=True)
                        
                        # Verify model is actually on CPU
                        try:
                            if self.model is not None:
                                model_device = next(self.model.parameters()).device
                                if model_device.type != 'cpu':
                                    print(f"[MINER] ⚠️  WARNING: Model is still on {model_device}, not CPU! This may cause OOM.", flush=True)
                        except (StopIteration, AttributeError):
                            pass
                    
                    print("[MINER] Running tests...", flush=True)
                    # Location of the test script we created
                    test_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "test_quasar_mining.py")
                    if not os.path.exists(test_script_path):
                         # Fallback to current dir / tests
                         test_script_path = os.path.abspath("tests/test_quasar_mining.py")

                    # Set environment variable for memory management
                    env = os.environ.copy()
                    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    
                    result = subprocess.run(
                        [sys.executable, test_script_path],
                        cwd=repo_path, 
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    
                    if result.returncode != 0:
                        print(f"[MINER] Tests Failed:\n{result.stderr}", flush=True)
                        
                        # Check if it's an OOM error
                        error_output = result.stderr + result.stdout
                        is_oom = "OutOfMemoryError" in error_output or "out of memory" in error_output.lower()
                        
                        if is_oom:
                            # Phase 2: Use error categorization
                            error_info = self.categorize_error(error_output)
                            print(f"[MINER] Error categorized: {error_info['type']} (severity: {error_info['severity']})", flush=True)
                            
                            # Track error statistics
                            self.error_statistics[error_info['type']] = self.error_statistics.get(error_info['type'], 0) + 1
                            
                            # Clear GPU memory after OOM (model should already be on CPU)
                            print("[MINER] OOM detected, performing aggressive GPU memory cleanup...", flush=True)
                            if torch.cuda.is_available():
                                # Use aggressive cleanup
                                self._clean_gpu_memory()
                                # Ensure model is on CPU
                                self._move_model_to_cpu()
                                # One more cleanup pass
                                self._clean_gpu_memory()
                            
                            previous_error = (
                                f"CUDA Out of Memory Error detected.\n"
                                f"The code is trying to allocate too much GPU memory.\n"
                                f"STDERR:\n{result.stderr}\n\n"
                                f"CRITICAL FIXES NEEDED:\n"
                                f"1. Use memory-efficient operations (avoid large intermediate tensors)\n"
                                f"2. Process data in smaller chunks if needed\n"
                                f"3. Use in-place operations where possible\n"
                                f"4. Avoid creating large tensor views/reshapes that require copies\n"
                                f"5. Free intermediate tensors with del after use\n"
                                f"6. The error occurred at: {error_output.split('File')[1].split('line')[0] if 'File' in error_output else 'unknown location'}\n"
                            )
                        else:
                            # Set error for next attempt with BOTH stdout and stderr for context
                            previous_error = (
                                f"The code failed to run.\n"
                                f"STDOUT:\n{result.stdout}\n"
                                f"STDERR:\n{result.stderr}\n\n"
                                f"CRITICAL: Fix the specific error shown above. Deeply analyze the trace and correct your code."
                            )
                        # Move model back to GPU for next retry attempt (need to generate code)
                        print("[MINER] Moving model back to GPU for next retry...", flush=True)
                        self._move_model_to_gpu()
                        continue # Retry
                    else:
                        print(result.stdout)
                        print(f"[MINER] Tests Passed!", flush=True)
                        # Extract score
                        tps_match = re.search(r"QuasarAttention achieved: (\d+) tokens/sec", result.stdout)
                        if tps_match:
                            score = float(tps_match.group(1))
                            bt.logging.info(f"Iteration {iteration} score: {score}")
                            print(f"[MINER] Benchmark Score: {score} tokens/sec")
                            
                            # Track successful fix (Phase 3: Success tracking)
                            if previous_error:
                                error_info = self.categorize_error(previous_error)
                                error_type = error_info.get('type', 'unknown')
                                self.successful_fixes[error_type] = self.successful_fixes.get(error_type, 0) + 1
                                print(f"[MINER] ✅ Successfully fixed {error_type} error (total fixes for this type: {self.successful_fixes[error_type]})", flush=True)
                            
                            # Move model back to GPU for next iteration
                            print("[MINER] Moving model back to GPU...", flush=True)
                            self._move_model_to_gpu()
                            
                            # Submit
                            commit_hash = self.get_commit_hash(repo_path)
                            self.submit_to_validator(fork_url, commit_hash, score)
                            success = True
                            break # Success, exit retry loop
                            
                except Exception as e:
                    print(f"[MINER] Error during build/test: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    messages.append({"role": "user", "content": f"System error during build/test: {e}"})

            if not success:
                print(f"[MINER] Optimization failed after {max_retries} attempts. Continuing to next iteration...", flush=True)
                
                # Log error statistics (Phase 3: Success tracking)
                if self.error_statistics:
                    print(f"[MINER] Error statistics for this iteration:", flush=True)
                    for error_type, count in sorted(self.error_statistics.items(), key=lambda x: x[1], reverse=True):
                        print(f"[MINER]   {error_type}: {count} occurrences", flush=True)
                
                if self.successful_fixes:
                    print(f"[MINER] Successful fixes so far:", flush=True)
                    for error_type, count in sorted(self.successful_fixes.items(), key=lambda x: x[1], reverse=True):
                        print(f"[MINER]   {error_type}: {count} fixes", flush=True)

            # Do NOT run install/test again here - the retry loop already did it.
            # Running tests again here would run with the model still on GPU (miner process
            # holds ~43 GB), so the test subprocess would OOM. Only the loop above runs tests
            # (after moving the model to CPU).

            print(f"[MINER] Waiting {self.optimization_interval}s...", flush=True)
            time.sleep(self.optimization_interval)

    def run_miner(self):
        """Main miner entry point."""
        try:
            print(f"\n[MINER] ========== Starting QuasarSubnet Miner ==========", flush=True)
            print(f"[MINER] Target repo: {self.TARGET_REPO}", flush=True)
            print(f"[MINER] Target sequence length: {self.target_sequence_length}", flush=True)
            print(f"[MINER] Agent iterations: {self.agent_iterations}", flush=True)
            print(f"[MINER] Optimization interval: {self.optimization_interval}s", flush=True)
            
            # Step 1: Create fork
            fork_url, fork_owner = self.create_github_fork()
            print(f"[MINER] Fork created: {fork_url}", flush=True)
            
            # Step 2: Clone fork
            repo_path = self.clone_fork(fork_url)
            print(f"[MINER] Fork cloned to: {repo_path}", flush=True)
            
            # Step 3: Run optimization loop
            self.run_optimization_loop(fork_url, repo_path)
            
            # Cleanup
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
                bt.logging.info(f"Cleaned up repository at {repo_path}")
            
            print(f"\n[MINER] ========== Miner completed ==========", flush=True)
            
        except Exception as e:
            bt.logging.error(f"Miner failed: {e}")
            print(f"[MINER] Error: {e}", flush=True)
            raise

    async def forward(self, synapse: quasar.protocol.BenchmarkEvaluationSynapse) -> quasar.protocol.BenchmarkEvaluationSynapse:
        """Not used - miner runs optimization loop."""
        synapse.response = "Miner runs optimization loop. Please use validator_api."
        synapse.processing_time = 0.0
        return synapse

    async def blacklist(self, synapse: quasar.protocol.BenchmarkEvaluationSynapse) -> typing.Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"
        return False, "Hotkey recognized!"

    async def priority(self, synapse: quasar.protocol.BenchmarkEvaluationSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

if __name__ == "__main__":
    import argparse
    
    # Create parser and add all Bittensor base arguments
    parser = argparse.ArgumentParser(description="QuasarSubnet Miner")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    Miner.add_args(parser)  # Adds miner-specific Bittensor args
    
    # Add custom miner-specific arguments
    parser.add_argument("--agent-iterations", type=int, default=100,
                       help="Number of agent optimization iterations (default: 100)")
    parser.add_argument("--target-seq-len", type=int, default=100000,
                       help="Target sequence length (default: 100000)")
    parser.add_argument("--optimization-interval", type=float, default=300,
                       help="Interval between iterations in seconds (default: 300)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Model name to use for optimization (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: only optimize chunk.py for quick testing")
    parser.add_argument("--max-length", type=int, default=8192,
                       help="Max token length for input tokenization (default: 8192)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                       help="Max new tokens to generate (default: 4096)")
    parser.add_argument("--repo-path", type=str, default=None,
                       help="Path to local repository (if not provided, will clone fork)")
    parser.add_argument("--byoc-file", type=str, default=None,
                       help="Path to expert's optimized code file (Bring-Your-Own-Code mode)")
    parser.add_argument("--use-full-context", action="store_true", default=None,
                       help="Use full repository context (default: from USE_FULL_CONTEXT env var)")
    parser.add_argument("--context-max-files", type=int, default=50,
                       help="Maximum number of files to include in context (default: 50)")
    parser.add_argument("--context-max-size", type=int, default=200000,
                       help="Maximum context size in characters (default: 200000)")
    
    # Create config - bt.Config will parse sys.argv automatically
    config = bt.Config(parser)
    
    # Parse args again to get custom arguments (bt.Config already parsed, but we need the namespace)
    args = parser.parse_args()

    with Miner(config=config) as miner:
        # Override config with command line args (from parsed args)
        miner.agent_iterations = args.agent_iterations
        miner.target_sequence_length = args.target_seq_len
        miner.optimization_interval = args.optimization_interval
        miner.model_name = args.model_name
        miner.agent_max_length = args.max_length
        miner.agent_max_new_tokens = args.max_new_tokens

        # Test mode: only optimize chunk.py
        if args.test_mode:
            miner.TARGET_FILES = ["chunk.py"]
            print(f"[MINER] Test mode: only optimizing chunk.py", flush=True)
        
        # Context builder configuration
        if args.repo_path:
            miner.repo_path = args.repo_path
            print(f"[MINER] Using provided repo path: {args.repo_path}", flush=True)
        if args.byoc_file:
            miner.byoc_file_path = args.byoc_file
            print(f"[MINER] BYOC mode enabled: {args.byoc_file}", flush=True)
        if args.use_full_context is not None:
            miner.use_full_context = args.use_full_context
        miner.context_max_files = args.context_max_files
        miner.context_max_size = args.context_max_size
        
        print(f"[MINER] Context configuration:", flush=True)
        print(f"  - Use full context: {miner.use_full_context}", flush=True)
        print(f"  - Max files: {miner.context_max_files}", flush=True)
        print(f"  - Max size: {miner.context_max_size} chars", flush=True)
        
        # Load model
        print(" [MINER] Loading model...")
        miner.load_model()
        
        # Run miner
        try:
            miner.run_miner()
        except KeyboardInterrupt:
            print("\n [MINER] Shutting down...")
            miner.should_exit = True