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
        
        # Get model name from config or use default
        self.model_name = getattr(self.config.miner, 'model_name', "Qwen/Qwen2.5-0.5B-Instruct")
        
        # Agent generation parameters
        self.agent_max_length = 8192
        self.agent_max_new_tokens = 4096

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

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model_loaded:
            return
        
        try:
            print(f" Loading tokenizer for {self.model_name}...")
            bt.logging.info(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            print(f" Loading model weights for {self.model_name}... (this can take several minutes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.max_length = getattr(self.config.miner, 'max_length', 32768)
            bt.logging.info(f"Miner max length set to: {self.max_length}")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f" Model Device Map: {self.model.hf_device_map}")
            else:
                print(f" Model Device: {self.model.device}")
                
            self.model.eval()
            print(f"\n [MINER] MY HOTKEY SS58: {self.wallet.hotkey.ss58_address} (COPY THIS FOR DASHBOARD)\n")
            bt.logging.info(f"Model loaded successfully: {self.model_name}")
            self.model_loaded = True
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            raise e

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
            payload = {
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "fork_url": fork_url,
                "commit_hash": commit_hash,
                "target_sequence_length": self.target_sequence_length,
                "tokens_per_sec": target_metrics.get("tokens_per_sec", performance),
                "vram_mb": float(target_metrics.get("vram_mb", 0.0)),
                "benchmarks": benchmarks,
                "signature": self._sign_message(f"{fork_url}{commit_hash}{performance}{json.dumps(benchmarks, sort_keys=True)}")
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

        for iteration in range(self.agent_iterations):
            print(f"\n[MINER] --- Iteration {iteration + 1}/{self.agent_iterations} ---", flush=True)
            
            # Read current files
            file_contents = {}
            for filename in self.TARGET_FILES:
                filepath = os.path.join(repo_path, "fla", "ops", "quasar", filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        file_contents[filename] = f.read()

            # Construct prompt
            system_prompt = (
                "You are an expert AI kernel engineer optimizing Quasar Attention.\n"
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
            max_retries = 10  # Increased from 3 to give model more chances to fix errors
            success = False
            previous_error = None

            for attempt in range(max_retries):
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

                if attempt > 0:
                    print(f"\n[MINER] --- Attempt {attempt + 1}/{max_retries} (Retry with feedback) ---", flush=True)

                # Construct messages afresh to save context window
                # We start with the base prompts
                current_system_prompt = system_prompt
                current_user_prompt = user_prompt
                
                # If we have a previous error, append it to the user prompt to give feedback
                # WITHOUT keeping the entire previous failed conversation history
                if previous_error:
                    current_user_prompt = (
                        f"The code you generated has an error. Here is the CURRENT broken file:\n\n"
                        f"=== chunk.py (CURRENT - HAS ERRORS) ===\n"
                        f"{file_contents.get('chunk.py', '')}\n\n"
                        f"ERROR FROM RUNNING TESTS:\n{previous_error}\n\n"
                        f"CRITICAL INSTRUCTIONS:\n"
                        f"1. Look at the error trace carefully\n"
                        f"2. Find the EXACT line causing the error\n"
                        f"3. Fix ONLY that specific issue\n"
                        f"4. Output the COMPLETE corrected file (don't skip any parts)\n"
                        f"5. Keep ALL imports and ALL functions intact\n"
                        f"6. Make MINIMAL changes - only fix the error\n"
                    )

                messages = [
                    {"role": "system", "content": current_system_prompt},
                    {"role": "user", "content": current_user_prompt}
                ]
                
                print(f"[MINER] Generating code...", flush=True)
                
                # Apply chat template - may return BatchEncoding or tensor
                tokenized = self.tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
                
                # Extract input_ids if it's a BatchEncoding, otherwise use directly
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
                print("[MINER] Clearing GPU memory...", flush=True)
                del input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    
                    # Clear GPU memory before running tests
                    print("[MINER] Clearing GPU memory before tests...", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                        print(f"[MINER] GPU memory before tests: {free_mem:.2f} GB free", flush=True)
                    
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
                            # Clear GPU memory after OOM
                            print("[MINER] OOM detected, clearing GPU memory...", flush=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                            
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
                
                print("[MINER] Running tests...", flush=True)
                # Location of the test script we created
                # It is likely in c:\quasar-kimi\tests\test_quasar_mining.py
                # We need to find absolute path
                test_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "test_quasar_mining.py")
                if not os.path.exists(test_script_path):
                     # Fallback to current dir / tests
                     test_script_path = os.path.abspath("tests/test_quasar_mining.py")

                result = subprocess.run(
                    [sys.executable, test_script_path],
                    cwd=repo_path, 
                    capture_output=True,
                    text=True
                )
                
                print(result.stdout)
                if result.returncode != 0:
                    print(f"[MINER] Tests Failed:\n{result.stderr}", flush=True)
                else:
                    print(f"[MINER] Tests Passed!", flush=True)
                    # Extract score
                    tps_match = re.search(r"QuasarAttention achieved: (\d+) tokens/sec", result.stdout)
                    if tps_match:
                        score = float(tps_match.group(1))
                        bt.logging.info(f"Iteration {iteration} score: {score}")
                        print(f"[MINER] Benchmark Score: {score} tokens/sec")
                        
                        # Submit
                        commit_hash = self.get_commit_hash(repo_path)
                        self.submit_to_validator(fork_url, commit_hash, score)
                        
            except Exception as e:
                print(f"[MINER] Error during build/test: {e}", flush=True)
                import traceback
                traceback.print_exc()

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
        
        # Load model
        print(" [MINER] Loading model...")
        miner.load_model()
        
        # Run miner
        try:
            miner.run_miner()
        except KeyboardInterrupt:
            print("\n [MINER] Shutting down...")
            miner.should_exit = True