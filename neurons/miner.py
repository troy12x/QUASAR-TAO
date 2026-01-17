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
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.model_name = getattr(self.config.miner, 'model_name', "silx-ai/Quasar-2M-Base")
        
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

    def _api_request(self, method: str, path: str, *, headers: Optional[dict] = None, json: Optional[dict] = None) -> requests.Response:
        """Make API request to validator."""
        url = f"{self.validator_api_url}{path}"
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=30,
            )
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
        
        return local_path

    def read_target_files(self, repo_path: str) -> Dict[str, str]:
        """Read the target Quasar files."""
        quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
        codebase = {}
        
        for filename in self.TARGET_FILES:
            file_path = os.path.join(quasar_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    codebase[filename] = f.read()
                bt.logging.info(f"Read {filename} ({len(codebase[filename])} chars)")
            else:
                bt.logging.warning(f"File not found: {file_path}")
        
        return codebase

    def run_agent_optimization(self, codebase: Dict[str, str], iteration: int) -> Dict[str, str]:
        """Run AI agent to optimize the code."""
        context = f"""You are a GPU kernel optimization expert specializing in Triton kernels for Quasar attention.

Iteration: {iteration}
Target sequence length: {self.target_sequence_length}

Target Files (ONLY modify these files in fla/ops/quasar/):
{chr(10).join([f"### {name} ###\n{code[:2000]}..." if len(code) > 2000 else f"### {name} ###\n{code}" for name, code in codebase.items()])}

Instructions:
1. Analyze the current implementation
2. Identify performance bottlenecks for sequence length {self.target_sequence_length}
3. Generate optimized kernel code
4. Focus on tokens/sec and memory usage
5. Keep changes ONLY in fla/ops/quasar/ files
6. Maintain numerical accuracy (rtol < 1e-3)
7. Return the complete optimized file contents

Response Format:
### filename.py ###
<optimized code>
### filename.py ###
<optimized code>
...
"""
        
        bt.logging.info(f"Running agent optimization iteration {iteration}...")
        print(f"[AGENT] Iteration {iteration}: Sending code to AI for optimization...", flush=True)
        
        messages = [
            {
                "role": "system",
                "content": "You are a GPU kernel optimization expert specializing in Triton kernels. Return ONLY the optimized file contents in the specified format."
            },
            {
                "role": "user",
                "content": context
            }
        ]
        
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [text_input],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        input_length = model_inputs.input_ids.shape[1]
        print(f"[AGENT] Input length: {input_length} tokens", flush=True)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=16000,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        ai_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        del model_inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.parse_agent_response(ai_response)

    def parse_agent_response(self, ai_response: str) -> Dict[str, str]:
        """Parse agent response to extract optimized file contents."""
        files = {}
        parts = ai_response.split("### ")
        
        for part in parts:
            if not part.strip():
                continue
            
            lines = part.split("\n", 1)
            if len(lines) >= 1:
                filename = lines[0].strip()
                if filename.endswith(" ###"):
                    filename = filename[:-4].strip()
                
                content = lines[1] if len(lines) > 1 else ""
                
                if filename and content:
                    files[filename] = content
                    bt.logging.info(f"Parsed optimized file: {filename} ({len(content)} chars)")
        
        return files

    def write_optimized_files(self, repo_path: str, optimized_files: Dict[str, str]) -> List[str]:
        """Write optimized files to repository."""
        modified_files = []
        
        for filename, content in optimized_files.items():
            file_path = os.path.join(repo_path, "fla/ops/quasar", filename)
            
            if not file_path.startswith(os.path.join(repo_path, "fla/ops/quasar")):
                bt.logging.warning(f"Skipping file outside quasar directory: {file_path}")
                continue
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            modified_files.append(filename)
            bt.logging.info(f"Wrote optimized file: {filename}")
        
        return modified_files

    def run_tests(self, repo_path: str, sequence_length: int = None) -> float:
        """Run tests on the optimized code."""
        if sequence_length is None:
            sequence_length = self.target_sequence_length
        
        # Create temporary test script with target sequence length
        temp_test_script = os.path.join(repo_path, "test_temp.py")
        with open(temp_test_script, 'w') as f:
            f.write(f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fla.layers.quasar import QuasarAttention

def test_quasar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1
    seq_len = {sequence_length}
    hidden_size = 512
    head_dim = 64
    num_heads = 8
    
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Warmup
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    num_runs = 10
    start = time.time()
    
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed
    
    print(f"RESULT: {{tokens_per_sec:.2f}}")
    return tokens_per_sec

if __name__ == "__main__":
    tps = test_quasar()
    print(f"Tokens/sec: {{tps:.2f}}")
"""
        
        # Run test
        bt.logging.info(f"Running tests for sequence length {sequence_length}...")
        print(f"[TEST] Running tests for sequence length {sequence_length}...", flush=True)
        
        try:
            result = subprocess.run(
                [sys.executable, temp_test_script],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            # Extract tokens/sec from output
            for line in output.split('\n'):
                if "RESULT:" in line:
                    tokens_per_sec = float(line.split("RESULT:")[1].strip())
                    print(f"[TEST] Tokens/sec: {tokens_per_sec:.2f}", flush=True)
                    return tokens_per_sec
            
            bt.logging.warning(f"Could not parse test results: {output}")
            return 0.0
            
        except subprocess.TimeoutExpired:
            bt.logging.error("Test timed out (300s)")
            return 0.0
        except Exception as e:
            bt.logging.error(f"Test failed: {e}")
            return 0.0
        finally:
            # Clean up temp test script
            if os.path.exists(temp_test_script):
                os.remove(temp_test_script)

    def commit_and_push(self, repo_path: str, message: str) -> bool:
        """Commit and push changes to fork."""
        try:
            bt.logging.info(f"Committing changes: {message}")
            print(f"[GIT] Committing changes...", flush=True)
            
            subprocess.run(
                ["git", "add", "fla/ops/quasar/"],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            
            subprocess.run(
                ["git", "push"],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
            
            print(f"[GIT] Changes pushed successfully", flush=True)
            return True
            
        except subprocess.CalledProcessError as e:
            bt.logging.error(f"Git operation failed: {e.stderr}")
            return False

    def get_commit_hash(self, repo_path: str) -> str:
        """Get the current commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def submit_to_validator(self, fork_url: str, commit_hash: str, performance: float) -> bool:
        """Submit optimization results to validator API."""
        try:
            payload = {
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "fork_url": fork_url,
                "commit_hash": commit_hash,
                "target_sequence_length": self.target_sequence_length,
                "tokens_per_sec": performance,
                "signature": self._sign_message(f"{fork_url}{commit_hash}{performance}")
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            bt.logging.info(f"Submitting to validator: {performance:.2f} tokens/sec")
            print(f"[API] Submitting to validator: {performance:.2f} tokens/sec", flush=True)
            
            response = self._api_request("POST", "/submit_optimization", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            bt.logging.info(f"Submission successful: {result.get('submission_id')}")
            print(f"[API] Submission successful: {result.get('submission_id')}", flush=True)
            return True
            
        except Exception as e:
            bt.logging.warning(f"Submission failed: {e}")
            print(f"[API] Submission failed: {e}", flush=True)
            return False

    def run_optimization_loop(self, fork_url: str, repo_path: str):
        """Run the continuous optimization loop with agents."""
        bt.logging.info("Starting optimization loop...")
        print(f"\n[LOOP] Starting optimization loop (iterations: {self.agent_iterations})...", flush=True)
        
        best_performance = 0.0
        best_commit = None
        
        for iteration in range(self.agent_iterations):
            if self.should_exit:
                break
            
            print(f"\n[LOOP] ========== Iteration {iteration + 1}/{self.agent_iterations} ==========", flush=True)
            
            # Read current code
            codebase = self.read_target_files(repo_path)
            
            # Run agent optimization
            try:
                optimized_files = self.run_agent_optimization(codebase, iteration + 1)
                
                if not optimized_files:
                    print(f"[LOOP] No optimizations generated, skipping...", flush=True)
                    continue
                
                # Write optimized files
                modified_files = self.write_optimized_files(repo_path, optimized_files)
                print(f"[LOOP] Modified {len(modified_files)} files", flush=True)
                
                # Run tests
                performance = self.run_tests(repo_path, self.target_sequence_length)
                
                if performance <= 0:
                    print(f"[LOOP] Tests failed, reverting changes...", flush=True)
                    subprocess.run(["git", "checkout", "--", "fla/ops/quasar/"], cwd=repo_path, capture_output=True)
                    continue
                
                print(f"[LOOP] Performance: {performance:.2f} tokens/sec", flush=True)
                
                # Check if this is the best performance
                if performance > best_performance:
                    best_performance = performance
                    commit_message = f"Optimization iteration {iteration + 1}: {performance:.2f} tokens/sec @ seq_len={self.target_sequence_length}"
                    
                    if self.commit_and_push(repo_path, commit_message):
                        best_commit = self.get_commit_hash(repo_path)
                        print(f"[LOOP] New best performance! Commit: {best_commit}", flush=True)
                        
                        # Submit to validator
                        self.submit_to_validator(fork_url, best_commit, best_performance)
                else:
                    print(f"[LOOP] Performance not improved, reverting changes...", flush=True)
                    subprocess.run(["git", "checkout", "--", "fla/ops/quasar/"], cwd=repo_path, capture_output=True)
                
                self.optimization_iterations += 1
                
            except Exception as e:
                bt.logging.error(f"Error in iteration {iteration + 1}: {e}")
                print(f"[LOOP] Error: {e}", flush=True)
                subprocess.run(["git", "checkout", "--", "fla/ops/quasar/"], cwd=repo_path, capture_output=True)
            
            # Wait before next iteration
            if iteration < self.agent_iterations - 1:
                print(f"[LOOP] Waiting {self.optimization_interval}s before next iteration...", flush=True)
                time.sleep(self.optimization_interval)
        
        print(f"\n[LOOP] Optimization loop completed", flush=True)
        print(f"[LOOP] Best performance: {best_performance:.2f} tokens/sec", flush=True)
        print(f"[LOOP] Best commit: {best_commit}", flush=True)

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
    
    parser = argparse.ArgumentParser(description="QuasarSubnet Miner")
    parser.add_argument("--agent-iterations", type=int, default=100,
                       help="Number of agent optimization iterations (default: 100)")
    parser.add_argument("--target-seq-len", type=int, default=100000,
                       help="Target sequence length (default: 100000)")
    parser.add_argument("--optimization-interval", type=float, default=300,
                       help="Interval between iterations in seconds (default: 300)")
    args = parser.parse_args()
    
    with Miner() as miner:
        # Override config with command line args
        miner.agent_iterations = args.agent_iterations
        miner.target_sequence_length = args.target_seq_len
        miner.optimization_interval = args.optimization_interval
        
        # Load model
        print(" [MINER] Loading model...")
        miner.load_model()
        
        # Run miner
        try:
            miner.run_miner()
        except KeyboardInterrupt:
            print("\n [MINER] Shutting down...")
            miner.should_exit = True