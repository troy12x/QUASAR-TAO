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
from typing import Optional, List, Any, Dict
from pydantic import Field
import torch
import bittensor as bt
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from cryptography.hazmat.primitives.asymmetric import ed25519

# Add the parent directory to path so we can import quasar
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.miner import BaseMinerNeuron

class Miner(BaseMinerNeuron):
    """
    Long Context Miner with configurable model selection.
    
    This miner is designed to participate in the Long Bench evaluation by processing
    long context prompts and generating accurate responses.
    
    Supported Models:
    - silx-ai/Quasar-2M-Base (default, long context specialist)
    - moonshotai/Kimi-Linear-48B-A3B-Instruct (high performance)
    - Qwen/Qwen3-Next-80B-A3B-Thinking (advanced reasoning)
    """
    
    # Supported models list
    SUPPORTED_MODELS = [
        "silx-ai/Quasar-2M-Base",
        "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "Qwen/Qwen3-Next-80B-A3B-Thinking",
    ]

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # Track active tasks for cleanup
        self.active_tasks = {}
        
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
        
        # Model will be loaded in load_model() method (after axon starts)
        self.model = None
        self.tokenizer = None
        self.max_length = None
        self.model_loaded = False
        
        # Validator API URL for submitting answers
        self.validator_api_url = os.getenv("VALIDATOR_API_URL", "https://quasar-subnet.onrender.com")
        try:
            self.api_timeout = float(os.getenv("VALIDATOR_API_TIMEOUT", "30"))
        except Exception:
            self.api_timeout = 30.0
        self.api_debug = os.getenv("MINER_API_DEBUG", "0").strip() == "1"
        self.api_warmup = os.getenv("MINER_API_WARMUP", "1").strip() == "1"
        try:
            self.poll_interval = float(os.getenv("MINER_POLL_INTERVAL", "60"))
        except Exception:
            self.poll_interval = 60.0
        
    def _log_api_debug(self, message: str) -> None:
        if self.api_debug:
            msg = f"[API_DEBUG] {message}"
            bt.logging.info(msg)
            print(msg, flush=True)

    def _api_request(self, method: str, path: str, *, headers: Optional[dict] = None, json: Optional[dict] = None) -> requests.Response:
        url = f"{self.validator_api_url}{path}"
        t0 = time.perf_counter()
        self._log_api_debug(f"{method} {url} timeout={self.api_timeout}s")
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=self.api_timeout,
            )
            dt = time.perf_counter() - t0
            self._log_api_debug(
                f"{method} {path} -> {resp.status_code} elapsed={dt:.3f}s bytes={len(resp.content) if resp.content is not None else 'n/a'}"
            )
            return resp
        except Exception as e:
            dt = time.perf_counter() - t0
            self._log_api_debug(f"{method} {path} -> EXCEPTION elapsed={dt:.3f}s type={type(e).__name__} msg={e}")
            raise

    def _warmup_validator_api(self) -> None:
        if not self.api_warmup:
            return
        try:
            resp = self._api_request("GET", "/health")
            if self.api_debug:
                self._log_api_debug(f"/health ok={resp.ok}")
        except Exception:
            return

    def load_model(self):
        """Load the model and tokenizer. Call this after starting the axon."""
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
            
            # Get max length from config or model config
            self.max_length = getattr(self.config.miner, 'max_length', 
                                     getattr(self.model.config, 'max_position_embeddings', 32768))
            bt.logging.info(f"Miner max length set to: {self.max_length}")
            
            # Log device map for debugging
            if hasattr(self.model, 'hf_device_map'):
                print(f" Model Device Map: {self.model.hf_device_map}")
            else:
                print(f" Model Device: {self.model.device}")
                
            self.model.eval()
            print(f"\n [MINER] MY HOTKEY SS58: {self.wallet.hotkey.ss58_address} (COPY THIS FOR DASHBOARD)\n")
            bt.logging.info(f"Model loaded successfully: {self.model_name}")
            bt.logging.info(f"Validator API URL: {self.validator_api_url}")
            self.model_loaded = True
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            bt.logging.warning("Please ensure you have access/internet or specify a valid model.")
            raise e

    def _sign_message(self, message: str) -> str:
        """Sign a message with the wallet's private key."""
        # Bittensor Keypair uses .sign() directly
        signature = self.wallet.hotkey.sign(message.encode())
        return signature.hex()

    def _get_auth_headers(self) -> dict:
        """Get authentication headers for API requests."""
        # Sign the hotkey address (as expected by validator_api)
        signature = self._sign_message(self.wallet.hotkey.ss58_address)
        return {
            "Hotkey": self.wallet.hotkey.ss58_address,
            "Signature": signature,
        }

    def _submit_longcode_to_api(self, task_id: str, code: str, function_name: str = "solve", miner_uid: int = 0) -> bool:
        """Submit code to validator_api (for longcode tasks) - stores as pending for Docker evaluation."""
        try:
            payload = {
                "task_id": task_id,
                "code": code,
                "function_name": function_name,
                "miner_uid": miner_uid
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            # Submit to pending endpoint - validator will evaluate with Docker
            response = self._api_request("POST", "/submit_longcode_pending", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            bt.logging.info(f"✅ Submitted longcode to API (pending): task_id={task_id}, status={result.get('status')}")
            print(f"[MINER] Submitted longcode to API (pending): task_id={task_id}, status={result.get('status')}", flush=True)
            return True
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to submit longcode to API: {e}")
            print(f"[MINER] Submit longcode failed: {e}", flush=True)
            return False

    def _fetch_task_from_api(self):
        """Fetch a task from validator_api (longcode only)."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                headers = self._get_auth_headers()

                # Fetch longcode task
                response = self._api_request("GET", "/get_longcode_task", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    bt.logging.info(f"📥 Fetched longcode task: {data['id']}")
                    print(f"[MINER] Fetched longcode task: {data['id']}", flush=True)
                    # Debug: print task structure
                    if self.api_debug:
                        print(f"[MINER] Task keys: {list(data.keys())}", flush=True)
                        print(f"[MINER] template_code present: {'template_code' in data}, len={len(data.get('template_code', ''))}", flush=True)
                        print(f"[MINER] prompt present: {'prompt' in data}, len={len(data.get('prompt', ''))}", flush=True)
                    return data
            except Exception as e:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"⚠️ Failed to fetch task (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"[MINER] Fetch task failed (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                    time.sleep(5)  # Wait longer before retry
                else:
                    bt.logging.warning(f"⚠️ Failed to fetch task after {max_retries} attempts: {e}")
                    print(f"[MINER] Fetch task failed after {max_retries} attempts: {e}", flush=True)
        return None

    def _fetch_task_stats(self):
        """Fetch task statistics from validator_api for this miner."""
        try:
            headers = self._get_auth_headers()
            response = self._api_request("GET", f"/get_task_stats?hotkey={self.wallet.hotkey.ss58_address}", headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"[MINER] Failed to fetch task stats: {e}", flush=True)
        return None

    def _run_mining_loop(self):
        """Main mining loop - fetch tasks, generate code, submit to API."""
        print(
            f"[MINER] API config: timeout={self.api_timeout}s debug={int(self.api_debug)} warmup={int(self.api_warmup)} poll_interval={self.poll_interval}s url={self.validator_api_url}",
            flush=True,
        )
        self._warmup_validator_api()

        # Track processed task IDs
        self.processed_tasks = []

        while not self.should_exit:
            # Print stats every 10 tasks
            if self.tasks_processed > 0 and self.tasks_processed % 10 == 0:
                elapsed_time = time.time() - self.start_time
                success_rate = (self.tasks_succeeded / self.tasks_processed * 100) if self.tasks_processed > 0 else 0
                print(f"\n[MINER] ========== STATS ==========", flush=True)
                print(f"[MINER] Tasks processed: {self.tasks_processed}", flush=True)
                print(f"[MINER] Tasks succeeded: {self.tasks_succeeded}", flush=True)
                print(f"[MINER] Tasks failed: {self.tasks_failed}", flush=True)
                print(f"[MINER] Success rate: {success_rate:.1f}%", flush=True)
                print(f"[MINER] Uptime: {elapsed_time/60:.1f} minutes", flush=True)
                print(f"[MINER] Avg tasks/min: {self.tasks_processed/(elapsed_time/60):.2f}", flush=True)
                print(f"[MINER] ============================\n", flush=True)

            # Fetch and display task stats
            task_stats = self._fetch_task_stats()
            if task_stats:
                print(f"[MINER] System Stats: Total={task_stats.get('total_tasks_in_dataset', 0)} | Completed={task_stats.get('completed_tasks', 0)} | Pending={task_stats.get('pending_tasks', 0)} | Active={task_stats.get('active_assignments', 0)}", flush=True)

            # Fetch task from API
            task = self._fetch_task_from_api()
            if not task:
                print(f"[MINER] No task fetched. Sleeping {self.poll_interval}s...", flush=True)
                time.sleep(self.poll_interval)
                continue

            template_code = task.get("template_code")
            prompt = task.get("prompt")
            if not template_code or not prompt:
                print(f"[MINER] Invalid longcode task payload for task_id={task.get('id')}", flush=True)
                self.tasks_failed += 1
                time.sleep(5)
                continue

            self.tasks_processed += 1
            task_id = task['id']
            self.processed_tasks.append(task_id)

            bt.logging.info(f"🧩 Generating code for task {task_id} (task #{self.tasks_processed})...")
            print(f"[MINER] ========== TASK #{self.tasks_processed} ==========", flush=True)
            print(f"[MINER] Task ID: {task_id}", flush=True)
            print(f"[MINER] Generating code...", flush=True)

            try:
                success = self._process_longcode_task(task_id=task_id, template_code=template_code, prompt=prompt)
                if success:
                    self.tasks_succeeded += 1
                    print(f"[MINER] ✅ Task #{self.tasks_processed} completed successfully", flush=True)
                else:
                    self.tasks_failed += 1
                    print(f"[MINER] ❌ Task #{self.tasks_processed} failed", flush=True)
            except Exception as e:
                self.tasks_failed += 1
                bt.logging.error(f"❌ Error processing task: {e}")
                print(f"[MINER] ❌ Task #{self.tasks_processed} error: {e}", flush=True)

            print(f"[MINER] =====================================", flush=True)
            # Wait before next task
            time.sleep(60)

    def _extract_function_name(self, template_code: str) -> str:
        for line in template_code.split("\n"):
            line = line.strip()
            if line.startswith("def "):
                return line.split("(")[0].replace("def ", "").strip()
        return "solve"

    def _process_longcode_task(self, *, task_id: str, template_code: str, prompt: str) -> bool:
        function_name = self._extract_function_name(template_code)

        messages = [
            {
                "role": "system",
                "content": """You are an expert Python programmer.

Return ONLY Python code.

Rules:
- Implement the function(s) in the provided template.
- Do not add any top-level code execution.
- Do not import any modules.
""",
            },
            {
                "role": "user",
                "content": f"""Template:
{template_code}

Requirements:
{prompt}
""",
            },
        ]

        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [text_input],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_length = model_inputs.input_ids.shape[1]
        print(f"[MINER] Input length: {input_length} tokens", flush=True)
        print(f"[MINER] Starting generation (max_new_tokens=1024)...", flush=True)

        import time
        start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        end_time = time.time()
        generation_time = end_time - start_time

        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_length = generated_ids[0].shape[0]
        tokens_per_sec = output_length / generation_time if generation_time > 0 else 0

        print(f"[MINER] Generated {output_length} tokens in {generation_time:.2f}s", flush=True)
        print(f"[MINER] Speed: {tokens_per_sec:.2f} tokens/sec", flush=True)
        print(f"[MINER] Device: {self.device}", flush=True)

        code = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        preview = code.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"[MINER] Generated code len={len(code)} preview={preview}", flush=True)

        # Submit to API
        print(f"[MINER] Submitting to API...", flush=True)
        submission_success = self._submit_longcode_to_api(task_id, code, function_name=function_name, miner_uid=self.uid)

        if submission_success:
            print(f"[MINER] ✅ Task {task_id} submitted successfully", flush=True)
        else:
            print(f"[MINER] ❌ Task {task_id} submission failed", flush=True)

        del model_inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return submission_success

    async def forward(
        self, synapse: quasar.protocol.BenchmarkEvaluationSynapse
    ) -> quasar.protocol.BenchmarkEvaluationSynapse:
        """Not used - miner uses API polling instead."""
        synapse.response = "Miner uses API polling. Please use validator_api."
        synapse.processing_time = 0.0
        return synapse

    async def blacklist(
        self, synapse: quasar.protocol.BenchmarkEvaluationSynapse
    ) -> typing.Tuple[bool, str]:
        
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized hotkeys.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"
            
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: quasar.protocol.BenchmarkEvaluationSynapse) -> float:
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  
        prirority = float(
            self.metagraph.S[caller_uid]
        ) 
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {prirority}"
        )
        return prirority

    async def forward_start_round(self, synapse: quasar.protocol.StartRoundSynapse) -> quasar.protocol.StartRoundSynapse:
        """Handle handshake from validator to check liveness."""
        
        # Set miner status
        synapse.is_ready = True
        synapse.available_capacity = 1  # Can handle 1 task at a time
        synapse.miner_version = "quasar-v1.0"
        synapse.error_message = None
        
        bt.logging.info(f"Handshake received: round_id={synapse.round_id}")
        print(f"  [MINER] Handshake: round_id={synapse.round_id}")
        
        return synapse

    async def forward_feedback(self, synapse: quasar.protocol.TaskFeedbackSynapse) -> quasar.protocol.TaskFeedbackSynapse:
        """Handle feedback from validator after evaluation."""
        
        # Log feedback
        bt.logging.info(
            f"Feedback received: task_id={synapse.task_id} "
            f"score={synapse.score:.4f} "
            f"latency={synapse.latency_seconds:.2f}s"
        )
        print(f"  [MINER] Feedback: task_id={synapse.task_id} score={synapse.score:.4f} latency={synapse.latency_seconds:.2f}s")
        
        if synapse.feedback_text:
            bt.logging.info(f"  Feedback: {synapse.feedback_text}")
        
        if synapse.suggestions:
            bt.logging.info(f"  Suggestions: {synapse.suggestions}")
        
        # Acknowledge receipt
        synapse.acknowledged = True
        
        return synapse

    async def forward_cleanup(self, synapse: quasar.protocol.TaskCleanupSynapse) -> quasar.protocol.TaskCleanupSynapse:
        """Handle cleanup signal from validator."""
        
        task_id = synapse.task_id
        
        try:
            # Clean up resources for this task
            if task_id in self.active_tasks:
                # Remove from active tasks
                del self.active_tasks[task_id]
                
                # Clean up any temporary files
                temp_dir = f"/tmp/quasar_task_{task_id}"
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                    bt.logging.info(f"  Cleanup: Removed temp directory {temp_dir}")
                
                # Clean up GPU memory if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                synapse.acknowledged = True
                synapse.cleanup_ok = True
                synapse.error_message = None
                
                bt.logging.info(f"Cleanup completed for task {task_id}")
                print(f"  [MINER] Cleanup: task_id={task_id} OK")
            else:
                synapse.acknowledged = True
                synapse.cleanup_ok = True
                synapse.error_message = "Task not found in active tasks"
                bt.logging.info(f"Cleanup: task {task_id} not in active tasks")
                
        except Exception as e:
            synapse.acknowledged = True
            synapse.cleanup_ok = False
            synapse.error_message = str(e)
            
            bt.logging.error(f"Cleanup failed for task {task_id}: {e}")
            print(f"  [MINER] Cleanup failed: {task_id} - {e}")
        
        return synapse

    def _log_memory_usage(self, stage: str):
        """Log current RAM and GPU memory usage."""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        bt.logging.info(f" [MEMORY] {stage} | RAM: {ram_usage:.2f} MB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_reserved(i) / (1024 * 1024)
                bt.logging.info(f" [MEMORY] {stage} | GPU {i}: {mem:.2f} MB")

    # ========== QUASAR KERNEL OPTIMIZATION ==========

    def clone_quasar_repo(self, repo_path: str = None) -> str:
        """Clone the flash-linear-attention repository."""
        if repo_path is None:
            repo_path = os.path.join(tempfile.gettempdir(), "flash-linear-attention")
        
        repo_url = "https://github.com/troy12x/flash-linear-attention.git"
        
        bt.logging.info(f"Cloning Quasar repository from {repo_url} to {repo_path}")
        print(f"[QUASAR] Cloning repository to {repo_path}...", flush=True)
        
        # Remove existing repo if present
        if os.path.exists(repo_path):
            bt.logging.info(f"Removing existing repository at {repo_path}")
            shutil.rmtree(repo_path)
        
        # Clone repo
        try:
            subprocess.run(
                ["git", "clone", repo_url, repo_path],
                check=True,
                capture_output=True,
                text=True
            )
            bt.logging.info(f" Repository cloned successfully to {repo_path}")
            return repo_path
        except subprocess.CalledProcessError as e:
            bt.logging.error(f" Failed to clone repository: {e.stderr}")
            raise

    def read_quasar_files(self, repo_path: str) -> Dict[str, str]:
        """Read all Quasar attention files."""
        quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
        
        files_to_read = [
            "chunk.py",
            "chunk_intra_token_parallel.py",
            "forward_substitution.py",
            "fused_recurrent.py",
            "gate.py",
            "__init__.py"
        ]
        
        codebase = {}
        for filename in files_to_read:
            file_path = os.path.join(quasar_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    codebase[filename] = f.read()
                bt.logging.info(f"Read {filename} ({len(codebase[filename])} chars)")
            else:
                bt.logging.warning(f"File not found: {file_path}")
        
        return codebase

    def analyze_and_optimize_with_ai(self, codebase: Dict[str, str], task_description: str) -> str:
        """Use AI to analyze code and generate optimizations."""
        
        # Prepare context for AI
        context = f"""You are a GPU kernel optimization expert specializing in Triton kernels for Quasar attention.

Task: {task_description}

Target Files (ONLY modify these files in fla/ops/quasar/):
{chr(10).join([f"### {name} ###\n{code}" for name, code in codebase.items()])}

Instructions:
1. Analyze the current implementation
2. Identify performance bottlenecks
3. Generate optimized kernel code
4. Focus on tokens/sec and memory usage for sequences 4K-100K
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
        
        bt.logging.info("Sending code to AI for optimization...")
        print(f"[QUASAR] Sending {len(codebase)} files to AI for optimization...", flush=True)
        
        # Prepare messages for LLM
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
        
        # Tokenize input
        text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(
            [text_input],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        input_length = model_inputs.input_ids.shape[1]
        print(f"[QUASAR] Input length: {input_length} tokens", flush=True)
        
        # Generate optimized code
        import time
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=16000,  # Allow long responses for multiple files
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_length = generated_ids[0].shape[0]
        tokens_per_sec = output_length / generation_time if generation_time > 0 else 0
        
        print(f"[QUASAR] Generated {output_length} tokens in {generation_time:.2f}s", flush=True)
        print(f"[QUASAR] Speed: {tokens_per_sec:.2f} tokens/sec", flush=True)
        
        ai_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        del model_inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return ai_response

    def parse_ai_response(self, ai_response: str) -> Dict[str, str]:
        """Parse AI response to extract optimized file contents."""
        files = {}
        
        # Split by ### filename ### pattern
        parts = ai_response.split("### ")
        
        current_file = None
        current_content = []
        
        for part in parts:
            if not part.strip():
                continue
            
            lines = part.split("\n", 1)
            if len(lines) >= 1:
                # Extract filename
                filename = lines[0].strip()
                if filename.endswith(" ###"):
                    filename = filename[:-4].strip()
                
                # Extract content
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
            
            # Verify file is in quasar directory
            if not file_path.startswith(os.path.join(repo_path, "fla/ops/quasar")):
                bt.logging.warning(f"Skipping file outside quasar directory: {file_path}")
                continue
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            modified_files.append(filename)
            bt.logging.info(f"Wrote optimized file: {filename}")
        
        return modified_files

    def generate_diff(self, repo_path: str) -> str:
        """Generate git diff of changes."""
        result = subprocess.run(
            ["git", "diff", "fla/ops/quasar/"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        diff = result.stdout
        bt.logging.info(f"Generated diff: {len(diff)} characters")
        
        return diff

    def get_base_commit(self, repo_path: str) -> str:
        """Get the base commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        return result.stdout.strip()

    def submit_quasar_diff(self, diff: str, description: str) -> bool:
        """Submit diff to QuasarSubnet API."""
        try:
            payload = {
                "miner_hotkey": self.wallet.hotkey.ss58_address,
                "diff": diff,
                "base_commit": "latest",
                "description": description,
                "signature": self._sign_message(diff)
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            # Submit to QuasarSubnet API
            response = self._api_request("POST", "/upload_submission", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            bt.logging.info(f" Submitted Quasar diff: {result.get('submission_id')}")
            print(f"[QUASAR]  Submitted diff: {result.get('submission_id')}", flush=True)
            return True
        except Exception as e:
            bt.logging.warning(f" Failed to submit Quasar diff: {e}")
            print(f"[QUASAR]  Submit diff failed: {e}", flush=True)
            return False

    def run_quasar_optimization(self) -> bool:
        """Run the complete Quasar optimization workflow."""
        try:
            print(f"\n[QUASAR] ========== Starting Quasar Optimization ==========", flush=True)
            
            # Step 1: Clone repository
            repo_path = self.clone_quasar_repo()
            
            # Step 2: Read Quasar files
            codebase = self.read_quasar_files(repo_path)
            print(f"[QUASAR] Read {len(codebase)} files", flush=True)
            
            # Step 3: Use AI to optimize
            task_description = "Optimize Quasar attention kernels for maximum tokens/sec and minimum memory usage on sequences 4K-100K. Focus on Triton kernel optimizations."
            ai_response = self.analyze_and_optimize_with_ai(codebase, task_description)
            
            # Step 4: Parse AI response
            optimized_files = self.parse_ai_response(ai_response)
            print(f"[QUASAR] Parsed {len(optimized_files)} optimized files", flush=True)
            
            if not optimized_files:
                bt.logging.error("No optimized files parsed from AI response")
                return False
            
            # Step 5: Write optimized files
            modified_files = self.write_optimized_files(repo_path, optimized_files)
            print(f"[QUASAR] Modified {len(modified_files)} files", flush=True)
            
            # Step 6: Generate diff
            diff = self.generate_diff(repo_path)
            print(f"[QUASAR] Generated diff: {len(diff)} characters", flush=True)
            
            # Step 7: Submit to subnet
            description = f"AI-optimized Quasar attention kernels. Modified files: {', '.join(modified_files)}"
            success = self.submit_quasar_diff(diff, description)
            
            if success:
                print(f"[QUASAR]  Optimization completed successfully", flush=True)
                self.tasks_succeeded += 1
            else:
                print(f"[QUASAR]  Optimization submission failed", flush=True)
                self.tasks_failed += 1
            
            # Cleanup
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
                bt.logging.info(f"Cleaned up repository at {repo_path}")
            
            return success
            
        except Exception as e:
            bt.logging.error(f" Error in Quasar optimization: {e}")
            print(f"[QUASAR]  Error: {e}", flush=True)
            self.tasks_failed += 1
            return False

# This is the main function, which runs the miner.
if __name__ == "__main__":
    import argparse
    import threading

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="QuasarSubnet Miner")
    parser.add_argument("--mode", type=str, default="longcode", 
                       choices=["longcode", "quasar"],
                       help="Mining mode: longcode (default) or quasar")
    parser.add_argument("--quasar-interval", type=float, default=3600,
                       help="Interval between Quasar optimizations in seconds (default: 3600)")
    args = parser.parse_args()

    # Note: Bittensor's config system will automatically parse --miner.model_name
    # and --miner.league from command line arguments
    with Miner() as miner:
        # Load model
        print(" [MINER] Loading model...")
        miner.load_model()
        
        # Run based on mode
        if args.mode == "quasar":
            print(f" [MINER] Starting Quasar optimization mode (interval: {args.quasar_interval}s)...")
            print(f" [MINER] Press Ctrl+C to stop\n")
            
            try:
                while not miner.should_exit:
                    # Run Quasar optimization
                    success = miner.run_quasar_optimization()
                    
                    # Wait before next optimization
                    print(f"\n[QUASAR] Waiting {args.quasar_interval}s before next optimization...", flush=True)
                    time.sleep(args.quasar_interval)
            except KeyboardInterrupt:
                print("\n [MINER] Shutting down...")
                miner.should_exit = True
        else:
            # Start mining loop (polls API for tasks)
            print(" [MINER] Starting mining loop (polling validator API)...")
            print(" [MINER] Press Ctrl+C to stop\n")
            
            try:
                miner._run_mining_loop()
            except KeyboardInterrupt:
                print("\n [MINER] Shutting down...")
                miner.should_exit = True