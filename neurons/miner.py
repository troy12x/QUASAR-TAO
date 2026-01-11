# The MIT License (MIT)
# Copyright 2026 SILX INC

import os
import time
import typing
import requests
import hashlib
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

    def _submit_answer_to_api(self, task_id: str, answer: str, miner_uid: int = 0) -> bool:
        """Submit answer to validator_api."""
        try:
            payload = {
                "task_id": task_id,
                "answer": answer,
                "miner_uid": miner_uid
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            response = requests.post(
                f"{self.validator_api_url}/receive_answers",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            bt.logging.info(f"✅ Submitted answer to API: task_id={task_id}, status={result.get('status')}")
            return True
        except Exception as e:
            bt.logging.warning(f"⚠️ Failed to submit answer to API: {e}")
            return False

    def _fetch_task_from_api(self) -> Optional[dict]:
        """Fetch a task from validator_api."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                headers = self._get_auth_headers()
                
                response = requests.get(
                    f"{self.validator_api_url}/get_task",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                bt.logging.info(f"📥 Fetched task: {data['id']}")
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    bt.logging.warning(f"⚠️ Failed to fetch task (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(5)  # Wait longer before retry
                else:
                    bt.logging.warning(f"⚠️ Failed to fetch task after {max_retries} attempts: {e}")
        return None

    def _run_mining_loop(self):
        """Main mining loop - fetch tasks, generate answers, submit to API."""
        while not self.should_exit:
            # Fetch task from API
            task = self._fetch_task_from_api()
            if not task:
                time.sleep(60)  # Wait before retry
                continue
            
            # Generate answer
            bt.logging.info(f"� Generating answer for task {task['id']}...")
            start_time = time.time()
            
            try:
                # Prepare input
                context = task.get('context', '')
                prompt = task.get('prompt', '')
                
                messages = [
                    {"role": "system", "content": """You are a financial document analysis assistant.

TASK:
Read the provided financial document text and answer the question accurately.

IMPORTANT:
- Extract the answer directly from the provided document text
- Be precise with numbers, dates, and financial terms
- Show your reasoning step-by-step

OUTPUT FORMAT:
You MUST format your final answer as: "Therefore, the answer is (insert answer here)."
Example: "Therefore, the answer is $100,000."
The formatted answer MUST be the final sentence of your response."""},
                    {"role": "user", "content": f"""Context:
{context}

Question: {prompt}

Please show your reasoning and provide the answer in the required format."""}
                ]
                
                text_input = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer(
                    [text_input], 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                elapsed = time.time() - start_time
                bt.logging.info(f"✅ Generated answer in {elapsed:.2f}s (length: {len(response)} chars)")
                
                # Submit to API
                self._submit_answer_to_api(task['id'], response, miner_uid=self.uid)
                
                # Cleanup
                del model_inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                bt.logging.error(f"❌ Error processing task: {e}")
            
            # Wait before next task
            time.sleep(60)

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
        bt.logging.info(f"💾 [MEMORY] {stage} | RAM: {ram_usage:.2f} MB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_reserved(i) / (1024 * 1024)
                bt.logging.info(f" [MEMORY] {stage} | GPU {i}: {mem:.2f} MB")

# This is the main function, which runs the miner.
if __name__ == "__main__":
    import argparse
    import threading

    # Note: Bittensor's config system will automatically parse --miner.model_name
    # and --miner.league from command line arguments
    with Miner() as miner:
        # Load model
        print(" [MINER] Loading model...")
        miner.load_model()
        
        # Start mining loop (polls API for tasks)
        print(" [MINER] Starting mining loop (polling validator API)...")
        print(" [MINER] Press Ctrl+C to stop\n")
        
        try:
            miner._run_mining_loop()
        except KeyboardInterrupt:
            print("\n [MINER] Shutting down...")
            miner.should_exit = True