# The MIT License (MIT)
# Copyright © 2026 SILX INC

import os
import time
import typing
from typing import Optional, List, Any, Dict
from pydantic import Field
import torch
import bittensor as bt
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

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
            self.model_loaded = True
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            bt.logging.warning("Please ensure you have access/internet or specify a valid model.")
            raise e

    async def forward(
        self, synapse: quasar.protocol.BenchmarkEvaluationSynapse
    ) -> quasar.protocol.BenchmarkEvaluationSynapse:
        """
        Processes the incoming synapse by generating a response to the prompt.
        """
        # Wait for model to be loaded
        max_wait = 300  # 5 minutes max wait
        waited = 0
        while not self.model_loaded and waited < max_wait:
            time.sleep(1)
            waited += 1
        
        if not self.model_loaded:
            synapse.response = "Error: Model not loaded yet. Please try again."
            synapse.processing_time = 0.0
            return synapse
        
        start_time = time.time()
        
        try:
            # Get context and prompt from synapse
            context = getattr(synapse, 'context', '') or ''
            prompt = getattr(synapse, 'prompt', '') or ''
            
            # If context/prompt missing, return error
            if not context or not prompt:
                synapse.response = "Error: Missing context or prompt in synapse."
                synapse.processing_time = time.time() - start_time
                return synapse
            
            print(f"\n[MINER] 📝 Received Task: {synapse.task_id}")
            print(f"[MINER] Prompt: {prompt}")
            print(f"[MINER] Context Length: {len(context)} characters")
            bt.logging.info(f"📨 Received request. Context len: {len(context)} chars")

            # formatting input for Qwen
            # Qwen instruct format usually:  system... user...
            # We can use apply_chat_template if available, or raw concatenation.
            
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
            
            self._log_memory_usage("Before Tokenization")
            
            model_inputs = self.tokenizer(
                [text_input], 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            self._log_memory_usage("After Tokenization / Before Generate")
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=512, # High capacity yet stable for 128k context
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # print(f"[MINER]  Generated Response: {response}")  # Commented to reduce log spam
            bt.logging.info(f" [MINER] Generated response length: {len(response)} chars")
            
            # Store task for cleanup later
            self.active_tasks[synapse.task_id] = {
                "start_time": time.time(),
                "context": context,
                "prompt": prompt
            }
            
            synapse.response = response
            synapse.processing_time = time.time() - start_time
            
            # Precise memory cleanup
            del model_inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Fill other metadata if required by protocol
            if hasattr(synapse, 'quasar_model_architecture'):
                synapse.quasar_model_architecture = self.model_name
            if hasattr(synapse, 'quasar_model_configuration'):
                synapse.quasar_model_configuration = {"max_length": self.max_length}
                
        except Exception as e:
            bt.logging.error(f"❌ Error during generation: {e}")
            synapse.response = f"Error: {str(e)}"
            synapse.processing_time = time.time() - start_time

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
        # Start the miner's axon in a separate thread
        def run_miner():
            miner.run()
        
        miner_thread = threading.Thread(target=run_miner, daemon=False)
        miner_thread.start()
        
        # Wait for axon to start serving
        time.sleep(3)
        
        # Load model in background while axon is serving
        print(" [MINER] Loading model in background (axon is serving)...")
        miner.load_model()
        
        # Keep miner running indefinitely
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n [MINER] Shutting down...")
            miner.should_exit = True
            miner_thread.join(timeout=5)