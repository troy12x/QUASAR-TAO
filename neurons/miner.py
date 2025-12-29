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
        
        bt.logging.info("Initializing Long Context Miner...")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bt.logging.info(f"Using device: {self.device}")
        
        # API Configuration
        self.api_root = getattr(self.config, 'api_root', "http://localhost:8000")
        bt.logging.info(f"🌐 Validator API Root: {self.api_root}")
        
        # Get model name from config or use default
        self.model_name = getattr(self.config.miner, 'model_name', "silx-ai/Quasar-2M-Base")
        
        # Validate model selection
        if self.model_name not in self.SUPPORTED_MODELS:
            bt.logging.warning(f"Model {self.model_name} not in supported list. Proceeding anyway...")
            bt.logging.info(f"Supported models: {', '.join(self.SUPPORTED_MODELS)}")
        
        try:
            print(f"🔄 Loading tokenizer for {self.model_name}...")
            bt.logging.info(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            print(f"🔄 Loading model weights for {self.model_name}... (this can take several minutes)")
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
                print(f"🗺️ Model Device Map: {self.model.hf_device_map}")
            else:
                print(f"🗺️ Model Device: {self.model.device}")
                
            self.model.eval()
            bt.logging.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            bt.logging.warning("Please ensure you have access/internet or specify a valid model.")
            raise e

    async def forward(
        self, synapse: quasar.protocol.BenchmarkEvaluationSynapse
    ) -> quasar.protocol.BenchmarkEvaluationSynapse:
        print(f"\n📨 [MINER] >>> RECEIVED SYNAPSE: {synapse.task_id} <<<")
        """
        Processes the incoming 'synapse' object by generating a response to the prompt 
        given the context.
        """
        # Note: The protocol class name needs to be confirmed. 
        # Assuming `InfiniteContextSynapse` or similar from previous code validation.
        # Existing miner used `quasar.protocol.InfiniteContextSynapse`.
        
        start_time = time.time()
        
        try:
            # 1. Check if we need to fetch context from API (Light Synapse)
            context = getattr(synapse, 'context', '')
            prompt = getattr(synapse, 'prompt', '')
            
            if not context or not prompt:
                print(f"📡 [MINER] Light Synapse detected for task {synapse.task_id}. Fetching details from API...")
                bt.logging.info(f"📡 Fetching task details for {synapse.task_id} from API...")
                api_details = self._call_api(f"get_task_details/{synapse.task_id}")
                if api_details:
                    context = api_details.get('context', '')
                    prompt = api_details.get('prompt', '')
                    print(f"✅ [MINER] Context successfully retrieved ({len(context)} chars) from API.")
                    bt.logging.info(f"✅ Fetched task details from API.")
                else:
                    print(f"❌ [MINER] Failed to fetch task details from API!")
                    bt.logging.error(f"❌ Failed to fetch task details for {synapse.task_id}")
                    # We continue, but it will likely fail generation or be empty
            
            print(f"\n[MINER] 📝 Received Task: {synapse.task_id}")
            print(f"[MINER] Prompt: {prompt}")
            print(f"[MINER] Context Length: {len(context)} characters")
            bt.logging.info(f"📨 Received request. Context len: {len(context)} chars")

            # formatting input for Qwen
            # Qwen instruct format usually: <|im_start|>system...<|im_start|>user...
            # We can use apply_chat_template if available, or raw concatenation.
            
            messages = [
                {"role": "system", "content": """You are a specialized execution agent for semantic code analysis.

TASK STEPS:
1. Find the TARGET ENTITY mentioned in the task prompt
2. Locate the SEMANTIC DESCRIPTION of its operational state in the context (e.g., 'severe hydraulic instability', 'latency high')
3. MAP that description to one of these standardized Modes: 'CRITICAL', 'OPTIMIZED', 'SAFE', or 'DEGRADED'
4. Find and EXECUTE the corresponding Python function with the Mode and input value

OUTPUT FORMAT REQUIREMENT:
- Show your step-by-step reasoning
- End with the final numeric answer in LaTeX boxed format: \\boxed{number}
- Example: \\boxed{223}
- The boxed answer MUST be the last line of your response"""},
                {"role": "user", "content": f"""Context:
{context}

Task: {prompt}

IMPORTANT: After showing your work, you MUST format your final answer as \\boxed{{number}}. For example, if the answer is 223, write \\boxed{{223}}."""}
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
                    max_new_tokens=64, # Optimized for testing, but enough for reasoning
                    do_sample=True,      # Enable sampling for temperature to be valid
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # print(f"[MINER] ✅ Generated Response: {response}")  # Commented to reduce log spam
            bt.logging.info(f"✅ Generated response length: {len(response)} chars")
            
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

    def _get_signature(self) -> str:
        """Sign the hotkey address to authenticate with the API."""
        hotkey = self.wallet.hotkey.ss58_address
        return f"0x{self.wallet.hotkey.sign(hotkey).hex()}"

    def _call_api(self, endpoint: str, method: str = "GET", data: dict = None) -> typing.Union[dict, None]:
        """Helper to call the Validator API with authentication headers."""
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        headers = {
            "Hotkey": self.wallet.hotkey.ss58_address,
            "Signature": self._get_signature()
        }
        try:
            import requests
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=120)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=120)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            bt.logging.error(f"❌ API Call Error ({endpoint}): {e}")
            return None

    def _log_memory_usage(self, stage: str):
        """Log current RAM and GPU memory usage."""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        bt.logging.info(f"💾 [MEMORY] {stage} | RAM: {ram_usage:.2f} MB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_reserved(i) / (1024 * 1024)
                bt.logging.info(f"💾 [MEMORY] {stage} | GPU {i}: {mem:.2f} MB")

# This is the main function, which runs the miner.
if __name__ == "__main__":
    import argparse
    
    # Note: Bittensor's config system will automatically parse --miner.model_name
    # from command line arguments
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
