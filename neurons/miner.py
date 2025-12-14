# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 QUASAR-TAO Team

import os
import time
import typing
import torch
import bittensor as bt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the parent directory to path so we can import template
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import template
from template.base.miner import BaseMinerNeuron

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
        
        # Get model name from config or use default
        self.model_name = getattr(self.config.miner, 'model_name', "silx-ai/Quasar-2M-Base")
        
        # Validate model selection
        if self.model_name not in self.SUPPORTED_MODELS:
            bt.logging.warning(f"Model {self.model_name} not in supported list. Proceeding anyway...")
            bt.logging.info(f"Supported models: {', '.join(self.SUPPORTED_MODELS)}")
        
        try:
            bt.logging.info(f"Loading model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.model.eval()
            bt.logging.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            bt.logging.error(f"Failed to load model {self.model_name}: {e}")
            bt.logging.warning("Please ensure you have access/internet or specify a valid model.")
            raise e

    async def forward(
        self, synapse: template.protocol.BenchmarkEvaluationSynapse
    ) -> template.protocol.BenchmarkEvaluationSynapse:
        """
        Processes the incoming 'synapse' object by generating a response to the prompt 
        given the context.
        """
        # Note: The protocol class name needs to be confirmed. 
        # Assuming `InfiniteContextSynapse` or similar from previous code validation.
        # Existing miner used `template.protocol.InfiniteContextSynapse`.
        
        start_time = time.time()
        
        try:
            # Extract inputs
            # The previous miner used `synapse.context` (which might be huge) + prompt logic?
            # Existing miner code: `context = getattr(synapse, 'context', '')`
            
            context = getattr(synapse, 'context', '')
            prompt = getattr(synapse, 'prompt', '') # LongBench usually has a specific question/prompt
            
            bt.logging.info(f"📨 Received request. Context len: {len(context)} chars")

            # formatting input for Qwen
            # Qwen instruct format usually: <|im_start|>system...<|im_start|>user...
            # We can use apply_chat_template if available, or raw concatenation.
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant capable of processing long contexts."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
            ]
            
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

            # Generate
            # Long context generation might be slow.
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512, # LongBench usually requires short answers
                    do_sample=False, # Deterministic for benchmarks often better, or low temp
                    temperature=0.1
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            bt.logging.info(f"✅ Generated response: {response[:100]}...")
            
            synapse.response = response
            synapse.processing_time = time.time() - start_time
            
            # Fill other metadata if required by protocol
            if hasattr(synapse, 'model_name'):
                synapse.model_name = self.model_name
                
        except Exception as e:
            bt.logging.error(f"❌ Error during generation: {e}")
            synapse.response = f"Error: {str(e)}"
            synapse.processing_time = time.time() - start_time

        return synapse

    async def blacklist(
        self, synapse: template.protocol.BenchmarkEvaluationSynapse
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

    async def priority(self, synapse: template.protocol.BenchmarkEvaluationSynapse) -> float:
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

# This is the main function, which runs the miner.
if __name__ == "__main__":
    import argparse
    
    # Note: Bittensor's config system will automatically parse --miner.model_name
    # from command line arguments
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
