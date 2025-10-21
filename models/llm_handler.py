import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import time
from models.config import model_config

class LLMHandler:
    """Handles LLM inference with M1 optimization"""
    
    def __init__(self):
        self.device = torch.device(model_config.device)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the Qwen model optimized for M1"""
        print(f"Loading model: {model_config.model_name}")
        print(f"Using device: {self.device}")
        
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name,
            trust_remote_code=True
        )
        
        # Load model with M1 optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            torch_dtype=torch.float16,  # Use FP16 for M1
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Set to evaluation mode
        self.model.eval()
        self.model_loaded = True
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"✓ Model size: {self._get_model_size():.2f} MB")
        
    def _get_model_size(self) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        temperature = temperature or model_config.temperature
        top_p = top_p or model_config.top_p
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model_config.max_length
        ).to(self.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Calculate tokens per second
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / inference_time if inference_time > 0 else 0
        
        return {
            "generated_text": generated_text,
            "inference_time": inference_time,
            "tokens_generated": num_tokens,
            "tokens_per_second": tokens_per_sec
        }
    
    def unload_model(self):
        """Free up memory"""
        if self.model:
            del self.model
            del self.tokenizer
            torch.mps.empty_cache()
            self.model_loaded = False
            print("✓ Model unloaded")