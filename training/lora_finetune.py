import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
import os

class LoRAFineTuner:
    """Fine-tune Qwen model with LoRA for IoT domain"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "mps"
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def load_base_model(self):
        """Load the base model and tokenizer"""
        print(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with float32 for MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for MPS training
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"✓ Base model loaded")
        print(f"✓ Model parameters: {self.model.num_parameters():,}")
        
    def setup_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list = None
    ):
        """Configure LoRA for efficient fine-tuning"""
        
        if target_modules is None:
            # Target attention layers for Qwen
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        print(f"\nConfiguring LoRA:")
        print(f"  - Rank (r): {r}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - Target modules: {target_modules}")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        print(f"\n✓ LoRA applied successfully")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")
        
    def prepare_dataset(self, data_file: str = "training/iot_chat_format.json"):
        """Load and prepare training dataset"""
        
        print(f"\nLoading training data from {data_file}...")
        
        # Load dataset
        dataset = load_dataset('json', data_files=data_file, split='train')
        
        def format_chat_template(example):
            """Format data using chat template"""
            messages = example['messages']
            
            # Format as chat
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            return {"text": text}
        
        # Format dataset
        dataset = dataset.map(format_chat_template, remove_columns=dataset.column_names)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding=False
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print(f"✓ Loaded {len(tokenized_dataset)} training examples")
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset,
        output_dir: str = "training/lora_model",
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-4,
        save_steps: int = 50
    ):
        """Train the model with LoRA"""
        
        print(f"\n{'='*60}")
        print("Starting LoRA Fine-tuning")
        print(f"{'='*60}")
        print(f"Training configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Output directory: {output_dir}")
        
        # Training arguments optimized for MPS
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=False,  # Disable fp16 for MPS
            save_steps=save_steps,
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            load_best_model_at_end=False,
            use_mps_device=True,  # Use MPS
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print(f"\n🚀 Starting training...\n")
        trainer.train()
        
        print(f"\n✓ Training complete!")
        
        # Save final model
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Model saved to {output_dir}")
        
        return trainer
    
    def test_finetuned_model(self, prompt: str):
        """Test the fine-tuned model"""
        
        print(f"\n{'='*60}")
        print("Testing Fine-tuned Model")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}\n")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"Response:\n{response}\n")
        print(f"{'='*60}")


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("LoRA Fine-tuning for IoT Domain")
    print("="*60)
    
    # Initialize fine-tuner
    finetuner = LoRAFineTuner(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="mps"
    )
    
    # Load base model
    finetuner.load_base_model()
    
    # Setup LoRA
    finetuner.setup_lora(
        r=8,              # LoRA rank
        lora_alpha=16,    # LoRA alpha
        lora_dropout=0.05 # Dropout rate
    )
    
    # Prepare dataset
    train_dataset = finetuner.prepare_dataset("training/iot_chat_format.json")
    
    # Train
    trainer = finetuner.train(
        train_dataset,
        output_dir="training/lora_model",
        num_epochs=2,  # Reduced epochs for demo
        batch_size=2,
        learning_rate=2e-4,
        save_steps=50
    )
    
    # Test the model
    test_prompt = """Analyze this surveillance sensor reading

Context: Motion detected: True, Zone: entrance, Confidence: 0.98, Device: CAM_505"""
    
    finetuner.test_finetuned_model(test_prompt)
    
    print("\n✓ Fine-tuning complete!")
    print("✓ LoRA adapter saved to: training/lora_model")
    print("\nYou can now load this adapter with:")
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base_model, 'training/lora_model')")


if __name__ == "__main__":
    main()
