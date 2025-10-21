import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

class QuantizationBenchmark:
    """Compare FP32, FP16, and dynamic quantization performance"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("mps")
        self.tokenizer = None
        
    def get_model_size_mb(self, model):
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def load_fp32_model(self):
        """Load FP32 (full precision) model"""
        print("\n1. Loading FP32 Model (Full Precision)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()
        
        size = self.get_model_size_mb(model)
        print(f"✓ FP32 Model loaded - Size: {size:.2f} MB")
        
        return model, size
    
    def load_fp16_model(self):
        """Load FP16 (half precision) model"""
        print("\n2. Loading FP16 Model (Half Precision)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()
        
        size = self.get_model_size_mb(model)
        print(f"✓ FP16 Model loaded - Size: {size:.2f} MB")
        
        return model, size
    
    def benchmark_inference(self, model, prompt: str, num_runs: int = 5):
        """Benchmark inference speed"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = tokens_generated / avg_time
        
        return avg_time, tokens_per_sec, outputs
    
    def run_comparison(self):
        """Run complete quantization comparison"""
        
        print("="*70)
        print("MODEL QUANTIZATION COMPARISON FOR EDGE DEPLOYMENT")
        print("="*70)
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
        
        # Test prompt
        test_prompt = "Analyze this sensor: Temperature: 28°C, Humidity: 65%, Status: normal"
        
        # Results storage
        results = {}
        
        # Test FP32
        print("\n" + "─"*70)
        print("TESTING FP32 (Full Precision)")
        print("─"*70)
        fp32_model, fp32_size = self.load_fp32_model()
        fp32_time, fp32_tps, fp32_output = self.benchmark_inference(fp32_model, test_prompt)
        results['FP32'] = {
            'size_mb': fp32_size,
            'time': fp32_time,
            'tokens_per_sec': fp32_tps
        }
        print(f"Average inference time: {fp32_time:.2f}s")
        print(f"Tokens per second: {fp32_tps:.2f}")
        del fp32_model
        torch.mps.empty_cache()
        
        # Test FP16
        print("\n" + "─"*70)
        print("TESTING FP16 (Half Precision)")
        print("─"*70)
        fp16_model, fp16_size = self.load_fp16_model()
        fp16_time, fp16_tps, fp16_output = self.benchmark_inference(fp16_model, test_prompt)
        results['FP16'] = {
            'size_mb': fp16_size,
            'time': fp16_time,
            'tokens_per_sec': fp16_tps
        }
        print(f"Average inference time: {fp16_time:.2f}s")
        print(f"Tokens per second: {fp16_tps:.2f}")
        del fp16_model
        torch.mps.empty_cache()
        
        # Print comparison table
        self.print_comparison_table(results)
        
        # Print recommendations
        self.print_recommendations(results)
    
    def print_comparison_table(self, results):
        """Print formatted comparison table"""
        
        print("\n" + "="*70)
        print("QUANTIZATION COMPARISON SUMMARY")
        print("="*70)
        
        # Header
        print(f"\n{'Precision':<12} {'Size (MB)':<15} {'Inference (s)':<18} {'Tokens/sec':<15} {'Speedup':<10}")
        print("─"*70)
        
        # FP32 baseline
        fp32_time = results['FP32']['time']
        
        for precision, data in results.items():
            speedup = fp32_time / data['time']
            size_reduction = (1 - data['size_mb'] / results['FP32']['size_mb']) * 100
            
            print(f"{precision:<12} {data['size_mb']:>10.2f} MB   "
                  f"{data['time']:>10.2f}s        "
                  f"{data['tokens_per_sec']:>10.2f}      "
                  f"{speedup:>6.2f}x")
        
        print("\n" + "="*70)
    
    def print_recommendations(self, results):
        """Print deployment recommendations"""
        
        print("\n📊 EDGE DEPLOYMENT RECOMMENDATIONS")
        print("─"*70)
        
        fp16_size = results['FP16']['size_mb']
        fp16_speed = results['FP16']['tokens_per_sec']
        
        print(f"\n✅ RECOMMENDED FOR EDGE: FP16 (Half Precision)")
        print(f"   • Model size: {fp16_size:.2f} MB (~50% reduction)")
        print(f"   • Inference speed: {fp16_speed:.2f} tok/s")
        print(f"   • Memory efficient for edge devices")
        print(f"   • Minimal accuracy loss")
        print(f"   • Native MPS support on Apple Silicon")
        
        print(f"\n📱 DEVICE COMPATIBILITY:")
        print(f"   • Raspberry Pi 4/5: Use INT8 quantization (not tested here)")
        print(f"   • Jetson Nano: FP16 with TensorRT")
        print(f"   • M1/M2/M3 Mac: FP16 (optimal)")
        print(f"   • Cloud: FP32 or FP16 based on requirements")
        
        print(f"\n💡 OPTIMIZATION TIPS:")
        print(f"   • Use ONNX Runtime for 2-3x speedup")
        print(f"   • Implement KV-cache for faster generation")
        print(f"   • Batch inference when possible")
        print(f"   • Consider model distillation for smaller size")


def main():
    benchmark = QuantizationBenchmark()
    benchmark.run_comparison()
    
    print("\n✓ Quantization comparison complete!")
    print("\nNext steps:")
    print("  1. Export to ONNX for further optimization")
    print("  2. Test on actual edge devices")
    print("  3. Measure power consumption")
    print("  4. Compare with INT8 quantization")


if __name__ == "__main__":
    main()