from models.llm_handler import LLMHandler

def test_llm():
    print("=" * 50)
    print("Testing LLM Handler")
    print("=" * 50)
    
    # Initialize handler
    handler = LLMHandler()
    
    # Load model (this will download ~3GB if first time)
    handler.load_model()
    
    # Test prompt
    prompt = "Analyze this IoT sensor data: Temperature: 25°C, Humidity: 60%. Is this normal?"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    # Generate
    result = handler.generate(prompt, max_new_tokens=100)
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"Generated Text:\n{result['generated_text']}\n")
    print(f"Inference Time: {result['inference_time']:.2f}s")
    print(f"Tokens Generated: {result['tokens_generated']}")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/sec")
    
    # Cleanup
    handler.unload_model()

if __name__ == "__main__":
    test_llm()