import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {torch.version.__version__ if hasattr(torch.version, '__version__') else 'N/A'}")
print(f"\nMPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✓ Using device: {device}")
    
    # Test tensor operations on MPS
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x @ y
    print(f"✓ Matrix multiplication test passed!")
    print(f"Result shape: {z.shape}")
else:
    print("\n✗ MPS not available, will use CPU")
    device = torch.device("cpu")