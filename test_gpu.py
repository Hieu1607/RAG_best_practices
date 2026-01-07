"""
Quick test script to verify GPU usage
Run this on Colab to check if GPU is properly configured
"""
import torch
import platform

print("=" * 60)
print("GPU Configuration Test")
print("=" * 60)

# System info
print(f"\nPlatform: {platform.system()}")
print(f"Python version: {platform.python_version()}")

# CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU memory: {total_memory:.2f} GB")
else:
    print("⚠️  WARNING: CUDA is not available!")
    print("Make sure you're running on Colab with GPU runtime:")
    print("  Runtime > Change runtime type > Hardware accelerator > GPU")

# Test tensor operation
print("\n" + "=" * 60)
print("Testing Tensor Operations")
print("=" * 60)

# Create a tensor and move to GPU
x = torch.randn(1000, 1000)
print(f"Tensor created on: {x.device}")

if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print(f"Tensor moved to: {x_gpu.device}")
    
    # Simple operation
    y = torch.matmul(x_gpu, x_gpu)
    print(f"Matrix multiplication result on: {y.device}")
    print("✓ GPU operations working correctly!")
else:
    print("Skipping GPU test - CUDA not available")

print("\n" + "=" * 60)
print("Checking bitsandbytes (for quantization)")
print("=" * 60)

try:
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes version: {bnb.__version__}")
    print("✓ Quantization (4bit/8bit) will be available")
except ImportError:
    print("⚠️  bitsandbytes not installed")
    print("Install with: !pip install bitsandbytes")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
