"""
Script to set PyTorch CUDA memory configuration for better memory management.
This helps prevent fragmentation issues.
"""
import os

# Set PyTorch CUDA allocator configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("✓ PyTorch CUDA memory configuration set:")
print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
print("\nThis helps prevent memory fragmentation.")
print("Import this at the top of your evaluation script.")
