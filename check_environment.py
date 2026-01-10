"""
Quick environment check script for Colab
Run this BEFORE running evaluation.py to verify your environment is correctly configured
"""
import sys
import platform

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

# 1. Platform check
print(f"\n1. Platform: {platform.system()} {platform.release()}")
print(f"   Python: {sys.version.split()[0]}")

# 2. PyTorch and CUDA check
try:
    import torch
    print(f"\n2. PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            print(f"           Memory: {total_memory:.2f} GB")
    else:
        print("   ⚠️  WARNING: CUDA NOT AVAILABLE!")
        print("   ")
        print("   On Colab, enable GPU:")
        print("   1. Click 'Runtime' in the menu")
        print("   2. Select 'Change runtime type'")
        print("   3. Hardware accelerator: GPU")
        print("   4. Click 'Save'")
        print("   5. Reconnect and run this script again")
        
except ImportError:
    print("\n2. ⚠️  ERROR: PyTorch not installed!")
    print("   Install with: !pip install torch")

# 3. Transformers check
try:
    import transformers
    print(f"\n3. Transformers: {transformers.__version__}")
except ImportError:
    print("\n3. ⚠️  ERROR: Transformers not installed!")
    print("   Install with: !pip install transformers")

# 4. bitsandbytes check (for quantization)
try:
    import bitsandbytes as bnb
    print(f"\n4. bitsandbytes: {bnb.__version__}")
    print("   ✓ Quantization (4bit/8bit) is available")
except ImportError:
    print("\n4. ⚠️  WARNING: bitsandbytes not installed!")
    print("   Quantization will NOT work without this library")
    print("   Install with: !pip install bitsandbytes")

# 5. Other required libraries
print("\n5. Other libraries:")
required_libs = ['sentence_transformers', 'faiss', 'pandas', 'numpy', 'datasets']
for lib in required_libs:
    try:
        module = __import__(lib.replace('-', '_'))
        version = getattr(module, '__version__', 'installed')
        print(f"   ✓ {lib}: {version}")
    except ImportError:
        print(f"   ✗ {lib}: NOT INSTALLED")

# 6. Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

import torch
cuda_ok = torch.cuda.is_available()
try:
    import bitsandbytes
    bnb_ok = True
except:
    bnb_ok = False

if cuda_ok and bnb_ok:
    print("\n✓ Environment is READY!")
    print("  You can run with quantization:")
    print("  !python evaluation.py --quant 4bit")
elif cuda_ok and not bnb_ok:
    print("\n⚠️  GPU available but bitsandbytes missing")
    print("  Install bitsandbytes to enable quantization:")
    print("  !pip install bitsandbytes")
    print("  Or run without quantization:")
    print("  !python evaluation.py")
elif not cuda_ok:
    print("\n⚠️  GPU NOT AVAILABLE!")
    print("  Enable GPU in Colab runtime settings")
    print("  This model requires GPU to run efficiently")
else:
    print("\n⚠️  Environment has issues. Review warnings above.")

print("=" * 70)
