# Memory Optimization Guide

## Các thay đổi đã thực hiện để giảm memory:

### 1. **Model Generation Settings** (language_model.py)
- ✅ `max_new_tokens`: Giảm từ 256/512 → **64 tokens**
- ✅ `num_beams`: Giảm từ 2 → **1** (greedy decoding, không beam search)
- ✅ `do_sample`: Tắt sampling (**False**)
- ✅ `use_cache`: Bật KV cache (**True**)
- ✅ `max_length`: Giới hạn context **512 tokens**
- ✅ `truncation`: Bật truncation
- ✅ `torch.no_grad()`: Tắt gradient computation

### 2. **Batch Size** (config.py)
- ✅ `batch_size`: Giảm từ 8 → **2**

### 3. **Document Retrieval** (config.py)
- ✅ `top_k_docs`: Giảm từ 2 → **1** document
- ✅ `top_k_titles`: Giảm từ 7 → **3** titles

### 4. **Memory Cleanup**
- ✅ `torch.cuda.empty_cache()`: Gọi sau mỗi batch
- ✅ `gc.collect()`: Python garbage collection
- ✅ Explicit tensor deletion: `del` tensors sau khi sử dụng

### 5. **Model Loading** (model_loader.py)
- ✅ Gradient checkpointing enabled
- ✅ FP16 precision
- ✅ Auto device mapping

### 6. **PyTorch CUDA Configuration** (evaluation.py)
- ✅ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Nếu vẫn OOM:

### Option 1: Giảm batch_size xuống 1
```python
# config.py
"batch_size": 1,
```

### Option 2: Giảm thêm max_new_tokens
```python
# language_model.py, line ~67
'max_new_tokens': min(max_new_tokens, 32),  # Từ 64 → 32
```

### Option 3: Giảm context length
```python
# language_model.py, line ~59
max_length=256  # Từ 512 → 256
```

### Option 4: Tắt hoàn toàn document retrieval
```python
# config.py
"top_k_docs": 0,
```

### Option 5: Sử dụng 8-bit quantization
Trong evaluation.py, thay đổi:
```python
# Từ:
quant_type = None

# Sang:
quant_type = '8bit'
```

### Option 6: Sử dụng CPU offloading
```python
# model_loader.py
model_kwargs = {
    'pretrained_model_name_or_path': self.model_name,
    'torch_dtype': torch.float16,
    'device_map': 'auto',
    'offload_folder': 'offload',  # Offload to disk
    'offload_state_dict': True,
}
```

## So sánh Memory Usage:

| Configuration | Memory Usage | Speed |
|--------------|--------------|-------|
| Original (batch=8, beams=2) | ~15.8GB | Fast |
| Current (batch=2, beams=1) | ~12-14GB | Medium |
| batch=1, max_tokens=32 | ~8-10GB | Slow |
| With 8-bit quant | ~6-8GB | Very Slow |

## Kiểm tra memory:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
```
