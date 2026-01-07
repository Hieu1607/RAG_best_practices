# RAG Best Practices - Google Colab Guide

## 📋 Checklist trước khi chạy:

✅ Code đã sẵn sàng cho Colab  
✅ Tự động detect và dùng GPU  
✅ File paths tương thích Linux  

## 🚀 Quy trình đưa lên Colab:

### **Cách 1: Upload trực tiếp (Khuyên dùng)**

1. **Zip toàn bộ folder:**
   ```powershell
   # Trên Windows, zip folder RAG_best_practices
   Compress-Archive -Path RAG_best_practices -DestinationPath RAG_best_practices.zip
   ```

2. **Mở Google Colab:**
   - Truy cập: https://colab.research.google.com
   - File > Upload notebook > Chọn `run_on_colab.ipynb`

3. **Enable GPU:**
   - Runtime > Change runtime type
   - Hardware accelerator > **GPU (T4)**
   - Save

4. **Chạy từng cell theo thứ tự** trong notebook `run_on_colab.ipynb`

---

### **Cách 2: Qua GitHub**

1. **Push code lên GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/RAG_best_practices.git
   git push -u origin main
   ```

2. **Clone trong Colab:**
   ```python
   !git clone https://github.com/YOUR_USERNAME/RAG_best_practices.git
   %cd RAG_best_practices
   ```

3. **Upload resources folder riêng** (vì file .pkl quá lớn cho GitHub):
   - Upload lên Google Drive
   - Mount Drive trong Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     !cp -r /content/drive/MyDrive/RAG_resources ./resources
     ```

---

### **Cách 3: Qua Google Drive**

1. **Upload folder lên Google Drive:**
   - Drag & drop `RAG_best_practices` vào Google Drive

2. **Mount trong Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/RAG_best_practices
   ```

3. **Cài dependencies và chạy**

---

## 🔧 Điều chỉnh cho Colab:

### ✅ **Đã tự động hoạt động:**

- ✅ GPU detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- ✅ File paths: Dùng relative paths (`resources/`, `./resources/`)
- ✅ Model loading: Tự động dùng `device_map="auto"`

### 🔄 **Thay đổi khuyên dùng:**

1. **Dùng faiss-gpu thay vì faiss-cpu:**
   ```bash
   pip install faiss-gpu
   ```

2. **Enable quantization** (code đã tắt trên Windows):
   - Colab có GPU → có thể dùng 4-bit/8-bit quantization
   - Edit `evaluation.py` line 129-130:
     ```python
     model_loader_generation = ModelLoader(config['generation_model_name'], 'causal', quant_type='4bit')
     model_loader_seq2seq = ModelLoader(config['seq2seq_model_name'], 'seq2seq', quant_type='4bit')
     ```

3. **Giảm test set** để chạy nhanh hơn (optional):
   ```python
   test_data = test_data.head(50)  # Chỉ test 50 samples đầu
   ```

---

## 📊 Ưu điểm chạy trên Colab:

| Feature | Local (CPU) | Google Colab (T4 GPU) |
|---------|------------|----------------------|
| Speed | 🐌 Slow | ⚡ Fast (5-10x) |
| Memory | Limited | 15GB GPU + 12GB RAM |
| Quantization | ❌ Not supported | ✅ 4-bit/8-bit |
| Cost | Free | Free (limited hours) |

---

## 🎯 Bước chạy trong Colab:

1. Upload notebook `run_on_colab.ipynb`
2. Enable GPU (T4)
3. Chạy Cell 1: Check GPU
4. Chạy Cell 2: Upload/Clone code
5. Chạy Cell 3: Install dependencies
6. Chạy Cell 4: Download resources
7. Chạy Cell 5: Clone mixtral-offloading
8. Chạy Cell 6: Run evaluation
9. Chạy Cell 7: Download results

---

## ⚠️ Lưu ý:

- **Session timeout**: Colab free có giới hạn ~12 giờ/session
- **GPU quota**: Giới hạn ~15-20 giờ GPU/tuần
- **Save outputs**: Download results về máy trước khi session end
- **Large models**: Mistral-7B cần ~14GB GPU memory (T4 có 15GB - vừa đủ)

---

## 🐛 Troubleshooting:

**Lỗi out of memory:**
```python
# Giảm batch_size trong config.py
"batch_size": 4  # Thay vì 8
```

**Lỗi quantization:**
```python
# Nếu 4-bit không work, thử 8-bit hoặc None
quant_type='8bit'  # hoặc None
```

**Lỗi triton (Mixtral-8x7B):**
- Code đã có fallback tự động
- Sẽ load model thông thường nếu triton không khả dụng

---

## 📝 Sau khi chạy xong:

1. Results được lưu trong `outputs/`
2. Download về máy: Chạy cell cuối trong notebook
3. Analyze results bằng pandas locally

Happy coding! 🚀
