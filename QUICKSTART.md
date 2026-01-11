# 🚀 RAG Best Practices - Quick Start Guide

This guide helps you run comprehensive evaluations of different RAG configurations, including expand_query, focus mode, ICL (In-Context Learning), and hybrid combinations.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Demo (10 samples)](#quick-demo-10-samples)
- [Full Evaluation](#full-evaluation)
- [Configuration Descriptions](#configuration-descriptions)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## ✅ Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space
- Internet connection (for downloading models and datasets)

### Recommended Environment:
- **Kaggle:** Tesla P100-PCIE-16GB
- **Colab:** T4 or better
- **Local:** RTX 3090/4090 or A100

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-repo/RAG_best_practices.git
cd RAG_best_practices
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# For Colab, use:
pip install -r requirements_colab.txt
```

### 3. Download Required Resources
```bash
# Download knowledge base (articles)
# Place articles_l3.pkl in resources/ folder
```

### 4. Install Spacy Model
```bash
python -m spacy download en_core_web_sm
```

---

## 🎯 Quick Demo (10 samples)

Perfect for testing and debugging. Runs in ~10-15 minutes.

### Test Suite (All 7 Configurations)

```bash
# TruthfulQA - 10 samples
python evaluation.py \
  --dataset truthfulqa \
  --num-samples 10 \
  --quant 8bit \
  --config-set test_suite

# MMLU - 10 samples  
python evaluation.py \
  --dataset mmlu \
  --num-samples 10 \
  --quant 8bit \
  --config-set test_suite
```

**What gets tested:**
1. ✅ Baseline (no features)
2. ✅ Expand Query only
3. ✅ Focus only
4. ✅ ICL only
5. ✅ Expand Query + Focus
6. ✅ Focus + ICL
7. ✅ Hybrid (All 3: Expand Query + Focus + ICL)

**Expected time:** 10-15 minutes (with 8-bit quantization)

---

## 🔥 Full Evaluation

Complete evaluation on full datasets. Runs in ~2-3 hours.

### Test Suite - Full Dataset

```bash
# TruthfulQA - All ~800 questions
python evaluation.py \
  --dataset truthfulqa \
  --quant 8bit \
  --config-set test_suite

# MMLU - All ~1024 questions (32 per subject)
python evaluation.py \
  --dataset mmlu \
  --quant 8bit \
  --config-set test_suite
```

**Expected time:** 
- TruthfulQA: ~1.5-2 hours
- MMLU: ~2-3 hours

---

## 🎨 Configuration Descriptions

### 1️⃣ Baseline
```yaml
Features: None
Purpose: Pure RAG baseline without any enhancements
Speed: ⚡⚡⚡ Fast
```

### 2️⃣ Expand Query Only
```yaml
Features: Query expansion with title-based filtering
Purpose: Tests if expanding queries into keywords improves retrieval
Parameters:
  - expand_query: True
  - top_k_titles: 5
  - top_k_docs: 3
Speed: ⚡⚡ Medium
```

### 3️⃣ Focus Only
```yaml
Features: Sentence-level refinement
Purpose: Tests hierarchical retrieval (docs → sentences)
Parameters:
  - top_k_docs: 10
  - focus: 3
Speed: ⚡ Slow (builds sentence index)
```

### 4️⃣ ICL Only
```yaml
Features: In-Context Learning with example Q&A pairs
Purpose: Tests learning from similar examples
Parameters:
  - icl_kb: True
  - top_k_docs: 2
Speed: ⚡⚡⚡ Fast
```

### 5️⃣ Expand Query + Focus
```yaml
Features: Query expansion + sentence refinement
Purpose: Tests combining title filtering with granular search
Parameters:
  - expand_query: True
  - top_k_docs: 10
  - focus: 3
Speed: ⚡ Slow
```

### 6️⃣ Focus + ICL
```yaml
Features: ICL examples + sentence refinement
Purpose: Tests if focus improves ICL retrieval
Parameters:
  - icl_kb: True
  - top_k_docs: 10
  - focus: 3
Speed: ⚡ Slow
```

### 7️⃣ Hybrid (All Features)
```yaml
Features: Expand Query + Focus + ICL (all combined)
Purpose: Tests optimal combination with separate ICL and article indices
Parameters:
  - hybrid_kb: True
  - top_k_icl: 2 (ICL examples)
  - top_k_docs: 10 (articles)
  - focus: 3 (sentences)
  - expand_query: True
Speed: ⚡ Slowest (most comprehensive)
```

---

## ⚡ Performance Tips

### 1. Use Quantization
```bash
# 8-bit (recommended - 2x speedup, minimal quality loss)
--quant 8bit

# 4-bit (fastest - 4x speedup, some quality loss)
--quant 4bit

# No quantization (slowest but best quality)
# Don't specify --quant
```

### 2. Adjust Batch Size
Edit `config.py`:
```python
"batch_size": 16,  # Increase if you have more VRAM
```

### 3. Cache Indices
Indices are automatically reused when configs share same index_builder settings.

### 4. Reduce Focus Parameters
For faster testing, reduce focus depth:
```python
"top_k_docs": 5,   # Instead of 10
"focus": 2,        # Instead of 3
```

### 5. Skip Heavy Configs
Run only specific configs:
```bash
# Only run lightweight configs (1, 2, 4)
# Edit config.py and comment out heavy configs
```

---

## 📊 Output Files

After running, find results in:
```
outputs/
└── truthfulqa/
    └── test_suite_01-11_14-30/
        ├── eval_results_all.json          # Summary of all configs
        ├── timing_summary.json            # Time breakdown
        ├── eval_results_1_Baseline.json   # Individual results
        ├── evaluation_1_Baseline.pkl      # Detailed results
        ├── config_1_Baseline.json         # Config used
        └── ...
```

### Key Metrics:
- **r1f1, r2f1, rLf1**: ROUGE F1 scores (higher = better)
- **similarity**: Cosine similarity (higher = better)
- **mauve**: MAUVE score (higher = better)
- **timing**: Time breakdown for each stage

---

## 🐛 Troubleshooting

### Error: CUDA out of memory
**Solution:**
```bash
# Use 8-bit quantization
--quant 8bit

# Or reduce batch size in config.py
"batch_size": 4
```

### Error: articles_l3.pkl not found
**Solution:**
```bash
# Make sure resources/articles_l3.pkl exists
# Download from project resources
```

### Error: Module 'spacy' not found
**Solution:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Slow Performance on Windows
**Note:** Quantization is not supported on Windows. Either:
1. Run without quantization (slower)
2. Use WSL2 or Linux
3. Use Kaggle/Colab

### Error: bitsandbytes not available
**Solution (Linux/Colab):**
```bash
pip install bitsandbytes
```

**Solution (Windows):** Quantization not supported - run without `--quant`

---

## 📈 Typical Runtimes (Tesla P100, 8-bit quant)

### Quick Demo (10 samples):
```
1_Baseline:              ~1 min
2_ExpandQuery_Only:      ~1.5 min
3_Focus_Only:            ~2 min
4_ICL_Only:              ~1 min
5_ExpandQuery_Focus:     ~2.5 min
6_Focus_ICL:             ~2 min
7_Hybrid_All_Features:   ~3 min
────────────────────────────────
Total:                   ~13 min
```

### Full Evaluation (TruthfulQA ~800 samples):
```
1_Baseline:              ~10 min
2_ExpandQuery_Only:      ~12 min
3_Focus_Only:            ~25 min
4_ICL_Only:              ~8 min
5_ExpandQuery_Focus:     ~30 min
6_Focus_ICL:             ~20 min
7_Hybrid_All_Features:   ~35 min
────────────────────────────────
Total:                   ~2.5 hours
```

---

## 🎓 Understanding the Output

### Example output (eval_results_7_Hybrid_All_Features.json):
```json
{
  "r1f1": 0.4532,        // ROUGE-1 F1 (unigram overlap)
  "r2f1": 0.3214,        // ROUGE-2 F1 (bigram overlap)
  "rLf1": 0.4123,        // ROUGE-L F1 (longest common subsequence)
  "similarity": 0.7845,  // Cosine similarity
  "mauve": 0.6234,       // MAUVE score (distribution similarity)
  "timing": {
    "model_load_time": 87.32,
    "rag_init_time": 245.18,
    "evaluation_time": 124.56,
    "total_time": 457.06
  }
}
```

**Higher is better** for all metrics!

---

## 🔬 Advanced Usage

### Custom Config
Create your own config in `config.py`:
```python
configs_custom = {
    "MyConfig": {
        "ralm": {
            "top_k_docs": 5,
            "expand_query": True,
            "focus": 2
        }
    }
}
```

### Run Specific Dataset Subset
```python
# In evaluation.py, after loading test_data:
test_data = test_data[test_data['category'] == 'science']
```

### Custom Knowledge Base
```python
# Replace in evaluation.py:
knowledge_base = pd.read_pickle('path/to/your/custom_kb.pkl')
```

---

## 📚 Additional Resources

- **Full Documentation:** See `README.md`
- **Colab Guide:** See `COLAB_GUIDE.md`
- **Kaggle Notebook:** See `run-on-kaggle.ipynb`
- **Paper:** [Link to paper if available]

---

## 💡 Tips for Best Results

1. **Start with Quick Demo:** Always test with 10 samples first
2. **Use 8-bit Quantization:** Best speed/quality tradeoff
3. **Monitor GPU Memory:** Use `nvidia-smi` to check usage
4. **Compare Metrics:** Focus on similarity and ROUGE-L for overall quality
5. **Timing Analysis:** Check `timing_summary.json` to identify bottlenecks

---

## 🎉 Next Steps

After running evaluations:
1. **Analyze Results:** Compare metrics across configs
2. **Visualize:** Create plots from JSON outputs
3. **Optimize:** Adjust parameters based on findings
4. **Deploy:** Use best config for production

---

## 📞 Support

- **Issues:** Open GitHub issue
- **Questions:** Check README.md FAQ section
- **Contributions:** PRs welcome!

---

**Happy Evaluating! 🚀**
