# Quick Commands Cheat Sheet

## 🚀 Most Common Commands

### Quick Demo (10 samples, ~15 minutes)
```bash
# Linux/Mac
./quick_eval.sh truthfulqa 10 8bit

# Windows  
quick_eval.bat truthfulqa 10 none

# Direct Python
python evaluation.py --dataset truthfulqa --num-samples 10 --quant 8bit --config-set test_suite
```

### Full Evaluation (~2-3 hours)
```bash
# TruthfulQA (800 samples)
python evaluation.py --dataset truthfulqa --quant 8bit --config-set test_suite

# MMLU (1024 samples)
python evaluation.py --dataset mmlu --quant 8bit --config-set test_suite
```

---

## 📋 All Config Sets

### Test Suite (7 configs - all combinations)
```bash
--config-set test_suite
```
Tests: Baseline, Expand Query, Focus, ICL, Expand+Focus, Focus+ICL, Hybrid (all 3)

### Run 1 (3 configs - original)
```bash
--config-set run1
```
Tests: Base, HelpV2, Instruct45B (Mixtral)

### Run 2 (4 configs - advanced)
```bash
--config-set run2
```
Tests: ICL1D+, Focus80_Doc80, Hybrid_ICL2_Doc3_Focus

---

## 🎯 Quick Examples

### Test specific feature
```bash
# Just test expand query
python evaluation.py --dataset truthfulqa --num-samples 10 --config-set test_suite
# Then check: outputs/.../eval_results_2_ExpandQuery_Only.json

# Just test focus
# Check: outputs/.../eval_results_3_Focus_Only.json

# Test hybrid (all 3 features)
# Check: outputs/.../eval_results_7_Hybrid_All_Features.json
```

### Different quantization
```bash
# 8-bit (recommended)
--quant 8bit

# 4-bit (fastest)
--quant 4bit

# No quantization (slowest, best quality)
# Don't specify --quant
```

### Different datasets
```bash
# TruthfulQA
--dataset truthfulqa

# MMLU
--dataset mmlu
```

---

## 📊 Check Results

```bash
# View all results
cat outputs/truthfulqa/test_suite_*/eval_results_all.json

# View timing
cat outputs/truthfulqa/test_suite_*/timing_summary.json

# View specific config
cat outputs/truthfulqa/test_suite_*/eval_results_7_Hybrid_All_Features.json
```

---

## 🔧 Customize Parameters

Edit `config.py`:
```python
# Make it faster (reduce these)
"top_k_docs": 3,      # Instead of 10
"focus": 2,           # Instead of 3
"batch_size": 16,     # Increase if more VRAM

# Make it slower but better
"top_k_docs": 20,
"focus": 5,
```

---

## 🐛 Common Issues

### CUDA out of memory
```bash
# Use 8-bit quantization
--quant 8bit

# Or edit config.py: "batch_size": 4
```

### Missing resources
```bash
# Download articles_l3.pkl to resources/
# Install spacy model:
python -m spacy download en_core_web_sm
```

### Windows quantization error
```bash
# Windows doesn't support quantization
# Run without --quant flag
quick_eval.bat truthfulqa 10 none
```

---

## 📚 More Info

- **Full Guide:** See `QUICKSTART.md`
- **Documentation:** See `README.md`
- **Colab:** See `COLAB_GUIDE.md`

---

**Need Help?** Check QUICKSTART.md for detailed documentation!
