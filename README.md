# RAG Best Practices

Framework đánh giá Retrieval-Augmented Generation (RAG) với nhiều cấu hình khác nhau.

## Giới thiệu

Dự án này triển khai hệ thống RAG để đánh giá tác động của các thành phần RAG khác nhau:
- **Query Expansion**: Mở rộng truy vấn để retrieve thêm context
- **Retrieval**: Tìm kiếm document/câu tương liên quan
- **Generation**: Sinh câu trả lời dựa trên context retrieved

## Cài đặt

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# Hoặc (lean version - chỉ PyTorch):
pip install -r requirements_colab.txt

# 2. Tải knowledge base từ Google Drive
# https://drive.google.com/drive/folders/1_-2PHI0-Wz1VjnW5Yvy5Ne9C7mMWk1nf
# Giải nén vào thư mục resources/
```

## Sử dụng

### Chạy đánh giá trên TruthfulQA:
```bash
python evaluation.py --dataset truthfulqa --config-set test_suite
```

### Chạy trên MMLU:
```bash
python evaluation.py --dataset mmlu --config-set test_suite
```

### Options:
- `--dataset`: truthfulqa hoặc mmlu
- `--config-set`: test_suite, run1, hoặc run2
- `--num-samples`: Số lượng samples để test nhanh
- `--quant`: 4bit, 8bit, hoặc None
- `--output-dir`: Thư mục output (mặc định: outputs)
- `--seed`: Random seed (mặc định: 42)

## Cấu hình

Các cấu hình được định nghĩa trong [config.py](config.py):

### Test Suite (7 cấu hình):
1. **Baseline**: RAG cơ bản
2. **ExpandQuery_Only**: Query expansion
3. **Focus_Only**: Focus mechanism
4. **ICL_Only**: In-Context Learning
5. **ExpandQuery_Focus**: Kết hợp expansion + focus
6. **Focus_ICL**: Kết hợp focus + ICL
7. **Hybrid_All_Features**: Tất cả tính năng

### Models mặc định:
- Generation: `mistralai/Mistral-7B-Instruct-v0.2`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Seq2seq: `google/flan-t5-small`

Chi tiết cấu hình xem [config.md](config.md).

## Kết quả

Kết quả được lưu trong thư mục `outputs/{dataset}/`:
```
outputs/
├── mmlu/
│   └── runtest_suite_{timestamp}/
│       ├── config_{name}.json
│       ├── eval_results_{name}.json
│       ├── eval_results_all.json
│       └── timing_summary.json
└── truthfulqa/
    └── runtest_suite_{timestamp}/
```

### Metrics:
- **ROUGE** (r1f1, r2f1, rLf1): Độ trùng khớp văn bản
- **Similarity**: Độ tương đồng ngữ nghĩa
- **MAUVE**: Khoảng cách phân phối văn bản
- **Timing**: Thời gian model loading, RAG init, evaluation

## Cấu trúc Project

```
RAG_best_practices/
├── model/                    # Core RAG implementation
│   ├── index_builder.py     # Xây dựng document index
│   ├── retriever.py         # Retrieve documents
│   ├── language_model.py    # LLM generation
│   ├── model_loader.py      # Load models
│   └── rag.py               # Main RAG pipeline
├── mixtral-offloading/      # Mixtral model offloading
├── resources/               # Knowledge base files
├── outputs/                 # Kết quả đánh giá
├── config.py                # Cấu hình hệ thống
├── evaluation.py            # Script đánh giá chính
└── config.md                # Chi tiết cấu hình 
│ ├── evaluation.py              # Runs the full RAG pipeline 
│ ├── requirements.txt           # Python dependencies 
│
├── resources/                   # Knowledge base
│ ├── articles_l3.pkl            # Knowledge base file (level 3)
│ ├── articles_l4.pkl            # Knowledge base file (level 4)
└── README.md 
```


## Run RAG System 
To evaluate our RAG system with different configurations, simply run:

```bash
python evaluation.py
```


## Citation
If you find our paper or code helpful, please cite our paper:
```
@inproceedings{li-etal-2025-enhancing-retrieval,
    title = "Enhancing Retrieval-Augmented Generation: A Study of Best Practices",
    author = "Li, Siran  and
      Stenzel, Linus  and
      Eickhoff, Carsten  and
      Bahrainian, Seyed Ali",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.449/",
    pages = "6705--6717"
}

```