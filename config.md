# Tài liệu cấu hình và chỉ số đánh giá RAG

## 1. Chỉ số đánh giá (Evaluation Metrics)

### 1.1 Chỉ số ROUGE
Được tính toán trong hàm `mean_metrics_item()` tại [evaluation.py](evaluation.py#L106-L117):

- **r1f1**: ROUGE-1 F1 Score - Đo lường sự trùng khớp của unigram (từ đơn) giữa câu trả lời sinh ra và câu trả lời đúng
- **r2f1**: ROUGE-2 F1 Score - Đo lường sự trùng khớp của bigram (cặp từ liên tiếp) 
- **rLf1**: ROUGE-L F1 Score - Đo lường chuỗi con chung dài nhất (Longest Common Subsequence)

### 1.2 Chỉ số Semantic Similarity
- **similarity**: Độ tương đồng ngữ nghĩa giữa câu trả lời sinh ra và câu trả lời đúng

### 1.3 Chỉ số MAUVE
- **mauve**: Đo lường khoảng cách giữa phân phối văn bản sinh ra và văn bản tham chiếu

### 1.4 Chỉ số thời gian (Timing Metrics)
Được thu thập trong [evaluation.py](evaluation.py#L145-L150):

- **model_load_time**: Thời gian tải các model (generation và seq2seq)
- **rag_init_time**: Thời gian khởi tạo hệ thống RAG (bao gồm xây dựng index)
- **evaluation_time**: Thời gian đánh giá trên toàn bộ dataset test
- **total_time**: Tổng thời gian cho một cấu hình

## 2. Cấu hình cơ bản (Base Configuration)

### 2.1 Models
```python
generation_model_name: "mistralai/Mistral-7B-Instruct-v0.2"
embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"
seq2seq_model_name: "google/flan-t5-small"
is_chat_model: True
instruct_tokens: ("[INST]", "[/INST]")
```

### 2.2 Index Builder Parameters
```python
tokenizer_model_name: None  # Mặc định dùng generation_model_name
chunk_size: 64              # Kích thước chunk văn bản
overlap: 8                  # Số token chồng lấn giữa các chunk
passes: 10                  # Số lượt pass khi xây dựng index
icl_kb: False              # Sử dụng knowledge base cho ICL
multi_lingo: False         # Hỗ trợ đa ngôn ngữ
```

### 2.3 RALM Parameters
```python
expand_query: False         # Mở rộng query
top_k_docs: 1              # Số lượng document retrieve
top_k_titles: 3            # Số lượng title retrieve
stride: -1                 # Stride cho sliding window
query_len: 200             # Độ dài query
do_sample: False           # Sampling khi generate
temperature: 1.0           # Temperature cho sampling
top_p: 0.1                 # Nucleus sampling parameter
num_beams: 1               # Beam search width
max_new_tokens: 25         # Số token tối đa sinh ra
batch_size: 1              # Batch size cho inference
kb_10K: False              # Sử dụng knowledge base 10K
icl_kb: False              # ICL từ knowledge base
icl_kb_incorrect: False    # Bao gồm câu trả lời sai trong ICL
focus: False               # Sử dụng focus mechanism
hybrid_kb: False           # Kết hợp nhiều knowledge base
top_k_icl: 0               # Số lượng ICL examples
```

### 2.4 System Prompts
```python
system_prompt: "You are a truthful expert question-answering bot and should correctly and concisely answer the following question"
repeat_system_prompt: True  # Lặp lại system prompt
```

## 3. Các cấu hình thử nghiệm (Test Suite Configurations)

### 3.1 Baseline (1_Baseline)
- Cấu hình RAG cơ bản không có tính năng đặc biệt
- Sử dụng các tham số mặc định từ base_config

### 3.2 ExpandQuery Only (2_ExpandQuery_Only)
- **expand_query**: True
- **top_k_docs**: 3
- **top_k_titles**: 5
- Mở rộng query để retrieve nhiều document hơn

### 3.3 Focus Only (3_Focus_Only)
- **top_k_docs**: 10
- **focus**: 3
- Retrieve nhiều document và focus vào top 3

### 3.4 ICL Only (4_ICL_Only)
- **chunk_size**: 200
- **overlap**: 0
- **icl_kb**: True (cả index_builder và ralm)
- **top_k_docs**: 2
- **icl_kb_incorrect**: False
- Sử dụng In-Context Learning với examples từ knowledge base

### 3.5 ExpandQuery + Focus (5_ExpandQuery_Focus)
- **expand_query**: True
- **top_k_docs**: 10
- **top_k_titles**: 5
- **focus**: 3
- Kết hợp query expansion và focus mechanism

### 3.6 Focus + ICL (6_Focus_ICL)
- **chunk_size**: 200
- **overlap**: 0
- **icl_kb**: True
- **top_k_docs**: 10
- **focus**: 3
- **icl_kb_incorrect**: False
- Kết hợp Focus và ICL

### 3.7 Hybrid All Features (7_Hybrid_All_Features)
- **hybrid_kb**: True
- **chunk_size**: 64
- **overlap**: 8
- **top_k_icl**: 2
- **top_k_docs**: 10
- **focus**: 3
- **expand_query**: True
- **top_k_titles**: 5
- **icl_kb_incorrect**: False
- Kết hợp tất cả các tính năng với hybrid knowledge base

## 4. Các cấu hình Run 1

### 4.1 Base
- Giống với baseline configuration

### 4.2 HelpV2
- Thay đổi system prompt:
  ```
  "You are an accurate and reliable question-answering bot. Please provide a precise and correct response to the question following"
  ```

### 4.3 Instruct45B
- **generation_model_name**: "mistralai/Mixtral-8x7B-Instruct-v0.1"
- **top_k_docs**: 2
- **batch_size**: 4
- **repeat_system_prompt**: True
- Sử dụng model lớn hơn (Mixtral-8x7B)

## 5. Các cấu hình Run 2

### 5.1 ICL1D+
- **chunk_size**: 200
- **overlap**: 0
- **icl_kb**: True
- **top_k_docs**: 1
- **icl_kb_incorrect**: True
- ICL với 1 document và bao gồm cả incorrect answers

### 5.2 Focus80_Doc80
- **top_k_docs**: 80
- **repeat_system_prompt**: True
- **focus**: 80
- Retrieve và focus vào 80 documents

### 5.3 Hybrid_ICL2_Doc3_Focus
- **hybrid_kb**: True
- **chunk_size**: 64
- **overlap**: 8
- **top_k_icl**: 2
- **top_k_docs**: 10
- **focus**: 3
- **expand_query**: True
- **top_k_titles**: 5
- **icl_kb_incorrect**: True
- Hybrid với ICL, focus và expand query, bao gồm incorrect answers

## 6. Datasets

### 6.1 TruthfulQA
- Split: validation
- Columns: question, best_answer, correct_answers, incorrect_answers
- Filtering: Chỉ giữ câu hỏi có > 1 correct_answers và > 1 incorrect_answers

### 6.2 MMLU
- Source: "cais/mmlu", "all"
- Split: test
- Samples: 32 câu hỏi đầu tiên từ mỗi subject
- Columns: question, best_answer, correct_answers, incorrect_answers

## 7. Arguments và Options

### 7.1 Command Line Arguments
- `--dataset`: Dataset để đánh giá ('truthfulqa' hoặc 'mmlu')
- `--output-dir`: Thư mục output (mặc định: 'outputs')
- `--seed`: Random seed (mặc định: 42)
- `--quant`: Loại quantization ('4bit', '8bit', hoặc None)
- `--num-samples`: Số lượng samples để test nhanh
- `--config-set`: Tập config để chạy ('test_suite', 'run1', 'run2')

### 7.2 Quantization Options
- **4bit**: 4-bit quantization để giảm memory footprint
- **8bit**: 8-bit quantization
- **None**: Không quantization (full precision)

## 8. Output Structure

### 8.1 Files được tạo ra cho mỗi configuration
```
outputs/{dataset}/run{run}_{timestamp}/
├── config_{name}.json              # Configuration đã dùng
├── evaluation_{name}.pkl           # Kết quả đánh giá chi tiết
├── eval_results_{name}.json        # Metrics trung bình
├── eval_results_all.json           # Tổng hợp tất cả configs
└── timing_summary.json             # Tổng hợp timing cho tất cả configs
```

### 8.2 eval_results_{name}.json structure
```json
{
    "r1f1": float,
    "r2f1": float,
    "rLf1": float,
    "similarity": float,
    "mauve": float,
    "timing": {
        "model_load_time": float,
        "rag_init_time": float,
        "evaluation_time": float,
        "total_time": float
    }
}
```

## 9. Knowledge Bases

### 9.1 Default Knowledge Base
- File: `resources/articles_l3.pkl`
- Sử dụng cho hầu hết các cấu hình

### 9.2 Extended Knowledge Base (10K)
- File: `resources/articles_l4.pkl`
- Được sử dụng khi `kb_10K: True`

### 9.3 ICL Knowledge Base
- Sử dụng test_data làm knowledge base khi `icl_kb: True`

### 9.4 Hybrid Knowledge Base
- Kết hợp giữa articles knowledge base và ICL examples
- Được kích hoạt khi `hybrid_kb: True`
