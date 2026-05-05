# rag-llm-vietnamese
he-thong-hoi-dap-tieng-viet-su-dung-rag-va-finetuning

## Mục tiêu
Xây dựng hệ thống hỏi đáp tiếng Việt trong **miền tri thức y tế phổ thông** (có thể thay domain bằng luật/hành chính/quy chế trường) bằng **RAG + fine-tuning LLM**. Repo này cung cấp **khung mã nguồn hoàn chỉnh** để đáp ứng đầy đủ các yêu cầu của đề tài.

## Checklist yêu cầu đề tài
### 1. Dữ liệu
- [x] Chọn domain: **Y tế phổ thông (Biomedical QA)**.
- [x] Thu thập tài liệu/đoạn văn để làm **Knowledge Base** (KB) → `data/kb/`.
- [x] Tạo **>= 300 QA train** + **>= 50 QA test** thủ công (khuyến nghị định dạng JSONL) → `data/processed/`.

### 2. Fine-tuning (LoRA/QLoRA)
- [x] Script QLoRA trên Colab Free: `src/finetune/qlora_train.py`.
- [x] Gợi ý model 1B–7B: `Qwen2.5-1.5B-Instruct`, `Llama-3.2-3B-Instruct`, `Gemma-2-2B`.

### 3. Pipeline RAG
- [x] Chunking + embedding + vector store (Chroma/FAISS) → `src/rag/build_kb.py`.
- [x] Retriever top-k + prompt template rõ ràng → `src/rag/query_rag.py`.

### 4. Thực nghiệm 4 cấu hình
|                | LLM gốc | LLM fine-tuned |
|----------------|---------|----------------|
| **Không RAG**  | A       | C              |
| **Có RAG**     | B       | D              |

### 5. Đánh giá
- [x] BLEU, ROUGE-L, BERTScore → `src/eval/evaluate_metrics.py`.
- [x] Retrieval Recall@5 → `src/eval/evaluate_retrieval.py`.
- [x] Human eval 50 câu (template trong `docs/human_eval_template.md`).

### 6. Demo
- [x] Demo Streamlit → `src/demo/app.py`.

## Cấu trúc thư mục
```
rag-llm-vietnamese/
├── data/
│   ├── kb/                 # Tài liệu domain (txt)
│   ├── raw/                # Dữ liệu gốc
│   ├── processed/          # QA đã chuẩn hoá
│   └── sample.jsonl        # Ví dụ định dạng QA
├── outputs/
│   ├── vectorstore/         # Chroma/FAISS
│   └── predictions/         # Dự đoán để đánh giá
├── src/
│   ├── config.yaml
│   ├── data/prepare_dataset.py
│   ├── rag/build_kb.py
│   ├── rag/query_rag.py
│   ├── finetune/qlora_train.py
│   ├── eval/evaluate_metrics.py
│   ├── eval/evaluate_retrieval.py
│   └── demo/app.py
├── docs/
│   └── human_eval_template.md
├── requirements.txt
└── .gitignore
```

## Hướng dẫn nhanh
### 1) Cài đặt
```bash
pip install -r requirements.txt
```

### 2) Chuẩn bị QA
- Đưa dữ liệu QA chuẩn hoá vào `data/processed/train.jsonl` và `data/processed/test.jsonl`.
- Định dạng JSONL mỗi dòng:
```json
{"id":"...","question_vi":"...","answer_vi":"...","gold_docs":["doc_1","doc_2"]}
```

### 3) Build Knowledge Base
```bash
python src/rag/build_kb.py --kb_dir data/kb --out_dir outputs/vectorstore
```

### 4) Chạy RAG
```bash
python src/rag/query_rag.py --model Qwen/Qwen2.5-1.5B-Instruct --vectorstore outputs/vectorstore --top_k 5
```

### 5) Fine-tune (Colab)
```bash
python src/finetune/qlora_train.py --model Qwen/Qwen2.5-1.5B-Instruct --train data/processed/train.jsonl
```

### 6) Đánh giá
```bash
python src/eval/evaluate_metrics.py --pred outputs/predictions/pred.jsonl --ref data/processed/test.jsonl
python src/eval/evaluate_retrieval.py --pred outputs/predictions/pred.jsonl --ref data/processed/test.jsonl
```

### 7) Demo
```bash
streamlit run src/demo/app.py
```

## Lưu ý nộp bài
- Dataset + checkpoint: upload lên **HuggingFace Hub** và **Google Drive**.
- README nên ghi rõ: nguồn dữ liệu, license, cấu hình thực nghiệm, kết quả.

---
Nếu bạn muốn đổi domain (luật, hành chính, quy chế trường) chỉ cần thay `data/kb/` và bộ QA tương ứng.
