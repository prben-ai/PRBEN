

# Data Format

## Query File

Each line in the query file is a JSON object:

```json
{
  "id": "1",
  "query": "best running shoes",
  "long_history": [
    "nike shoes",
    "sportswear brands"
  ],
  "top_5_long_history": [
    "marathon training",
    "running socks"
  ],
  "province": "Beijing",
  "age": "25",
  "gender": "男"
}
```

---

## Retrieval File

```json
{
  "query": "best running shoes",
  "docs": [
    "doc_1",
    "doc_2",
    "doc_3"
  ]
}
```

---

## Document File

```json
{
  "id": "doc_1",
  "contents": "Running shoes with strong cushioning..."
}
```

---

# Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

# Build Personalized RAG Prompts

Run the following command:

```bash
python scripts/build_prompt.py \
  --query_file data/queries/test.jsonl \
  --doc_rank_file data/retrieval/retrieval_results.jsonl \
  --doc_dir data/docs \
  --output_file data/outputs/prompt_data.jsonl
```

The generated prompts will be saved to:

```text
data/outputs/prompt_data.jsonl
```

---

# Output Format

Each output sample is formatted as:

```json
{
  "system": "",
  "src": [
    "prompt text ..."
  ]
}
```

---
You can easily enable or disable different personalized features (e.g., user profile, search history, behavioral signals) by simply commenting or uncommenting the corresponding code sections. The top-5 historical records can be obtained using your own retrieval strategy or semantic similarity methods, and the generated prompts can then be applied to any target large language model for evaluation.
