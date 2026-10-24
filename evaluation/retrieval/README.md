## Evaluation

Run retrieval evaluation:

```bash
python evaluation/evaluate.py \
  --gold_file data/gold/test_gold.jsonl \
  --result_dir data/retrieval_results/rewrite_qwen \
  --output_file results/eval_results.json
```

### Gold File Format

The gold relevance annotations are stored in `.jsonl` format.

Example:

```json
{"query": "1001", "gold_doc_list": ["doc_1", "doc_5"]}
{"query": "1002", "gold_doc_list": ["doc_3", "doc_8"]}
```

Fields:

- `query`: Query identifier
- `gold_doc_list`: List of relevant document ids

---

### Retrieval Result Format

Retrieval outputs are stored in `.json` format.

Example:

```json
[
  {
    "query": "1001",
    "1": "doc_5||13.2",
    "2": "doc_9||12.7",
    "3": "doc_2||11.4"
  },
  {
    "query": "1002",
    "1": "doc_3||15.1",
    "2": "doc_7||13.8"
  }
]
```

Fields:

- `query`: Query identifier
- `1`, `2`, ... : Retrieved rank position
- `doc_id||score`: Retrieved document id and retrieval score

---

### Output Metrics

The evaluation script reports:

- Recall@5
- Recall@10
- MRR@5
- MRR@10
- NDCG@5
- NDCG@10
- MAP

Example output:

```json
{
  "rewrite_qwen.json": {
    "k=5": {
      "Recall@5": 0.1794,
      "MRR@5": 0.5106,
      "NDCG@5": 0.3748,
      "MAP": 0.2317,
      "QueryCount": 12000
    },
    "k=10": {
      "Recall@10": 0.2972,
      "MRR@10": 0.5227,
      "NDCG@10": 0.3272,
      "MAP": 0.2317,
      "QueryCount": 12000
    }
  }
}
```

