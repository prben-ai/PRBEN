## Retrieval

Dense retrieval with **BGE-M3** (CLS pooling + L2 normalize → cosine similarity):

- `bge_retriever.py` — query encoding + top-k search over precomputed corpus embeddings

### 1. Build the dense index (once per corpus)

```bash
python scripts/build_bge_index.py data/corpus/docs.jsonl 0 data/corpus/bge_m3_emb
```

- `docs.jsonl`: `{"id": "doc_182", "contents": "..."}` per line
- Output: `shard_XX_emb.npy` (float16, N×1024) + `shard_XX_ids.txt` (docid per line),
  sharded at 100K docs per shard, resumable

### 2. Retrieve

```bash
python codes/retrieval/bge_retriever.py \
  --model_dir models/bge-m3 \
  --emb_dir data/corpus/bge_m3_emb \
  --input_dir data/queries/rewrite_qwen \
  --output_dir data/retrieval_results/rewrite_qwen \
  --topk 100 --gpu 0
```

- `--model_dir`: local path to BAAI/bge-m3 (same encoder must be used for
  corpus and queries)
- `--query_field`: field to retrieve on — `query` (original) or `rewrite` (personalized)

### Input Query Format

```json
{"id": "1001", "query": "适合女生的机械键盘推荐"}
```

Fields:

- `id`: Query identifier
- `query`: Query text

### Retrieval Output Format

```json
[
  {
    "query": "1001",
    "1": "doc_182||0.842",
    "2": "doc_991||0.791",
    "3": "doc_204||0.773"
  }
]
```

Fields:

- `query`: Query identifier
- `1`, `2`, ...: Retrieved document rank
- `doc_id||score`: Retrieved document id and cosine similarity score

### Corpus Format

```json
{"id": "doc_182", "contents": "机械键盘通常分为红轴、茶轴和青轴..."}
```

---

# Personalized Query Rewrite Prompt Construction

(unchanged — see `retrieval_prompt.py`; constructs personalized rewrite prompts
from user history / top-5 relevant history / profile)
