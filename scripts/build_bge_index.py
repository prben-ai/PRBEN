#!/usr/bin/env python3
"""Build BGE-M3 dense index: encode corpus docs to sharded embeddings.

  {OUT}/shard_{N}_emb.npy   float16, shape (shard_size, 1024)
  {OUT}/shard_{N}_ids.txt   docid per line, aligned with emb rows

Resume: shards already complete (line count matches) are skipped.

Usage:
  python scripts/build_bge_index.py <docs.jsonl> [GPU_ID] [OUT_DIR]
docs.jsonl: {"id": "doc_1", "contents": "..."} per line
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("BGE_MODEL_DIR", "/home/gyz/models/bge-m3")
SHARD = 100_000
BATCH = 256
MAXLEN = 512
PRETRUNC = 2000


def iter_docs(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            yield str(r["id"]), (r.get("contents") or r.get("text") or r.get("doc") or "")


def encode(model, tok, texts):
    outs = []
    with torch.inference_mode():
        for s in range(0, len(texts), BATCH):
            enc = tok(texts[s:s+BATCH], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(model.device)
            emb = model(**enc).last_hidden_state[:, 0]
            outs.append(torch.nn.functional.normalize(emb, p=2, dim=1).to(torch.float16).cpu())
    return torch.cat(outs, dim=0).numpy()


def main():
    docs_path = sys.argv[1]
    gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "data/corpus/bge_m3_emb"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    docs = list(iter_docs(docs_path))
    print(f"total docs: {len(docs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

    n_shards = (len(docs) + SHARD - 1) // SHARD
    for si in range(n_shards):
        emb_f, ids_f = OUT / f"shard_{si}_emb.npy", OUT / f"shard_{si}_ids.txt"
        chunk = docs[si*SHARD:(si+1)*SHARD]
        if emb_f.exists() and ids_f.exists():
            if sum(1 for _ in open(ids_f)) == len(chunk):
                print(f"[skip] shard {si} complete", flush=True); continue
        t0 = time.time()
        ids = [d[0] for d in chunk]
        texts = [d[1][:PRETRUNC] if d[1] else d[0] for d in chunk]
        emb = encode(model, tok, texts)
        np.save(emb_f, emb)
        with open(ids_f, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        rate = len(chunk) / (time.time() - t0)
        eta = (n_shards - si - 1) * SHARD / rate / 3600
        print(f"[done] shard {si}/{n_shards}: {len(chunk)} docs, "
              f"{rate:.0f} docs/s, ETA {eta:.1f}h", flush=True)
    print("ALL EMBEDDINGS DONE", flush=True)


if __name__ == "__main__":
    main()
