import os
import json
import argparse

from glob import glob
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="BGE-M3 dense retrieval (personalized / rewritten queries)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/bge-m3",
        help="Path to local BGE-M3 model (HuggingFace BAAI/bge-m3)"
    )

    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Directory containing precomputed corpus embeddings "
             "(shard_XX_emb.npy + shard_XX_ids.txt)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing query jsonl files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save retrieval results"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Top-k retrieval results"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Query encoding batch size"
    )

    parser.add_argument(
        "--query_field",
        type=str,
        default="query",
        help="Field to retrieve on ('query' for original, "
             "'rewrite' / model output field for personalized)"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value"
    )

    return parser.parse_args()


def encode(texts, model, tok, bs):
    """BGE-M3 query encoding: CLS pooling + L2 normalize (cosine/IP)."""

    outs = []

    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            enc = tok(
                texts[s:s+bs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            emb = model(**enc).last_hidden_state[:, 0]

            outs.append(
                torch.nn.functional.normalize(emb, p=2, dim=1)
            )

    return torch.cat(outs, dim=0)


def main():

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # =========================
    # Load corpus embeddings
    # =========================
    corpus_ids = []
    mats = []

    emb_files = sorted(
        glob(os.path.join(args.emb_dir, "shard_*_emb.npy")),
        key=lambda p: int(
            os.path.basename(p).split("_")[1]
        )
    )

    for ef in emb_files:
        shard = os.path.basename(ef).split("_")[1]

        ids_file = os.path.join(
            args.emb_dir,
            f"shard_{shard}_ids.txt"
        )

        with open(ids_file, "r", encoding="utf-8") as f:
            corpus_ids.extend(
                line.strip()
                for line in f
                if line.strip()
            )

        mats.append(np.load(ef, mmap_mode="r"))

    mat = torch.from_numpy(
        np.ascontiguousarray(
            np.concatenate(mats, axis=0)
        )
    ).cuda()

    print(f"Corpus: {len(corpus_ids)} docs, "
          f"dim={mat.shape[1]}")

    # =========================
    # Load BGE-M3 model
    # =========================
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    model = AutoModel.from_pretrained(
        args.model_dir,
        dtype=torch.float16
    ).cuda().eval()

    # =========================
    # Create output directory
    # =========================
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================
    # Load all jsonl files
    # =========================
    input_files = glob(os.path.join(args.input_dir, "*.jsonl"))

    print(f"Found {len(input_files)} files")

    # =========================
    # Process each file
    # =========================
    for input_file in input_files:

        print(f"\nProcessing: {input_file}")

        retrieval_results = []

        file_name = os.path.basename(input_file)

        output_file = os.path.join(
            args.output_dir,
            file_name.replace(".jsonl", ".json")
        )

        # =========================
        # Read jsonl
        # =========================
        with open(input_file, "r", encoding="utf-8") as f:

            rows = []

            for line in tqdm(f):

                if not line.strip():
                    continue

                try:
                    item = json.loads(line.strip())

                    rows.append(item)

                except Exception as e:
                    print(f"ERROR: {e}")

        # =========================
        # Batch encode + search
        # =========================
        for s in tqdm(range(0, len(rows), args.batch_size)):

            batch = rows[s:s+args.batch_size]

            texts = [
                item.get(args.query_field, "").strip()
                for item in batch
            ]

            keep = [i for i, t in enumerate(texts) if t]

            if not keep:
                continue

            q_emb = encode(
                [texts[i] for i in keep],
                model,
                tok,
                args.batch_size
            )

            scores = q_emb @ mat.T

            ts, ti = torch.topk(
                scores,
                k=min(args.topk, scores.shape[1]),
                dim=1
            )

            ti = ti.cpu().tolist()
            ts = ts.float().cpu().tolist()

            for j, i in enumerate(keep):

                result_item = {
                    "query": str(batch[i].get("id", "")).strip()
                }

                for rank in range(len(ti[j])):

                    result_item[str(rank + 1)] = (
                        f"{corpus_ids[ti[j][rank]]}"
                        f"||{round(ts[j][rank], 6)}"
                    )

                retrieval_results.append(result_item)

        # =========================
        # Save output
        # =========================
        with open(output_file, "w", encoding="utf-8") as fw:

            json.dump(
                retrieval_results,
                fw,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved to: {output_file}")
        print(f"Total queries: {len(retrieval_results)}")

    print("\nDone")


if __name__ == "__main__":
    main()
