import json
import os
import argparse


def keyword_hit_rate(text, keyword_list):
    hit_keywords = [
        kw for kw in keyword_list
        if kw in text
    ]

    hit_rate = (
        len(hit_keywords) / len(keyword_list)
        if keyword_list else 0.0
    )

    return hit_rate


def load_gold_keywords(gold_file):

    id2keywords = {}

    with open(gold_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                id2keywords[data["id"]] = (
                    data.get("keywords", [])
                )

            except Exception:
                continue

    return id2keywords


def compute_hit_score_for_file(
    input_file,
    id2keywords
):

    scores = []

    with open(input_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                continue

            qid = data.get("id")
            answer = data.get("answer", "")

            if qid not in id2keywords:
                continue

            keywords = id2keywords[qid]

            hit_rate = keyword_hit_rate(
                answer,
                keywords
            )

            scores.append(hit_rate)

    avg_hit_rate = (
        sum(scores) / len(scores)
        if scores else 0.0
    )

    return {
        "file": os.path.basename(input_file),
        "num_samples": len(scores),
        "avg_hit_rate": avg_hit_rate
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    id2keywords = load_gold_keywords(
        args.gold_file
    )

    results = []

    for filename in os.listdir(args.input_dir):

        if not filename.endswith(".jsonl"):
            continue

        file_path = os.path.join(
            args.input_dir,
            filename
        )

        result = compute_hit_score_for_file(
            file_path,
            id2keywords
        )

        results.append(result)

    print("\n====== Keyword Hit Rate ======\n")

    for res in results:

        print(
            f"{res['file']}: "
            f"samples={res['num_samples']}, "
            f"avg_hit_rate="
            f"{res['avg_hit_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
