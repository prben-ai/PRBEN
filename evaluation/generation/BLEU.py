import json
import argparse
import jieba

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)


def compute_bleu(reference, hypothesis):

    ref_tokens = list(
        jieba.cut(str(reference))
    )

    hyp_tokens = list(
        jieba.cut(str(hypothesis))
    )

    smoothie = (
        SmoothingFunction()
        .method4
    )

    bleu = sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        smoothing_function=smoothie
    )

    return bleu


def load_answers(file_path):

    id2answer = {}

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                id2answer[data["id"]] = (
                    data.get("answer", "")
                )

            except Exception:
                continue

    return id2answer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ref_file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--hyp_file",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None
    )

    args = parser.parse_args()

    ref_answers = load_answers(
        args.ref_file
    )

    hyp_answers = load_answers(
        args.hyp_file
    )

    bleu_results = []

    for qid, ref_text in ref_answers.items():

        hyp_text = hyp_answers.get(qid, "")

        if (
            not ref_text.strip()
            or not hyp_text.strip()
        ):
            continue

        try:

            bleu = compute_bleu(
                ref_text,
                hyp_text
            )

        except Exception:

            bleu = 0.0

        bleu_results.append({
            "id": qid,
            "bleu": bleu
        })

    avg_bleu = (
        sum(
            x["bleu"]
            for x in bleu_results
        ) / len(bleu_results)
        if bleu_results else 0.0
    )

    print("\n====== BLEU Evaluation ======\n")

    print(f"samples={len(bleu_results)}")
    print(f"avg_bleu={avg_bleu:.4f}")

    if args.output_file:

        with open(
            args.output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                bleu_results,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"\nSaved to: "
            f"{args.output_file}"
        )


if __name__ == "__main__":
    main()
