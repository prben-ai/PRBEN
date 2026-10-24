import json
import argparse
import jieba
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer


class SpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()


def compute_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
        tokenizer=SpaceTokenizer()
    )

    ref = " ".join(jieba.cut(reference))
    hyp = " ".join(jieba.cut(hypothesis))

    scores = scorer.score(ref, hyp)
    return scores["rougeL"]


def load_answers(file_path):
    id2answer = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                id2answer[data["id"]] = data.get("answer", "")
            except Exception:
                continue

    return id2answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--hyp_file", type=str, required=True)

    args = parser.parse_args()

    ref_answers = load_answers(args.ref_file)
    hyp_answers = load_answers(args.hyp_file)

    precisions, recalls, f1s = [], [], []

    for qid, ref_answer in ref_answers.items():
        if qid not in hyp_answers:
            continue

        hyp_answer = hyp_answers[qid]

        if not ref_answer.strip() or not hyp_answer.strip():
            continue

        rouge_l = compute_rouge_l(ref_answer, hyp_answer)

        precisions.append(rouge_l.precision)
        recalls.append(rouge_l.recall)
        f1s.append(rouge_l.fmeasure)

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    print(f"Evaluation Samples: {len(f1s)}")
    print(f"Average ROUGE-L Precision: {avg_p:.6f}")
    print(f"Average ROUGE-L Recall:    {avg_r:.6f}")
    print(f"Average ROUGE-L F1:        {avg_f1:.6f}")


if __name__ == "__main__":
    main()
