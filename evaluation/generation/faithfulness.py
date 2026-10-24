# evaluate_faithfulness.py

```python
import json
import re
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# =========================
# NLI Faithfulness Scorer
# =========================
class FaithfulnessScorer:

    def __init__(self,
                 model_name="microsoft/deberta-xlarge-mnli",
                 device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, premises, hypotheses):

        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)

        # entailment label index = 2
        return probs[:, 2].tolist()


# =========================
# Sentence Splitter
# =========================
def split_sentences(text):

    text = text.replace("\n", " ")

    sentences = re.split(r"[。！？!?；;]", text)

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# =========================
# Faithfulness Evaluation
# =========================
def evaluate_faithfulness(
        input_dir,
        model_name,
        batch_size):

    scorer = FaithfulnessScorer(
        model_name=model_name
    )

    overall_scores = {}

    for fname in os.listdir(input_dir):

        if not fname.endswith(".jsonl"):
            continue

        input_file = os.path.join(input_dir, fname)

        file_scores = []

        with open(input_file, "r", encoding="utf-8") as f:

            for line in tqdm(
                    f,
                    desc=f"Faithfulness | {fname}"):

                try:
                    data = json.loads(line)

                except Exception:
                    continue

                answer = data.get("answer", "")
                docs = data.get("doc_list", [])

                if not answer or not docs:
                    continue

                sentences = split_sentences(answer)

                sentence_scores = []

                for sent in sentences:

                    premises = []
                    hypotheses = []

                    for doc in docs:

                        premises.append(doc)
                        hypotheses.append(sent)

                        if len(premises) == batch_size:

                            scores = scorer.score_pairs(
                                premises,
                                hypotheses
                            )

                            sentence_scores.extend(scores)

                            premises = []
                            hypotheses = []

                    # tail batch
                    if premises:

                        scores = scorer.score_pairs(
                            premises,
                            hypotheses
                        )

                        sentence_scores.extend(scores)

                if sentence_scores:

                    faithfulness = (
                        sum(sentence_scores)
                        / len(sentence_scores)
                    )

                    file_scores.append(faithfulness)

        avg_score = (
            sum(file_scores) / len(file_scores)
            if file_scores else 0.0
        )

        overall_scores[fname] = {
            "num_samples": len(file_scores),
            "avg_faithfulness": avg_score
        }

        print(
            f"[OK] {fname} | "
            f"samples={len(file_scores)} | "
            f"avg={avg_score:.4f}"
        )

    # =========================
    # Overall Summary
    # =========================
    all_vals = [
        v["avg_faithfulness"]
        for v in overall_scores.values()
        if v["num_samples"] > 0
    ]

    overall_avg = (
        sum(all_vals) / len(all_vals)
        if all_vals else 0.0
    )

    print("\n====== Faithfulness Evaluation ======")

    print(f"处理文件数: {len(overall_scores)}")

    print(f"整体平均 Faithfulness: {overall_avg:.4f}")


# =========================
# Main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="input jsonl directory"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="NLI model name"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size"
    )

    args = parser.parse_args()

    evaluate_faithfulness(
        input_dir=args.input_dir,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
