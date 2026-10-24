import os
import json
import math
import argparse

from tqdm import tqdm


################################
# Metric Functions
################################

def recall_at_k(relevant_docs, retrieved_docs, k):

    if not relevant_docs:
        return 0.0

    relevant = set(relevant_docs)
    retrieved_k = set(retrieved_docs[:k])

    return len(relevant & retrieved_k) / len(relevant)


def mrr_at_k(relevant_docs, retrieved_docs, k):

    for rank, doc_id in enumerate(
        retrieved_docs[:k],
        start=1
    ):

        if doc_id in relevant_docs:
            return 1.0 / rank

    return 0.0


def dcg_at_k(relevant_docs, retrieved_docs, k):

    dcg = 0.0

    for i, doc_id in enumerate(retrieved_docs[:k]):

        if doc_id in relevant_docs:
            dcg += 1 / math.log2(i + 2)

    return dcg


def idcg_at_k(relevant_docs, k):

    ideal_hits = min(len(relevant_docs), k)

    return sum(
        1 / math.log2(i + 2)
        for i in range(ideal_hits)
    )


def ndcg_at_k(relevant_docs, retrieved_docs, k):

    dcg = dcg_at_k(
        relevant_docs,
        retrieved_docs,
        k
    )

    idcg = idcg_at_k(
        relevant_docs,
        k
    )

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(relevant_docs, retrieved_docs):

    relevant_docs = set(relevant_docs)

    hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(retrieved_docs):

        if doc_id in relevant_docs:

            hits += 1

            precision_at_i = hits / (i + 1)

            sum_precisions += precision_at_i

    if not relevant_docs:
        return 0.0

    return sum_precisions / len(relevant_docs)


################################
# Evaluation
################################

def evaluate_query_list(
    qrels,
    retrieved,
    query_list,
    k
):

    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_ap = 0.0

    count = 0

    for query in query_list:

        if query not in qrels:
            continue

        if query not in retrieved:
            continue

        rel_docs = qrels[query]

        ret_docs = retrieved[query]

        total_recall += recall_at_k(
            rel_docs,
            ret_docs,
            k
        )

        total_mrr += mrr_at_k(
            rel_docs,
            ret_docs,
            k
        )

        total_ndcg += ndcg_at_k(
            rel_docs,
            ret_docs,
            k
        )

        total_ap += average_precision(
            rel_docs,
            ret_docs
        )

        count += 1

    if count == 0:
        return None

    return {
        f"Recall@{k}": round(total_recall / count, 4),
        f"MRR@{k}": round(total_mrr / count, 4),
        f"NDCG@{k}": round(total_ndcg / count, 4),
        "MAP": round(total_ap / count, 4),
        "QueryCount": count
    }


################################
# Argument Parser
################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="Gold relevance file"
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing retrieval outputs"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )

    return parser.parse_args()


################################
# Main
################################

def main():

    args = parse_args()

    ################################
    # Build qrels
    ################################

    qrels = {}

    with open(
        args.gold_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            query = str(item["query"])

            gold_doc_list = item.get(
                "gold_doc_list",
                []
            )

            qrels[query] = gold_doc_list

    print(f"Loaded {len(qrels)} queries")

    ################################
    # Evaluate all retrieval files
    ################################

    all_results = {}

    for filename in tqdm(os.listdir(args.result_dir)):

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(
            args.result_dir,
            filename
        )

        print(f"\nProcessing: {filename}")

        retrieved = {}

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            input_data = json.load(f)

            for item in input_data:

                query = str(item["query"])

                doc_list = []

                for k, v in item.items():

                    if k == "query":
                        continue

                    doc_id = v.split("||")[0]

                    doc_list.append(doc_id)

                retrieved[query] = doc_list

        query_list = list(qrels.keys())

        result_k5 = evaluate_query_list(
            qrels,
            retrieved,
            query_list,
            k=5
        )

        result_k10 = evaluate_query_list(
            qrels,
            retrieved,
            query_list,
            k=10
        )

        all_results[filename] = {
            "k=5": result_k5,
            "k=10": result_k10
        }

    ################################
    # Save results
    ################################

    os.makedirs(
        os.path.dirname(args.output_file),
        exist_ok=True
    )

    with open(
        args.output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n===== FINAL RESULTS =====\n")

    print(
        json.dumps(
            all_results,
            indent=4,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
