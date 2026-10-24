import os
import json
import argparse

from glob import glob
from tqdm import tqdm


# =====================================================
# Argument Parser
# =====================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build Personalized RAG Prompts"
    )

    parser.add_argument(
        "--query_file",
        type=str,
        required=True,
        help="Input query jsonl file"
    )

    parser.add_argument(
        "--doc_rank_file",
        type=str,
        required=True,
        help="Retrieved document ranking file"
    )

    parser.add_argument(
        "--doc_dir",
        type=str,
        required=True,
        help="Directory containing document jsonl files"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output prompt file"
    )

    return parser.parse_args()


# =====================================================
# Prompt Template
# =====================================================

PROMPT_TEMPLATE = """
你是一名个性化检索增强智能问答助手（Personalized RAG Assistant）。

你的目标是在严格依赖检索文档内容的前提下，
结合用户搜索历史与用户画像，
生成个性化回答。

【样本ID】{id}

【用户当前问题】
{query}

【检索文档内容（Top-5）】
{retrieved_contents}

【用户近期搜索历史】
{recent_search_history}

【用户长期搜索历史】
{long_term_search_history}

【用户画像信息】
{user_profile}

【任务规则】
1. 回答必须严格基于检索文档
2. 不得编造文档外事实
3. 回答尽量详细
4. 若无法回答则输出“未找到结果”
5. 生成10个个性化关键词

【输出格式】

- 若能回答：
用户个性化需求分析：XXX
ID：{id}
关键词列表：[关键词1, ..., 关键词10]
【个性化回答】：XXX

- 若无法回答：
ID：{id}
关键词列表：[未找到]
检索内容中未找到相关答案。

现在请回答用户问题：{query}
""".strip()


# =====================================================
# Utils
# =====================================================

def load_jsonl(path):

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                yield json.loads(line)


def load_doc_corpus(doc_dir):

    print("Loading document corpus...")

    doc_text = {}

    doc_files = sorted(
        glob(os.path.join(doc_dir, "*.jsonl"))
    )

    for fp in doc_files:

        print(f"Loading: {fp}")

        for item in load_jsonl(fp):

            docid = item.get("id")
            contents = item.get("contents", "")

            if docid:
                doc_text[docid] = contents

    print(f"Loaded {len(doc_text)} documents.")

    return doc_text


# =====================================================
# Main
# =====================================================

def main():

    args = parse_args()

    os.makedirs(
        os.path.dirname(args.output_file),
        exist_ok=True
    )

    # =================================================
    # Load retrieval results
    # =================================================

    print("Loading retrieval results...")

    query_to_docs = {}

    for item in load_jsonl(args.doc_rank_file):

        query = item.get("query")

        if query:
            query_to_docs[query] = item.get(
                "docs",
                []
            )

    print(f"Loaded {len(query_to_docs)} retrieval entries.")

    # =================================================
    # Load document corpus
    # =================================================

    doc_text = load_doc_corpus(args.doc_dir)

    # =================================================
    # User profile format
    # =================================================

    loc_format = "住在{}"
    age_format = "年龄{}"
    gender_format = "{}性"

    # =================================================
    # Build prompts
    # =================================================

    print("Generating prompts...")

    with open(args.query_file, "r", encoding="utf-8") as f_in, \
         open(args.output_file, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in):

            line = line.strip()

            if not line:
                continue

            try:

                data = json.loads(line)

                qid = str(data.get("id"))

                query = data.get(
                    "query",
                    ""
                ).strip()

                if not query:
                    continue

                # =========================================
                # Retrieve Docs
                # =========================================

                if query not in query_to_docs:
                    continue

                docids = query_to_docs[query][:5]

                retrieved_segments = []

                for docid in docids:

                    text = doc_text.get(docid, "")

                    if text:
                        retrieved_segments.append(
                            f"[{docid}]\n{text}"
                        )

                retrieved_block = "\n\n".join(
                    retrieved_segments
                )

                # =========================================
                # User History
                # =========================================

                recent_history = "\n".join(
                    data.get("long_history", [])[1:6]
                ) or "无"

                long_history = "\n".join(
                    data.get("top_5_long_history", [])
                ) or "无"

                # =========================================
                # User Profile
                # =========================================

                province = data.get(
                    "province",
                    ""
                ).strip()

                age = data.get(
                    "age",
                    ""
                ).strip()

                gender = data.get(
                    "gender",
                    ""
                ).strip()

                user_profile = "，".join([
                    loc_format.format(province),
                    age_format.format(age),
                    gender_format.format(gender)
                ])

                # =========================================
                # Build Prompt
                # =========================================

                prompt = PROMPT_TEMPLATE.format(
                    id=qid,
                    query=query,
                    retrieved_contents=retrieved_block,
                    recent_search_history=recent_history,
                    long_term_search_history=long_history,
                    user_profile=user_profile
                )

                output_item = {
                    "system": "",
                    "src": [prompt]
                }

                f_out.write(
                    json.dumps(
                        output_item,
                        ensure_ascii=False
                    ) + "\n"
                )

            except Exception as e:

                print(f"Error: {e}")

    print(f"\nDone!")
    print(f"Saved to: {args.output_file}")


# =====================================================
# Entry
# =====================================================

if __name__ == "__main__":
    main()
