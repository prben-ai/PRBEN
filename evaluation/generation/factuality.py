import json
import os
import argparse
from tqdm import tqdm


FACT_PROMPT_TEMPLATE = """你是一名严格的事实核查专家（Automated Fact Checker）。

你的任务是评估以下文本在整体层面上的【事实准确性（Factual Correctness）】。

【定义】
事实准确性用于衡量文本是否在整体上符合客观事实与常识，包括但不限于：
1）是否包含明显的事实性错误
2）是否存在虚构、不可靠或高度可疑的信息
3）是否存在自相矛盾或逻辑不一致的事实陈述
4）在缺乏明确证据的情况下，是否给出了过度确定的断言

【评测说明】
- 请基于你的一般世界知识与常识进行判断
- 不需要外部检索或引用证据
- 即使文本表面上看起来合理，只要存在明显“幻觉式编造”，也应降低评分
- 若文本内容主要为主观观点或泛化描述，应重点判断其是否错误或误导

【评分方式】
- 给出一个 0 到 1 之间的连续分值：
  - 1.0：文本在整体上高度事实准确、可靠
  - 0.0：文本在整体上严重不准确或充满事实错误

【重要约束】
- 不要输出任何解释、理由或分析
- 不要修改或复述原文本
- 只输出 JSON，不得包含多余字符

【输出格式（严格遵守）】
{{"factual_correctness": 0.0 到 1.0 之间的小数}}

【待评估文本】
{answer}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate factual correctness evaluation prompts"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input jsonl files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated prompts"
    )

    return parser.parse_args()


def load_answers(file_path):
    """
    读取 jsonl:
    {
        "id": xxx,
        "answer": xxx
    }
    """

    id2answer = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                sample_id = data.get("id", "")
                answer = data.get("answer", "")

                if sample_id:
                    id2answer[sample_id] = answer

            except Exception as e:
                print(f"[WARN] Skip {file_path} line {line_num}: {e}")

    return id2answer


def build_prompt(answer):
    return FACT_PROMPT_TEMPLATE.format(
        answer=answer.strip()
    )


def process_file(input_path, output_path):

    answers = load_answers(input_path)

    count = 0

    with open(output_path, "w", encoding="utf-8") as fout:

        for qid, answer in answers.items():

            if not answer.strip():
                continue

            prompt = build_prompt(answer)

            output_data = {
                "system": "",
                "src": [prompt]
            }

            fout.write(
                json.dumps(output_data, ensure_ascii=False) + "\n"
            )

            count += 1

    return count


def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_files = 0
    total_prompts = 0

    files = sorted(os.listdir(args.input_dir))

    for fname in tqdm(files):

        if not fname.endswith(".jsonl"):
            continue

        input_path = os.path.join(args.input_dir, fname)

        output_name = fname.rsplit(".", 1)[0] + "_fact_prompt.jsonl"

        output_path = os.path.join(
            args.output_dir,
            output_name
        )

        try:

            cnt = process_file(
                input_path=input_path,
                output_path=output_path
            )

            print(f"[OK] {fname} -> {cnt} prompts")

            total_files += 1
            total_prompts += cnt

        except Exception as e:

            print(f"[ERROR] {fname}: {e}")

    print("\n========== DONE ==========")
    print(f"Processed files : {total_files}")
    print(f"Generated prompts : {total_prompts}")
    print(f"Output dir : {args.output_dir}")


if __name__ == "__main__":
    main()
