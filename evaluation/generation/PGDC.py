import json
import os
import argparse


PROMPT_TEMPLATE = """你是一名个性化文本生成评测专家。

你的任务是评估生成答案与个性化标准答案之间的
【个性化生成方向一致性（Personalized Generation Direction Consistency，PGDC）】。

定义：
PGDC 用于衡量生成答案在以下方面是否与个性化标准答案保持一致：
1）用户意图是否对齐
2）关注重点是否一致
3）用户偏好取向是否被正确体现
4）是否存在明显的主题偏移或用户特定信息缺失

你将获得两段文本：
- 个性化标准答案（针对特定用户编写）
- 模型生成的答案

评测要求：
- 请基于整体一致性进行评分，而非逐句对比。
- 不要考虑事实正确性，仅关注个性化方向是否一致。

评分标准（1–5 分）：
5 分：高度一致。生成答案在意图、关注重点和用户偏好取向上与标准答案几乎完全一致，没有明显偏移或个性化信息缺失。
4 分：基本一致。整体个性化方向正确，仅存在轻微表述差异或次要强调不同。
3 分：部分一致。生成答案在部分意图或关注点上存在对齐，但出现明显偏移或遗漏部分用户偏好。
2 分：一致性较弱。生成答案仅弱相关于标准答案，存在明显主题偏移或较多用户偏好缺失。
1 分：不一致。生成答案与标准答案在个性化意图、关注重点和偏好取向上基本不匹配。

重要说明：
- 不要判断事实是否正确。
- 不要给出任何解释或分析。
- 只输出评分结果。

输出格式（仅限 JSON）:
{{"PGDC_score": 1-5 的整数}}

个性化标准答案：
{y_gold}

生成答案：
{y_generated}
"""


def load_answers(file_path):
    """
    读取 jsonl 文件：id -> answer
    """
    id2answer = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                qid = data.get("id")
                answer = data.get("answer", "")

                if qid is not None:
                    id2answer[qid] = answer

            except Exception as e:
                print(f"[WARN] {file_path} 第 {line_num} 行解析失败: {e}")

    return id2answer


def build_pgdc_prompts(gold_file, input_dir, output_dir):
    """
    构建 PGDC prompts
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载 gold answers
    gold_answers = load_answers(gold_file)

    total_files = 0
    total_prompts = 0

    # 遍历生成答案文件
    for fname in os.listdir(input_dir):

        if not fname.endswith(".jsonl"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        gen_answers = load_answers(input_path)

        cnt = 0

        with open(output_path, "w", encoding="utf-8") as fout:

            for qid, y_gold in gold_answers.items():

                if qid not in gen_answers:
                    continue

                y_generated = gen_answers[qid]

                if not y_gold.strip() or not y_generated.strip():
                    continue

                prompt = PROMPT_TEMPLATE.format(
                    y_gold=y_gold.strip(),
                    y_generated=y_generated.strip()
                )

                fout.write(json.dumps({
                    "system": "",
                    "src": [prompt]
                }, ensure_ascii=False) + "\n")

                cnt += 1

        print(f"[OK] {fname} -> 生成 PGDC prompts: {cnt}")

        total_files += 1
        total_prompts += cnt

    print("\n========== 完成 ==========")
    print(f"处理文件数: {total_files}")
    print(f"生成 PGDC prompt 总数: {total_prompts}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="gold answer jsonl file"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="generated answer directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output prompt directory"
    )

    args = parser.parse_args()

    build_pgdc_prompts(
        gold_file=args.gold_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
