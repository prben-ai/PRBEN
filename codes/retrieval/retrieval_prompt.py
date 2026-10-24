import os
import json
import argparse

from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build Personalized Rewrite Prompts"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input query jsonl file"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output prompt jsonl file"
    )

    parser.add_argument(
        "--id_prompt_output_file",
        type=str,
        required=True,
        help="Output id-prompt mapping file"
    )

    return parser.parse_args()


PROMPT_TEMPLATE = """
你是一名专业的搜索查询个性化改写助手。

你的核心目标是：
结合用户搜索行为、长期兴趣偏好与用户画像，
对当前搜索 query 进行个性化改写，
生成更符合用户真实搜索意图的 query。

--------------------------------------------------

【改写原则】

1. 改写前需先进行用户意图推理：
   - 分析用户近期搜索行为
   - 分析长期兴趣方向
   - 结合用户画像信息
   - 判断当前 query 的真实语义指向

2. 改写应体现明确的信息增益：
   - 不允许简单同义词替换
   - 不允许机械堆砌关键词
   - 不允许生成冗长 query
   - 改写后的 query 必须语义完整自然

3. 若 query 不存在明显歧义，
   或用户信息无法提供有效增益，
   则保持原 query 不变。

--------------------------------------------------

【示例】

示例1：
用户历史：
大众、特斯拉、比亚迪

当前query：
小米

改写：
小米汽车

示例2：
用户历史：
红豆、高粱、玉米种植

当前query：
小米

改写：
小米农作物

--------------------------------------------------

【输入信息】

用户近期搜索历史：
{}

用户长期搜索历史：
{}

用户画像信息：
{}

用户当前搜索query：
{}

--------------------------------------------------

【输出格式】

严格按照以下格式输出：

用户个性化需求分析：XXX
【改写结果：XXX】

--------------------------------------------------

【要求】

- 个性化需求分析需说明：
  - 用户兴趣方向
  - 当前 query 是否存在歧义
  - 为什么需要改写

- 改写结果必须：
  - 简洁
  - 自然
  - 完整
  - 具有明确个性化信息增益

- 不允许输出额外内容
""".strip()


def main():

    args = parse_args()

    os.makedirs(
        os.path.dirname(args.output_file),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(args.id_prompt_output_file),
        exist_ok=True
    )

    loc_format = "住在{}"
    age_format = "年龄{}"
    gender_format = "{}性"

    with open(args.input_file, "r", encoding="utf-8") as f_in, \
         open(args.output_file, "w", encoding="utf-8") as f_out, \
         open(args.id_prompt_output_file, "w", encoding="utf-8") as f_id:

        for line in tqdm(f_in):

            line = line.strip()

            if not line:
                continue

            try:

                data = json.loads(line)

                sample_id = data.get("id", "")

                query = data.get(
                    "query",
                    ""
                ).strip()

                if not query:
                    continue

                recent_history = data.get(
                    "long_history",
                    []
                )[1:6]

                recent_history_str = (
                    "\n".join(recent_history)
                    if recent_history else "无"
                )

                long_history = data.get(
                    "top_5_long_history",
                    []
                )

                long_history_str = (
                    "\n".join(long_history)
                    if long_history else "无"
                )

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

                user_profile_str = "，".join([
                    loc_format.format(province),
                    age_format.format(age),
                    gender_format.format(gender)
                ])

                prompt_text = PROMPT_TEMPLATE.format(
                    recent_history_str,
                    long_history_str,
                    user_profile_str,
                    query
                )

                output_item = {
                    "system": "",
                    "src": [prompt_text]
                }

                f_out.write(
                    json.dumps(
                        output_item,
                        ensure_ascii=False
                    ) + "\n"
                )

                id_prompt_item = {
                    "id": sample_id,
                    "prompt": prompt_text
                }

                f_id.write(
                    json.dumps(
                        id_prompt_item,
                        ensure_ascii=False
                    ) + "\n"
                )

            except Exception as e:

                print(f"Error: {e}")

    print("\nDone!")
    print(f"Prompt File: {args.output_file}")
    print(f"ID-Prompt File: {args.id_prompt_output_file}")


if __name__ == "__main__":
    main()
