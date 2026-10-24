## Factuality Evaluation

Run factual correctness prompt generation:

```bash
python evaluation/build_fact_prompts.py \
  --input_dir data/answers \
  --output_dir data/fact_prompts
```

### Input Answer Format

Input answers are stored in `.jsonl` format.

Example:

```json
{"id": "prben_id_00001", "answer": "Paris is the capital of France."}
{"id": "prben_id_00002", "answer": "The Earth revolves around the Sun."}
```

Fields:

- `id`: Sample identifier
- `answer`: Generated answer to be evaluated

---

### Output Prompt Format

Generated factuality prompts are stored in `.jsonl` format.

Example:

```json
{
  "system": "",
  "src": [
    "你是一名严格的事实核查专家（Automated Fact Checker）..."
  ]
}
```

Fields:

- `system`: System prompt
- `src`: Prompt list for factual correctness evaluation

---

### Factual Correctness Scoring

The evaluator model outputs a continuous score between `0.0` and `1.0`.

Example:

```json
{"factual_correctness": 0.92}
```

Score interpretation:

- `1.0`: Highly factual and reliable
- `0.0`: Severely incorrect or hallucinated

---
---


## PGDC Evaluation

Run PGDC prompt generation:

```bash
python evaluation/build_pgdc_prompts.py \
  --gold_file data/gold/gold_answers.jsonl \
  --input_dir data/answers \
  --output_dir data/pgdc_prompts
```

### Gold Answer Format

Gold personalized answers are stored in `.jsonl` format.

Example:

```json
{"id": "prben_id_00001", "answer": "Personalized gold answer."}
{"id": "prben_id_00002", "answer": "Another personalized gold answer."}
```

Fields:

- `id`: Sample identifier
- `answer`: Personalized reference answer

---

### Generated Answer Format

Generated answers are stored in `.jsonl` format.

Example:

```json
{"id": "prben_id_00001", "answer": "Model generated answer."}
{"id": "prben_id_00002", "answer": "Another generated answer."}
```

Fields:

- `id`: Sample identifier
- `answer`: Model generated answer

---

### Output Prompt Format

Generated PGDC prompts are stored in `.jsonl` format.

Example:

```json
{
  "system": "",
  "src": [
    "你是一名个性化文本生成评测专家..."
  ]
}
```

Fields:

- `system`: System prompt
- `src`: Prompt list for PGDC evaluation

---

### PGDC Scoring

The evaluator model outputs an integer score between `1` and `5`.

Example:

```json
{"PGDC_score": 5}
```

Score interpretation:

- `5`: Highly consistent with personalized generation direction
- `1`: Completely inconsistent with personalized intent and preference

---
---

## Faithfulness Evaluation

Run faithfulness evaluation:

```bash
python evaluation/evaluate_faithfulness.py \
  --input_dir data/answers \
  --model_name microsoft/deberta-xlarge-mnli \
  --batch_size 8
```

### Input Format

Input files are stored in `.jsonl` format.

Example:

```json
{
  "id": "prben_id_00001",
  "answer": "Paris is the capital of France.",
  "doc_list": [
    "Paris is the capital city of France.",
    "France is located in Europe."
  ]
}
```

Fields:

- `id`: Sample identifier
- `answer`: Generated answer
- `doc_list`: Retrieved supporting documents

---

### Faithfulness Scoring

The script computes sentence-level entailment scores between generated answers and retrieved documents using an NLI model.

Default model:

```text
microsoft/deberta-xlarge-mnli
```

The final faithfulness score is computed as:

```text
Average entailment probability across all answer-document sentence pairs
```

---

### Output Example

```text
[OK] qwen.jsonl | samples=12000 | avg=0.8421
[OK] gemini.jsonl | samples=12000 | avg=0.8574

====== Faithfulness Evaluation ======
处理文件数: 2
整体平均 Faithfulness: 0.8497
```
---
---
## PTC Evaluation

Run keyword hit rate evaluation:

```bash
python evaluation/keyword_hit_rate.py \
  --gold_file data/gold/gold_answers.jsonl \
  --input_dir data/generated_answers
```

### Gold File Format

```json
{
  "id": "prben_id_00001",
  "keywords": [
    "生肖",
    "春秋笔法",
    "鼠"
  ]
}
```

Fields:

- `id`: Sample identifier
- `keywords`: Personalized keyword list

---

### Generated Answer Format

```json
{
  "id": "prben_id_00001",
  "answer": "春秋笔法通常对应鼠生肖。"
}
```

Fields:

- `id`: Sample identifier
- `answer`: Generated answer

---

### Metric

```text
HitRate = matched_keywords / total_keywords
```

Score range:

- `1.0`: All keywords matched
- `0.0`: No keywords matched

---

### Example Output

```text
rewrite_qwen.jsonl:
samples=12000, avg_hit_rate=0.7821
```

---
---

## BLEU Evaluation

Run BLEU evaluation:

```bash
python evaluation/bleu_eval.py \
  --ref_file data/gold/gold_answers.jsonl \
  --hyp_file data/generated_answers/rewrite_qwen.jsonl \
  --output_file results/bleu_scores.json
```

### Reference File Format

```json
{
  "id": "prben_id_00001",
  "answer": "春秋笔法对应鼠生肖。"
}
```

### Generated Answer Format

```json
{
  "id": "prben_id_00001",
  "answer": "春秋笔法通常对应鼠。"
}
```

Fields:

- `id`: Sample identifier
- `answer`: Generated answer text

---

### Metric

BLEU is computed using:

- Word-level BLEU
- `jieba` Chinese tokenization
- NLTK sentence BLEU with smoothing

Score range:

- `1.0`: Perfect overlap
- `0.0`: No overlap

---

### Example Output

```text
====== BLEU Evaluation ======

samples=12000
avg_bleu=0.1721
```
---
---
## ROUGE-L Evaluation

Run ROUGE-L evaluation:

```bash
python evaluation/compute_rouge.py \
  --ref_file data/gold_answers.jsonl \
  --hyp_file data/generated_answers.jsonl
```

---

### Reference / Hypothesis Format

Both files use `.jsonl` format.

Example:

```json
{"id": "1001", "answer": "巴黎是法国的首都。"}
{"id": "1002", "answer": "地球围绕太阳公转。"}
```

Fields:

- `id`: Sample identifier
- `answer`: Answer text

---

### Output Example

```text
Evaluation Samples: 12000
Average ROUGE-L Precision: 0.301245
Average ROUGE-L Recall:    0.287631
Average ROUGE-L F1:        0.294118
```

---

### Metric Description

The script computes:

- ROUGE-L Precision
- ROUGE-L Recall
- ROUGE-L F1

Chinese text is segmented using `jieba` before evaluation.
