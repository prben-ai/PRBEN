# PRBEN
Personalized RAG in the Wild: Benchmarking Personalized RAG for Web Search with Authentic User Behavioral Signals

## Introduction
We introduce PRBEN, a benchmark for systematically studying personalized RAG with authentic user behavioral signals from real-world Web search. PRBEN is constructed from anonymized user interactions collected over a three-month period from a large-scale commercial search engine serving primarily Chinese-language Web search. The benchmark contains 12,886 evaluation instances with historical queries, click feedback, and coarse-grained user attributes, covering diverse information needs and query types. Based on these naturally occurring interactions, PRBEN constructs personalized retrieval and generation references grounded in user behavior and personalized intent, enabling unified evaluation of personalized retrieval, generation quality, and preference alignment.

<!-- <p align="center">
<img src="https://github.com/user-attachments/assets/3619af74-d323-46da-ad7b-60a0953d1c56" width="600" height="320" />
</p> -->
<!-- <p align="center">
<img src="https://github.com/user-attachments/assets/3619af74-d323-46da-ad7b-60a0953d1c56" width="600" height="320" />
</p> -->
<!-- <p align="center">
<img width="800" height="400" alt="PRBEN" src="https://github.com/user-attachments/assets/312d7c10-32b0-4f99-acec-5211bac7196b" />
</p> -->
<img width="2226" height="1013" alt="PRBEN" src="https://github.com/user-attachments/assets/a85549d5-157d-4807-a57a-040cc0125a5f" />

The dataset is derived from anonymized search logs. To protect user privacy, direct user identifiers are removed, potentially identifying information is removed or masked, and user attributes are retained only in coarse-grained form. We further apply *k*-anonymity (*k*=5) over the released demographic attribute combinations to reduce re-identification risk. Only processed and anonymized data are released; raw search logs and direct user identifiers are not included in the benchmark. Any attempt to re-identify individual users is strictly prohibited.

## Dataset Statistics
The basic statistics of PRBEN are shown as follows:

| Statistic | Value |
|---|---|
| Evaluation instances | 12,886 |
| History query length (range) | 1–822 |
| History query length (average) | 210 |
| Gender distribution | Female / Male |
| Age range | 18–65+ |
| Number of provinces/regions | 32 |
| Clicked URLs | 17,425 |
| Average clicked URLs per query | 1.35 |
| Corpus size | 2,859,860 documents |
| Reference documents per query | 21.13 (average) |

## Project Structure

The PRBEN repository is organized into four major components: retrieval, generation, evaluation, and datasets.  
The `codes/` directory contains the core personalized retrieval and answer generation pipelines.  
The `data/` directory stores the retrieval corpus, user behavior data, evaluation references, and retrieval outputs.  
The `evaluation/` directory provides both retrieval and generation evaluation scripts, including standard retrieval metrics, lexical overlap metrics, personalization alignment metrics, factuality evaluation, and faithfulness scoring.  
Utility scripts such as BM25 index construction are placed under `scripts/`.  
Detailed descriptions, data formats, and usage instructions are provided in the README files inside each subdirectory, allowing users to quickly set up the project and perform personalized retrieval and generation evaluation.

```text
PRBEN/
├── codes/
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   └── retrieval_prompt.py
│   │
│   └── generation/
│       └── generation_prompt.py
│
├── scripts/
│   └── build_bm25_index.sh
│
├── data/
│   ├── corpus/
│   │   ├── download.sh
│   │   └── manifest.json
│   │
│   ├── gold/
│   │   ├── D_star.jsonl
│   │   └── Y_star.jsonl
│   │
│   ├── user_data/
│   │   ├── user_click_url.jsonl
│   │   ├── user_history_part1.jsonl
│   │   ├── user_history_part2.jsonl
│   │   ├── user_history_part3.jsonl
│   │   ├── user_profile.jsonl
│   │   └── user_query.jsonl
│   │
│   └── retrieval_results/
│
├── evaluation/
│   ├── retrieval/
│   │   └── retrieval_val.py
│   │
│   └── generation/
│       ├── BLEU.py
│       ├── PGDC.py
│       ├── PTC.py
│       ├── ROUGEL.py
│       ├── factuality.py
│       └── faithfulness.py
│
├── requirements.txt
└── README.md
```

<!--
## Data Content and Format
The folder PRBEN will contain three files: user_data.jsonl, data_gold_target.jsonl, and prompt.py.

### (1) user_data.jsonl
This file contains the user information, including user_id, query, click_url, long_history_query, gender, age, and province. The format of each line is:

```json
{"user_id": ,"query": ,"click_url": ,"long_history_query": ,"gender": ,"age": ,"province": }
```

- user_id: the anonymized PRBEN ID of the user.
- query: the user's current query.
- click_url: the URLs clicked by the user for the current query.
- long_history_query: the user's long-term historical queries.
- gender: the user's gender.
- age: the user's age range.
- province: the user's location (province).

### (2) data_gold_target.jsonl

This file contains the retrieval reference documents and reference responses for the retrieval and generation stages. The format of each line is:

```json
{"user_id": ,"reference_docs": ,"reference_answer": ,"reference_keywords": }
```

- user_id: the anonymized PRBEN ID of the user.
- reference_docs: the reference document set for retrieval evaluation.
- reference_answer: the reference response for generation evaluation.
- reference_keywords: the keywords associated with the reference response.

### (3) prompt.py
The prompt.py file encapsulates the key prompts employed in both the personalized retrieval and personalized generation stages. These prompts are used to guide the model to:

- Personalized Retrieval – adapt query reformulation based on the user's historical interactions and preferences.
- Personalized Generation – generate responses grounded in retrieved evidence while incorporating the user's intent and personalized context.

## Corpus
The corpus contains 2,859,860 documents collected from real-world Web pages. The documents are stored in the following format:

```json
{"id": ,"contents": }
```

- id: the ID of the document in the text corpus.
- contents: the textual content of the document.

The corpus can be downloaded from the following Hugging Face URL:
https://huggingface.co/datasets/gyz-ai/PRBEN
-->

## Baseline models
We evaluate several open-source and closed-source models on PRBEN. The following presents a subset of the personalized retrieval results.

| Personalized Retrieval | Recall@10 | MRR@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|
| **Non-Personalized Retrieval** | | | | |
| Original Query | 0.2857 | 0.8260 | 0.8214 | 0.5759 |
| Q2D | 0.0877 | 0.3775 | 0.3659 | 0.1293 |
| BART | 0.2772 | 0.8047 | 0.7999 | 0.5210 |
| Gemini-2.5-Flash | 0.2825 | 0.8157 | 0.8112 | 0.5376 |
| DeepSeek-V4.1-Flash | 0.2767 | 0.8036 | 0.7990 | 0.5226 |
| Qwen3-8B | 0.2871 | 0.8180 | 0.8133 | 0.5482 |
| **Long History** | | | | |
| Gemini-2.5-Flash | 0.3073 | 0.8440 | 0.8399 | 0.5928 |
| DeepSeek-V4.1-Flash | 0.3051 | 0.8407 | 0.8368 | 0.5896 |
| Qwen3-8B | 0.2748 | 0.7740 | 0.7690 | 0.5233 |
| **Short History** | | | | |
| Gemini-2.5-Flash | 0.3204 | 0.8667 | 0.8634 | 0.6235 |
| DeepSeek-V4.1-Flash | 0.3158 | 0.8541 | 0.8507 | 0.6128 |
| Qwen3-8B | 0.2823 | 0.7877 | 0.7827 | 0.5402 |
| **Long + Short History** | | | | |
| Gemini-2.5-Flash | 0.3259 | 0.8754 | 0.8723 | 0.6349 |
| DeepSeek-V4.1-Flash | 0.3234 | 0.8671 | 0.8640 | 0.6301 |
| Qwen3-8B | 0.2899 | 0.8054 | 0.8008 | 0.5562 |
| **User Attributes** | | | | |
| Gemini-2.5-Flash | 0.2829 | 0.8071 | 0.8024 | 0.5356 |
| DeepSeek-V4.1-Flash | 0.2630 | 0.7623 | 0.7570 | 0.4944 |
| Qwen3-8B | 0.2638 | 0.7633 | 0.7580 | 0.4970 |

Additionally, we evaluate model performance across the complete end-to-end personalized RAG pipeline. Each configuration follows the format retrieval-generation: N, R, and G denote original-query, rewrite-based personalized, and reference-conditioned retrieval, respectively, while generation N/P denotes non-personalized/personalized generation. The following presents a subset of the results:

| Method | ROUGE-L | BLEU-4 | PTC | PGDC | Faithfulness | Factuality |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B (R-P) | 0.1963 | 0.0869 | 0.1346 | 0.7187 | 0.4981 | 0.7128 |
| DeepSeek-V4.1-Flash (R-P) | 0.1974 | 0.0814 | 0.1937 | 0.7732 | 0.5610 | 0.7753 |
| Gemini-2.5-Flash (R-P) | 0.2111 | 0.1016 | 0.1917 | 0.7510 | 0.5485 | 0.6250 |
| Qwen3-8B (G-P) | 0.2508 | 0.1377 | 0.2420 | 0.9294 | 0.5292 | 0.6978 |
| DeepSeek-V4.1-Flash (G-P) | 0.2794 | 0.1368 | 0.3507 | 0.9847 | 0.5518 | 0.7664 |
| Gemini-2.5-Flash (G-P) | 0.2816 | 0.1729 | 0.3344 | 0.9268 | 0.5312 | 0.6350 |

---

### License
This repository is licensed under the Apache-2.0 License.

The PRBEN benchmark is licensed under CC BY-NC-SA 4.0.

## FQA
