# Extended Robustness Benchmark Data

This directory contains additional datasets used to evaluate the LLM harness beyond the local development split. The tasks cover multilingual classification, out-of-domain label spaces, multiple-choice inputs, and selected vertical domains.

## Layout

### `data1` - classification variants

Includes development-style classification, multiple-choice, out-of-domain labels, larger label spaces, and Chinese variants.

Example:

```bash
python run.py \
  --train mock-data/extended-benchmark/data1/train_dev.jsonl \
  --dev mock-data/extended-benchmark/data1/test_dev.jsonl \
  --workers 20 \
  --runs 1
```

### `data2` - multi-task classification and QA

Includes base classification, education, restaurant, tech-support, and complex multiple-choice tasks.

### `data3` - multi-domain and mixed bilingual tasks

Includes domain splits for finance, HR, logistics, policy, security, and software, plus mixed bilingual aggregate tasks.

Run the multi-task helper:

```bash
python mock-data/extended-benchmark/data3/multi_task_eval.py \
  --runner run.py \
  --workers 20 \
  --runs 1
```

## Format

Each dataset file is JSONL with two fields:

```json
{"text": "...", "label": "..."}
```

All labels in each test split are present in the corresponding training split unless the task is explicitly designed as an out-of-domain label-space stress test.
