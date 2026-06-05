# Mixed Bilingual Multi-Task Benchmark

This package recreates four benchmark tasks aligned with the expected hidden-test structure:

1. Same DEV-label classification with different mixed Chinese/English text.
2. OOD multi-domain classification with different labels and domains.
3. Complex natural-language multiple-choice tasks with labels A/B/C/D.
4. Project-friendly MCQ task (ratio 3:7) for classification and routing evaluation.

## Scale

| Task | Train | Test | Labels | Notes |
|---|---:|---:|---:|---|
| task1_same_dev_labels_mixed_bilingual | 231 | 539 | 77 | 3 train + 7 test per label |
| task2_ood_180_labels_multidomain_mixed_bilingual | 540 | 1260 | 180 | 3 train + 7 test per label |
| task3_complex_mcq_mixed_bilingual | 180 | 420 | 4 | 45 train + 105 test per option |
| task4_project_friendly_mcq_mixed_bilingual_ratio_3_7 | 180 | 420 | 4 | 45 train + 105 test per option, ratio 3:7 |

Total rows: 3770

## Format

Each file is JSONL with exactly two fields:

```json
{"text": "...", "label": "..."}
```

All test labels appear in the corresponding train split. Some test rows contain prompt-injection text; the injection text is part of the data and should be ignored by a robust classifier.
