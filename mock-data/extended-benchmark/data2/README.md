# Multi-Task Classification Benchmark

本目录包含一组用于 Harness 鲁棒性评测的多任务数据，按任务拆分为训练集与测试集。

## 数据结构

每条样本为一行 JSON，文件格式为 JSONL。当前样本字段如下：

- `text`: 输入文本
- `label`: 样本标签

## 文件清单

| 任务目录 | 文件 | 样本数 |
| --- | --- | ---: |
| `task1` | `train.jsonl` | 300 |
| `task1` | `test.jsonl` | 685 |
| `task2_education` | `train.jsonl` | 116 |
| `task2_education` | `test.jsonl` | 249 |
| `task2_restaurant` | `train.jsonl` | 116 |
| `task2_restaurant` | `test.jsonl` | 249 |
| `task2_techsupport` | `train.jsonl` | 140 |
| `task2_techsupport` | `test.jsonl` | 299 |
| `task3` | `train.jsonl` | 287 |
| `task3` | `test.jsonl` | 383 |

## 校验

所有 `.jsonl` 文件均已按 UTF-8 逐行校验，可解析为 JSON。
