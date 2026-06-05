# Lightweight LLM Harness for Robust Few-Shot Classification

[English](README.md) | [中文](README.zh-CN.md)

This repository presents a lightweight LLM harness for robust few-shot classification under a constrained context window. The project was originally developed on top of the 2026 SII Harness Engineering assessment scaffold, where the official interface requires a frozen LLM to infer labels from a streaming set of labeled examples. The main contribution of this repository is the design and implementation of `MyHarness` in `solution.py`, together with a cleaned public configuration, extended benchmarks, and reproducible documentation.

## Research Motivation

Modern LLM applications often rely less on parameter updates and more on the external harness surrounding the model: prompt construction, memory retrieval, context budgeting, output parsing, fallback control, and injection resistance. This project studies a compact but representative setting:

> Given a frozen Qwen3-8B model and a prompt budget of 2048 tokens, how can an external harness perform reliable exact-match classification from a small labeled training stream?

The problem is closely related to test-time adaptation, LLM evaluation, and agent/harness engineering. The model remains fixed; all task-specific adaptation occurs through external memory and deterministic control logic.

## Method Overview

`MyHarness` combines four mechanisms:

1. **Hybrid dynamic retrieval**: ranks stored training examples using multiset Jaccard similarity over word-level tokens and character 3-grams, improving robustness on short and noisy texts.
2. **Label-space task routing**: detects whether the current task is standard intent classification or multiple-choice reasoning by inspecting the runtime label space.
3. **Structured LLM generation**: asks the model to produce a compact XML-like response with `<thought>` and `<label>` fields, then extracts the final label deterministically.
4. **Token-budgeted context assembly**: removes lower-ranked demonstrations until the prompt fits the budget, with exact-match, cleaned-match, substring-match, and nearest-neighbor fallback paths.

The solution intentionally avoids heavyweight external dependencies such as sklearn, torch, or embedding indexes, matching the original assessment constraints while keeping the design transparent and auditable.

## Repository Structure

```text
.
├── solution.py                 # Main contribution: retrieval, routing, prompting, and parsing logic
├── harness_base.py             # Official harness base class
├── run.py                      # Official local evaluation entrypoint
├── llm_client.py               # OpenAI-compatible client configured through environment variables
├── requirements.txt
├── data/                       # Official DEV data: train_dev / test_dev
├── tokenizer/                  # Local tokenizer for prompt-budget accounting
├── mock-data/                  # Community/mock datasets for extended evaluation
├── scripts/
│   └── benchmark.py            # Extended benchmark runner
├── docs/
│   ├── assignment-spec-2026-summer.pdf
│   └── exploration_report.md
└── assets/
    ├── benchmark-results.png
    └── benchmark-results-repeat.png
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure an OpenAI-compatible endpoint. Copy `.env.example` locally if desired, but do not commit real credentials.

PowerShell:

```powershell
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_API_KEY="your-api-key"
$env:LLM_MODEL="qwen3-8b"
$env:LLM_ENABLE_THINKING="false"
```

Bash:

```bash
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="qwen3-8b"
export LLM_ENABLE_THINKING="false"
```

Run the official DEV evaluation:

```bash
python run.py --runs 1
```

Run the extended benchmark suite:

```bash
python scripts/benchmark.py
```

## Results

The official DEV split contains 231 training examples, 539 test examples, and 77 labels. Under the Qwen3-8B non-thinking setting, this harness obtains approximately 83% accuracy on the local DEV split. The extended benchmark covers same-distribution classification, out-of-domain labels, multiple-choice tasks, bilingual inputs, and selected vertical domains.

![Extended benchmark results](assets/benchmark-results.png)

See [docs/exploration_report.md](docs/exploration_report.md) for design details, ablation observations, and data-quality analysis.

## Attribution and Contribution Boundary

The repository deliberately preserves the original evaluation layout so that the assessment contract remains reproducible and the contribution boundary is clear.

- Provided by the original assessment scaffold: `harness_base.py`, `run.py`, `data/`, `tokenizer/`, and the assignment specification.
- Implemented or organized in this repository: the harness strategy in `solution.py`, environment-based API configuration in `llm_client.py`, the extended benchmark runner, experiment documentation, and repository-level README files.
- Extended data: `mock-data/` contains public community datasets, mock assessment data, and partially self-constructed examples. See [mock-data/SII-26Summer-HE-Data-main/README.md](mock-data/SII-26Summer-HE-Data-main/README.md) for source notes.

## Engineering Rationale

The project is not refactored into a large Python package because the official evaluation environment is defined around root-level `solution.py` and `run.py`. For a research-facing demonstration, preserving that contract improves reproducibility and makes the official scaffold versus personal implementation boundary explicit. The engineering work is therefore focused on safe configuration, clean documentation, reproducible scripts, and experimental traceability rather than superficial restructuring.

## Safety and Privacy

The public repository should not contain private application materials, real API keys, or personal correspondence. API configuration is read from environment variables, and `.gitignore` excludes local `.env` files and private application documents. If a real key has appeared in Git history, rotate the key and clean the history before publishing the repository.
