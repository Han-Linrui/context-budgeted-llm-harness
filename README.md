<h1 align="center">Context-Budgeted LLM Harness for Robust Classification</h1>

<h3 align="center">Retrieval, prompt-budget control, task routing, and exact-match parsing for LLM classification</h3>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  LLM Harness | Context Engineering | Dynamic Retrieval | Structured Output | Robust Evaluation
</p>

---

This repository studies a compact inference-time adaptation setting for LLM classification. Given labeled examples and a frozen OpenAI-compatible model, the harness builds a controlled layer around the model call: it stores examples, retrieves relevant demonstrations at prediction time, keeps prompts within a fixed token budget, routes between task formats, and parses the model response into an exact-match label.

The project is organized as a small-scale AI systems experiment. The controllable components are retrieval policy, prompt-budget management, task framing, and output validation; the evaluation checks local accuracy as well as robustness under out-of-domain labels, multiple-choice formats, bilingual inputs, and domain shifts.

## Technical Framing

The design is motivated by three practical questions:

- Can external memory and retrieval provide useful adaptation when model weights are fixed?
- How much reliability is gained by treating the prompt budget as a first-class constraint?
- How should a generative chat response be constrained and parsed for exact-match classification?

Together, these components form a compact loop from method design to implementation, evaluation, and error analysis.

## Engineering Highlights

- **External example memory**: `update(text, label)` records labeled examples in memory and maintains the observed label space.
- **Hybrid retrieval**: word-level tokens and character 3-grams are combined to retrieve useful few-shot examples for short, noisy, and multilingual inputs.
- **Runtime task routing**: the harness detects option-like label spaces and switches between intent classification and multiple-choice prompting.
- **Prompt-budget control**: retrieved examples are added only while the prompt remains within the configured token limit, avoiding uncontrolled truncation by the runner.
- **Structured output parsing**: the model is asked for a compact XML-like response, and the parser applies exact, cleaned, substring, and nearest-neighbor fallbacks.
- **Reproducible evaluation scripts**: the root runner covers the local development split, while `scripts/benchmark.py` evaluates additional robustness datasets.

## Core Flow

The core implementation is `MyHarness` in `solution.py`.

```text
update(text, label)
    -> store example
    -> update label statistics

predict(text)
    -> inspect label space
    -> retrieve similar examples
    -> assemble a token-budgeted prompt
    -> call the configured LLM
    -> parse and validate the returned label
```

This structure keeps the model backend replaceable while making the harness behavior easy to inspect and debug.

## Results

With Qwen3-8B in non-thinking mode, the harness reaches about **83% accuracy** on the local DEV split. The extended benchmark covers same-distribution classification, out-of-domain label spaces, multiple-choice tasks, bilingual inputs, and selected vertical domains.

![Extended benchmark results](assets/benchmark-results.png)

The main observations from the experiments are:

- hybrid word + 3-gram retrieval is more stable than purely lexical matching on short or noisy text;
- explicit prompt budgeting reduces failures caused by long examples displacing key instructions;
- structured responses and deterministic parsing reduce exact-match errors from verbose model outputs;
- low scores on one corrupted Chinese MCQ subset are best treated as a data-quality stress case.

Additional experiment notes are available in [docs/exploration_report.md](docs/exploration_report.md).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure an OpenAI-compatible endpoint. The project reads credentials from environment variables; do not commit real keys.

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

Run the local evaluation:

```bash
python run.py --runs 1
```

Run the extended benchmark:

```bash
python scripts/benchmark.py
```

## Repository Structure

```text
.
|-- solution.py          # Core harness implementation
|-- harness_base.py      # Minimal base interface for harness implementations
|-- run.py               # Local evaluation runner
|-- llm_client.py        # OpenAI-compatible client configured by environment variables
|-- requirements.txt
|-- data/                # Local development split
|-- tokenizer/           # Tokenizer used for prompt-budget accounting
|-- mock-data/           # Additional benchmark datasets
|-- scripts/
|   `-- benchmark.py     # Extended benchmark runner
|-- docs/
|   `-- exploration_report.md
`-- assets/
    |-- benchmark-results.png
    `-- benchmark-results-repeat.png
```

## Limitations

This is a focused classification harness, not a general-purpose agent framework. Its behavior still depends on the instruction-following quality of the selected LLM, and the multiple-choice router currently uses a simple label-shape heuristic. The implementation also avoids trained embeddings and task-specific classifiers, keeping adaptation in the retrieval, prompting, and parsing layer.

## Safety

API keys should be supplied through environment variables. `.env` files are ignored by default, and `.env.example` documents the expected configuration names.
