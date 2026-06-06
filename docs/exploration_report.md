# Exploration Report: Lightweight LLM Harness under a Token Budget

This report summarizes the design choices behind `solution.py`. The implementation studies how a frozen LLM can be guided by an external harness to perform few-shot text classification and natural-language multiple-choice prediction under a strict prompt budget.

## 1. Problem Setting

The harness exposes a small interface:

- `update(text, label)`: receive labeled training examples and store external memory.
- `predict(text)`: predict an exact-match label for an unlabeled input.
- `call_llm(messages)`: query a frozen OpenAI-compatible model.
- `count_messages_tokens(messages)`: estimate prompt length before the API call.

The main constraint is that each prompt must stay within `max_prompt_tokens = 2048`. Since the training stream may contain many examples, naively concatenating all memory items is both inefficient and brittle. The harness therefore needs to decide which examples to retrieve, how to format them, and how to parse the model output reliably.

## 2. Method

### 2.1 Hybrid Dynamic Retrieval

For each test input, the harness scores all stored examples by a lightweight similarity function:

- word-level features capture explicit lexical overlap;
- character 3-gram features improve tolerance to short texts, spelling variation, and noisy abbreviations;
- multiset Jaccard similarity ranks candidate examples without external dependencies.

The top retrieved examples are used as few-shot demonstrations. This keeps the prompt compact while still exposing the model to label-specific evidence.

### 2.2 Label-Space Task Routing

The benchmark suite includes both intent classification and multiple-choice question answering. Instead of hard-coding option names, the harness inspects the label space at runtime. If all labels are short strings, such as `A`, `B`, `C`, or `D`, it switches to a multiple-choice prompt. Otherwise, it uses an intent-classification prompt.

This rule is intentionally simple: it avoids overfitting to one benchmark format while giving the LLM a better task framing for long natural-language questions.

### 2.3 Structured Generation and Output Parsing

The prompt asks the model to emit:

```xml
<thought>brief reasoning</thought>
<label>final label</label>
```

The parser then extracts the content inside `<label>`. If the model violates the format, the harness falls back through exact match, cleaned match, substring match, and finally the label of the most similar retrieved example. This design makes `predict()` return a valid label even when the LLM output is noisy.

### 2.4 Token Budget Control

The harness starts from the top retrieved examples and removes the lowest-ranked demonstration until the assembled prompt fits the token budget. This explicit check reduces accidental truncation by the local runner and leaves room for longer OOD or MCQ inputs.

## 3. Experimental Observations

The local DEV set contains 231 training examples, 539 test examples, and 77 labels. The current implementation reached about 83% accuracy on this split under the Qwen3-8B non-thinking setting.

An extended benchmark was also run on community/mock datasets covering same-distribution classification, OOD labels, MCQ tasks, bilingual inputs, and selected vertical domains:

![Extended benchmark results](../assets/benchmark-results.png)

Several design observations came from this process:

- Hybrid word + 3-gram retrieval was more robust than a hand-written BM25 variant on short texts in the observed DEV setting.
- Increasing the number of few-shot examples could slightly improve local DEV accuracy, but it also raised truncation and latency risk. A moderate top-15 retrieval setting offered a better robustness trade-off.
- The Chinese MCQ subset performed near random baseline. Manual inspection suggested that this subset contained severe machine-translation artifacts and weak semantic consistency, so it should be treated as a data-quality stress case rather than a clean reasoning benchmark.

## 4. Limitations

This harness is not a general-purpose agent framework. It is a compact evaluation harness for classification-style tasks under a fixed API and token budget. Its main limitations are:

- dependence on the stability and instruction-following behavior of the underlying LLM;
- a heuristic MCQ detector based on label length;
- no learned embedding model or trained classifier, by design;
- no guarantee on heavily corrupted or semantically invalid datasets.

## 5. Takeaway

The project demonstrates a practical engineering pattern for test-time LLM adaptation: keep the model frozen, move task-specific learning into controlled external memory, retrieve compact evidence under a token budget, constrain the output schema, and use deterministic post-processing to recover exact-match labels. This is a small but complete example of harness-level design for robustness and reproducibility.
