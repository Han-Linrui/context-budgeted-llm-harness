# Token Budget Notes

- Individual `text` fields are intentionally short.
- Approximate maximum text length is recorded in `manifest.json`.
- A full prompt containing all labels and all train examples for the 180-label OOD task may exceed 2000 tokens.
- For a strict 2000-token budget, use candidate-label retrieval and only a few examples per candidate label.
