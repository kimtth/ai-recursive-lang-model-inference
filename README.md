# Recursive Language Models (RLM)

Minimal implementation of [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) using Azure OpenAI.

## Concept

RLMs treat long prompts as external environment variables that can be:
- Programmatically examined via Python REPL
- Decomposed and processed in chunks
- Queried recursively via sub-LM calls

## Run

```bash
python main.py
```

## How It Works

1. Load context as variable in Python REPL
2. LM writes code to explore context
3. LM calls itself recursively on sub-tasks
4. Returns final answer

## Sample

- OUTPUT.md