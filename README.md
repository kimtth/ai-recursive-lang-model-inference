# Recursive Language Models Demo

This is a small educational implementation of the [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) using Azure OpenAI.

The idea is simple: instead of putting a huge document directly into the model prompt, the app loads the document into a Python REPL. The model can then inspect the document with code, split it into chunks, ask smaller LLM questions about selected chunks, and combine the results.

This project is meant for junior developers and IT ops teams who want to understand the concept. It is intentionally not a production framework.

## What You Can Learn

- How a long document can live in a Python variable instead of the model prompt.
- How the model can use code to search, count, slice, and inspect that document.
- How recursive sub-calls can ask smaller questions about selected chunks.
- How the REPL keeps variables between turns so the model can build an answer step by step.

The REPL is not sandboxed. Run it only in a trusted learning environment.

## Setup

Create a `.env` file with Azure OpenAI settings:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=your-root-deployment
AZURE_OPENAI_SUB_DEPLOYMENT=your-sub-deployment
```

`AZURE_OPENAI_SUB_DEPLOYMENT` is optional. If omitted, recursive calls use the root deployment.

## Run

Ask a question about the cached RLM paper:

```bash
python main.py "List the benchmark tasks used to evaluate RLMs."
```

Use your own context file:

```bash
python main.py --context notes.txt "What are the main unresolved risks?"
```

Turn off recursive LLM calls and use only the Python REPL:

```bash
python main.py --no-subcalls "Count the paper authors."
```

The only command-line options are `--context` and `--no-subcalls`. Learning settings like `CHUNK_SIZE`, `MAX_TURNS`, and `MAX_SUBCALLS` are constants near the top of [main.py](main.py).

## How It Works

1. [main.py](main.py) loads the context text.
2. `RecursiveLanguageModel` stores it as `context_text` and splits it into `context_chunks`.
3. The root model writes fenced `repl` code blocks.
4. The app executes those blocks and sends printed output back to the model.
5. The model finishes with `FINAL(...)` or `FINAL_VAR(final_answer)`.

## REPL Example

The root LM is instructed to execute code like this:

```repl
print(context_text[:1000])
matches = search_context("RLM with REPL", max_matches=5)
print(matches)
```

For semantic chunking, it can call a sub-LM:

```repl
answers = query_chunks("Which models were evaluated?")
final_answer = llm_query("Aggregate these answers:\n" + "\n".join(answers))
```

It must finish outside code with one of:

```text
FINAL(the answer text)
FINAL_VAR(final_answer)
```

The full trace is written to `OUTPUT.md` by default.