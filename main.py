import argparse
import ast
import contextlib
import io
import os
import re
import textwrap
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


# Simple settings for the educational demo. Change these in code while learning.
PAPER_URL = "https://arxiv.org/html/2512.24601v1"
PAPER_TEXT_FILE = Path("doc/rlm_paper.txt")
OUTPUT_FILE = Path("OUTPUT.md")
CHUNK_SIZE = 200_000
MAX_TURNS = 8
MAX_SUBCALLS = 20
OBSERVATION_LIMIT = 8_000

log_lines: list[str] = []


def log(message: str = "") -> None:
    """Print a message and remember it for OUTPUT.md."""
    print(message)
    log_lines.append(message)


def save_log() -> None:
    OUTPUT_FILE.write_text("\n".join(log_lines), encoding="utf-8")


def shorten(text: object, limit: int = OBSERVATION_LIMIT) -> str:
    """Keep REPL observations small enough to send back to the model."""
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... <truncated {len(value) - limit} chars>"


class TextExtractor(HTMLParser):
    """Tiny HTML-to-text helper for the arXiv HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"br", "p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "li", "div", "section", "article", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    parser = TextExtractor()
    parser.feed(html)
    text = unescape("".join(parser.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def load_paper_text() -> str:
    """Read the cached paper text, or download it once if missing."""
    if PAPER_TEXT_FILE.exists():
        return PAPER_TEXT_FILE.read_text(encoding="utf-8")

    response = requests.get(PAPER_URL, timeout=60)
    response.raise_for_status()

    text = html_to_text(response.text)
    PAPER_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TEXT_FILE.write_text(text, encoding="utf-8")
    return text


def load_context(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return load_paper_text()


def split_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split long text so a sub-LM can read one piece at a time."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def azure_client() -> AzureOpenAI:
    missing = [
        name
        for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError("Missing Azure OpenAI settings: " + ", ".join(missing))

    # Strip /openai/v1 suffix if present — AzureOpenAI builds this path itself.
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def ask_llm(messages: list[dict[str, str]], deployment: str | None = None) -> str:
    deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Set AZURE_OPENAI_DEPLOYMENT in .env")

    response = azure_client().chat.completions.create(
        model=deployment,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def find_repl_blocks(model_text: str) -> list[str]:
    """Find code blocks written as ```repl ... ``` or ```python ... ```."""
    pattern = re.compile(r"```(?:repl|python)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(model_text)]


def remove_code_blocks(model_text: str) -> str:
    return re.sub(
        r"```(?:repl|python)\s*\n.*?```",
        "",
        model_text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def find_final_answer(model_text: str, repl_variables: dict) -> str | None:
    """Read FINAL(...) or FINAL_VAR(name) from the model response."""
    text = remove_code_blocks(model_text)

    variable_match = re.search(r"FINAL_VAR\s*\(\s*([A-Za-z_]\w*)\s*\)", text)
    if variable_match:
        variable_name = variable_match.group(1)
        return str(repl_variables.get(variable_name, f"Missing variable: {variable_name}"))

    final_match = re.search(r"FINAL\s*\((.*)\)\s*$", text, re.DOTALL)
    if final_match:
        value = final_match.group(1).strip()
        # If the argument is a bare variable name that exists in the REPL, resolve it.
        if re.fullmatch(r"[A-Za-z_]\w*", value) and value in repl_variables:
            return str(repl_variables[value])
        return value

    return None


class RecursiveLanguageModel:
    """RLM: the model reads long context through a Python REPL."""

    def __init__(self, context_text: str, allow_subcalls: bool = True) -> None:
        self.context_text = context_text
        self.context_chunks = split_chunks(context_text)
        self.allow_subcalls = allow_subcalls
        self.subcall_count = 0
        self.repl_variables: dict = {}

    def run(self, question: str) -> str:
        self.subcall_count = 0
        self.repl_variables = self.start_repl(question)

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": question},
        ]

        for step in range(MAX_TURNS):
            model_text = ask_llm(messages)
            log(f"[step {step}] assistant:\n{model_text}\n")

            blocks = find_repl_blocks(model_text)
            observations = self.run_repl_blocks(model_text)

            if blocks:
                answer_val = self.repl_variables.get("answer", "")
                log(f"[step {step}] exec result: Executed. answer={answer_val}\n")
                text = remove_code_blocks(model_text)
                var_match = re.search(r"FINAL_VAR\s*\(\s*([A-Za-z_]\w*)\s*\)", text)
                if var_match:
                    var_name = var_match.group(1)
                    value = self.repl_variables.get(var_name)
                    # Only accept if the variable exists and is non-empty.
                    if value is not None and value != [] and value != "" and value != {}:
                        return str(value)
                answer_val = self.repl_variables.get("answer")
                if answer_val is not None and answer_val != [] and answer_val != "":
                    return str(answer_val)
            else:
                final_answer = find_final_answer(model_text, self.repl_variables)
                if final_answer is not None:
                    return final_answer

            messages.append({"role": "assistant", "content": model_text})
            messages.append({"role": "user", "content": observations or self.next_step_hint()})

        return str(self.repl_variables.get("answer", "No answer found"))

    def system_prompt(self) -> str:
        subcall_tool = ""
        subcall_step = "3. Use sub-LM calls only for semantic reading of selected chunks."
        if self.allow_subcalls:
            subcall_tool = """
            - llm_query(prompt): ask a sub-LM about a selected snippet or buffer.
            - query_chunks(question): ask the same question over every context chunk.
            """
        else:
            subcall_step = "3. Use only Python search, counting, slicing, and aggregation."

        return textwrap.dedent(
            f"""
            You are a Recursive Language Model demo.

            The long context is not in this chat message. It is loaded into a Python
            REPL so you can inspect it with code.

            REPL variables and helpers:
            - context_text: the full context string ({len(self.context_text)} characters).
            - context_chunks: the same context split into {len(self.context_chunks)} chunks.
            - search_context(pattern): regex search with small surrounding snippets.
            - peek(start, chars): view part of context_text.
            {subcall_tool}

            Work pattern:
            1. Run fenced repl code blocks to inspect the context.
            2. Use Python for exact search, counting, slicing, and aggregation.
            {subcall_step}
            4. Store useful results in variables such as final_answer.

            Every single response MUST end with exactly one of these two lines,
            with no text after it:
            FINAL(your answer here)
            FINAL_VAR(variable_name)
            """
        ).strip()

    def start_repl(self, question: str) -> dict:
        helpers = {
            "context_text": self.context_text,
            "context_chunks": self.context_chunks,
            "question": question,
            "peek": self.peek,
            "search_context": self.search_context,
            "re": re,
            "textwrap": textwrap,
        }
        if self.allow_subcalls:
            helpers["llm_query"] = self.llm_query
            helpers["query_chunks"] = self.query_chunks
        return helpers

    def run_repl_blocks(self, model_text: str) -> str:
        observations = []
        for index, code in enumerate(find_repl_blocks(model_text), start=1):
            output = self.run_code(code)
            observations.append(f"REPL block {index} output:\n{output}")
        return "\n\n".join(observations)

    def run_code(self, code: str) -> str:
        """Execute code like a Python REPL: auto-print the last expression's value."""
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    # Run all statements before the last expression.
                    if len(tree.body) > 1:
                        preceding = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(preceding, "<repl>", "exec"), self.repl_variables)
                    # Eval + print the last expression (REPL-style).
                    last = ast.Expression(body=tree.body[-1].value)
                    result = eval(compile(last, "<repl>", "eval"), self.repl_variables)
                    if result is not None:
                        print(repr(result))
                else:
                    exec(code, self.repl_variables)
        except Exception as error:
            return shorten(output.getvalue() + f"\nERROR: {type(error).__name__}: {error}")

        printed = output.getvalue().strip()
        return shorten(printed or "Code ran successfully. No output was printed.")

    def llm_query(self, prompt: str) -> str:
        if self.subcall_count >= MAX_SUBCALLS:
            raise RuntimeError(f"Subcall limit reached: {MAX_SUBCALLS}")

        self.subcall_count += 1
        deployment = os.getenv("AZURE_OPENAI_SUB_DEPLOYMENT") or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT"
        )
        return ask_llm([{"role": "user", "content": prompt}], deployment=deployment)

    def query_chunks(self, question: str) -> list[str]:
        answers = []
        for index, chunk in enumerate(self.context_chunks, start=1):
            prompt = textwrap.dedent(
                f"""
                Answer using only this chunk. If it does not help, say NO_ANSWER.

                Question: {question}

                Chunk {index}/{len(self.context_chunks)}:
                {chunk}
                """
            ).strip()
            answer = self.llm_query(prompt)
            if "NO_ANSWER" not in answer:
                answers.append(f"Chunk {index}: {answer}")
        return answers

    def search_context(self, pattern: str, max_matches: int = 10, window: int = 300) -> list[dict]:
        matches = []
        for match in re.finditer(pattern, self.context_text, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(self.context_text), match.end() + window)
            matches.append(
                {
                    "position": match.start(),
                    "match": match.group(0),
                    "snippet": self.context_text[start:end],
                }
            )
            if len(matches) >= max_matches:
                break
        return matches

    def peek(self, start: int = 0, chars: int = 1000) -> str:
        return self.context_text[start : start + chars]

    def next_step_hint(self) -> str:
        return (
            "Please inspect the context with a fenced repl code block, or finish with "
            "FINAL(...) / FINAL_VAR(final_answer)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational Recursive LM demo")
    parser.add_argument("query", nargs="*", help="Question to ask about the context")
    parser.add_argument("--context", help="Optional text file to use instead of the paper")
    parser.add_argument("--no-subcalls", action="store_true", help="Use REPL only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_text = load_context(args.context)

    if args.query:
        question = " ".join(args.query).strip()
        log("# Recursive Language Model Demo\n")
        log(f"**Timestamp**: {datetime.now().isoformat()}\n")
        log(f"**Q**: {question}")
        rlm = RecursiveLanguageModel(context_text, allow_subcalls=not args.no_subcalls)
        answer = rlm.run(question)
        log(f"**A**: {answer}")
    else:
        log("# Recursive Language Model Demo\n")
        log(f"**Timestamp**: {datetime.now().isoformat()}\n")
        log("## Query Phase 1: Model Evaluation\n")
        log(f"Fetching RLM paper...\nPaper length: {len(context_text)} chars\n")

        rlm = RecursiveLanguageModel(context_text, allow_subcalls=not args.no_subcalls)

        q1 = "What models were compared in the paper?"
        log(f"**Q1: {q1}**")
        a1 = rlm.run(q1)
        log(f"\n**A**: {a1}\n")

        q2 = "What are the main tasks evaluated?"
        log(f"**Q2: {q2}**")
        a2 = rlm.run(q2)
        log(f"\n**A**: {a2}\n")

        log("\n" + "=" * 50 + "\n")
        log("## Query Phase 2: Author Count\n")

        q3 = "How many authors does the paper have?"
        log(f"**Q: {q3}**")
        a3 = rlm.run(q3)
        log(f"\n**A**: {a3}")

    save_log()
    log(f"\nSaved trace to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
