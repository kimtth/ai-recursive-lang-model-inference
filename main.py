import os
import re
import textwrap
from datetime import datetime
from html import unescape
from html.parser import HTMLParser

import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
)
model_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Logging
log_output = []


def log(msg: str = ""):
    # Always encode to UTF-8 first to avoid console encoding issues
    safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    try:
        print(safe_msg)
    except Exception:
        # Worst case: strip non-ASCII
        ascii_only = safe_msg.encode("ascii", errors="ignore").decode("ascii")
        print(ascii_only)
    log_output.append(msg)


def save_log(filepath="OUTPUT.md"):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(log_output))


# === Data Processing Functions ===


class _TextExtractor(HTMLParser):
    """Extract clean text from HTML."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {
            "br",
            "p",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "tr",
            "ul",
            "ol",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "div", "section", "article", "tr", "ul", "ol"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if data:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Convert HTML to clean text."""
    # Remove script and style tags
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)

    # Parse HTML
    parser = _TextExtractor()
    parser.feed(cleaned)
    text = unescape("".join(parser.parts))

    # Clean whitespace
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_paper():
    """Download and cache paper."""
    paper_path = os.path.join("doc", "rlm_paper.txt")
    if os.path.exists(paper_path):
        with open(paper_path, "r", encoding="utf-8") as f:
            return f.read()

    url = "https://arxiv.org/html/2512.24601v1"
    response = requests.get(url)
    response.raise_for_status()

    text = html_to_text(response.text)

    os.makedirs("doc", exist_ok=True)
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# === LLM API Functions ===


def llm_query(prompt: str, model: str = model_deployment_name) -> str:
    """Make a single LLM API call."""
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# === Core RLM Logic ===


class RLM:
    """Recursive Language Model - treats prompts as environment variables."""

    def __init__(self, context: str, model: str = model_deployment_name):
        self.context = context
        self.model = model
        self.env = {
            "context": context,
            "llm_query": self._sub_query,
            "recursive_search": self._recursive_search,
        }

    def _sub_query(self, prompt: str) -> str:
        """Wrapper for recursive LLM calls."""
        return llm_query(prompt, self.model)

    def _recursive_search(
        self, question: str, chunk_size: int = 3000, overlap: int = 200
    ) -> str:
        """Split context into chunks and search recursively. Returns joined string of results."""
        # Chunk the context
        chunks = []
        start = 0
        while start < len(self.context):
            end = min(len(self.context), start + chunk_size)
            chunks.append(self.context[start:end])
            if end >= len(self.context):
                break
            start = end - overlap

        # Query each chunk
        partials = []
        for i, chunk in enumerate(chunks):
            prompt = f"Answer based ONLY on this chunk. If not found, respond with 'NO_ANSWER'.\\n\\nQuestion: {question}\\n\\nChunk {i + 1}/{len(chunks)}:\\n{chunk}"
            try:
                result = self._sub_query(prompt)
                if "NO_ANSWER" not in result:
                    partials.append(result)
            except Exception as e:
                log(f"Recursive search error on chunk {i + 1}: {e}")

        # Return as joined string so generated code can treat it as a string
        return "\n\n---\n\n".join(partials) if partials else ""

    def run(self, query: str, max_turns: int = 10) -> str:
        """Main inference loop."""
        system = textwrap.dedent(f"""
            You have a REPL with 'context' variable ({len(self.context)} chars).
            Use recursive_search(question) or llm_query(prompt) for recursive sub-queries.
            You MUST make at least one recursive call (recursive_search or llm_query) before FINAL.
            Analyze context with Python code. Store answer in 'answer' variable.
            When done, output either:
            1) A python code block that sets answer and prints it, then write FINAL
            or
            2) Write FINAL: <answer> directly if no code is needed.
        """).strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        for step in range(max_turns):
            try:
                response = client.chat.completions.create(
                    model=self.model, messages=messages
                )
                content = response.choices[0].message.content
            except Exception as e:
                log(f"[step {step}] API Error: {e}")
                break

            log(f"[step {step}] assistant:\n{content}\n")

            # Check for code block
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0].strip()

                try:
                    # Redirect stdout to capture prints
                    import io
                    import sys

                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()

                    exec(code, self.env)

                    sys.stdout = old_stdout

                    ans = self.env.get("answer", "not set")
                    result = f"Executed. answer={ans}"
                except Exception as e:
                    sys.stdout = old_stdout
                    result = f"Error during execution: {str(e)[:100]}"

                log(f"[step {step}] exec result: {result}\n")

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": result})

                # Check for FINAL after code execution
                if "FINAL" in content:
                    final_text = content.split("FINAL")[-1].strip().lstrip(":\n ")
                    if final_text:
                        return final_text
                    ans = self.env.get("answer")
                    if ans:
                        return str(ans)
                    return "Process complete"
            else:
                messages.append({"role": "assistant", "content": content})

                # Check for FINAL in non-code response
                if "FINAL" in content:
                    final_text = content.split("FINAL")[-1].strip().lstrip(":\n ")
                    if final_text:
                        return final_text
                    ans = self.env.get("answer")
                    if ans:
                        return str(ans)
                    return "Process complete"

                messages.append(
                    {
                        "role": "user",
                        "content": "Please respond with a python code block or FINAL: <answer>.",
                    }
                )

        return self.env.get("answer", "No answer found")


if __name__ == "__main__":
    log("# Recursive Language Model Demo\n")
    log(f"**Timestamp**: {datetime.now().isoformat()}\n")

    log("## Query Phase 1: Model Evaluation\n")
    log("Fetching RLM paper...")
    paper = fetch_paper()
    log(f"Paper length: {len(paper)} chars\n")

    rlm = RLM(paper)

    log("**Q1: What models were compared in the paper?**")
    answer = rlm.run("What language models were used for evaluation in this paper?")
    log(f"**A**: {answer}\n")

    log("**Q2: What are the main tasks evaluated?**")
    answer = rlm.run("List the benchmark tasks used to evaluate RLMs.")
    log(f"**A**: {answer}\n")

    log("\n" + "=" * 50 + "\n")

    log("## Query Phase 2: Author Count\n")
    log("**Q: How many authors does the paper have?**")
    answer = rlm.run("Count the number of authors of this paper.")
    log(f"**A**: {answer}\n")

    save_log()
    log("\n✓ Output saved to OUTPUT.md")
