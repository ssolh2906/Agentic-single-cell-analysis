import contextlib
import io
import os
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

MODEL = "claude-sonnet-5-5"  # switch to claude-sonnet-5-5 for the Sonnet run

# One folder per run so results and logs never overwrite each other
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RUN_DIR = PROJECT_DIR / "runs" / f"{datetime.now():%Y%m%d_%H%M}_{MODEL.removeprefix('claude-')}_llm"
RUN_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirror console output into run.log."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        self._log.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_log_file = open(RUN_DIR / "run.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"Run folder: {RUN_DIR}")

# Agent-generated plotting code runs in this process; the default Tk backend
# crashes at exit when figures are left open, so render off-screen instead.
os.environ.setdefault("MPLBACKEND", "Agg")

# No Biomni here: the Anthropic SDK is called directly, so the LangChain patches
# (prefill, temperature, empty-response loop) are not needed.
import anthropic

# ---------------------------------
# Load the API key from .env into this process only, so it never lives in code
# and Claude Code itself keeps using its own login instead of this key.
from dotenv import load_dotenv
load_dotenv()
if os.getenv("ANTHROPIC_API_KEY", "your-api-key-here") == "your-api-key-here":
    raise RuntimeError("Set ANTHROPIC_API_KEY in .env before running.")

# Set defaults for running the agent (same limits as the Biomni run)
TIMEOUT_SECONDS = 60  # Kill execution if it hangs too long
MAX_ITERATIONS = 20  # model calls; Biomni's 40 graph steps were ~20 iterations
MAX_OBSERVATION_CHARS = 10_000  # keep long outputs from flooding the context

# 1. Initialize a minimal code-executing agent: the model writes Python, we run it
# in one persistent namespace (like Biomni's REPL) and return the output.
client = anthropic.Anthropic()
_namespace = {"__name__": "__agent__"}

SYSTEM_PROMPT = (
    "You are an expert bioinformatics agent. Solve the task by running Python with the "
    "run_python tool. Variables persist between calls. scanpy, anndata, numpy, pandas, "
    "scipy and matplotlib are installed; check before using anything else. Work in small "
    "steps, inspect outputs, and fix errors. When the task is complete, reply with a "
    "concise final summary of the results and any deviations, without calling a tool."
)

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Execute Python code in a persistent namespace and return stdout, stderr and any "
        f"traceback. Each call times out after {TIMEOUT_SECONDS} seconds."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _timeout_handler(signum, frame):
    raise TimeoutError(f"Execution exceeded {TIMEOUT_SECONDS} seconds.")


def run_python(code):
    """Run agent code and return its captured output as the observation."""
    buffer = io.StringIO()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(code, _namespace)
    except BaseException:
        buffer.write(traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    output = buffer.getvalue() or "(no output)"
    if len(output) > MAX_OBSERVATION_CHARS:
        output = output[:MAX_OBSERVATION_CHARS] + "\n... [output truncated]"
    return output


transcript = []  # (heading, text) pairs for the markdown report


def go(task):
    """Agent loop: call the model, run its code, feed results back until it answers."""
    messages = [{"role": "user", "content": task}]
    transcript.append(("Task", task))
    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=[RUN_PYTHON_TOOL],
            messages=messages,
            # Retry on a fallback model if the request is declined for policy reasons
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # Append the full content so thinking blocks are passed back unchanged
        messages.append({"role": "assistant", "content": response.content})

        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if text:
            print(f"\n=== Iteration {iteration} | {response.model} ===\n{text}")
            transcript.append((f"Agent (iteration {iteration})", text))

        if response.stop_reason == "refusal":
            return f"Stopped: the model declined the request ({response.stop_details})."
        if response.stop_reason == "max_tokens":
            messages.append({"role": "user", "content": "Your reply was cut off. Please continue."})
            continue
        if response.stop_reason != "tool_use":
            return text

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            code = block.input.get("code", "")
            print(f"\n--- Executing code ---\n{code}")
            observation = run_python(code)
            print(f"--- Observation ---\n{observation}")
            transcript.append(("Code", f"```python\n{code}\n```"))
            transcript.append(("Observation", f"```\n{observation}\n```"))
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": observation})
        messages.append({"role": "user", "content": tool_results})

    return None


# 2. MCP tools are skipped: mcp_config.yaml only points at Biomni's pubmed module.

# 3. Define your single-cell bioinformatics tasks as a single cohesive prompt
sc_rna_task = (
    f"Analyze this single-cell RNA-seq dataset located at {(DATA_DIR / 'pbmc_sample.h5ad').as_posix()}:\n"
    "1. Run quality control (QC) filtering out low-quality cells (high mitochondrial ratio or low counts).\n"
    "2. Perform normalization, log-transformation, and find highly variable genes.\n"
    "3. Compute PCA and cluster the cell populations using a UMAP embedding.\n"
    "4. Annotate the cell clusters using canonical marker genes.\n"
    "5. Identify differentially expressed genes (DEGs) between the Control and Treated conditions.\n"
    f"Treat the input as read-only. Save every output (plots, tables, h5ad) in {RUN_DIR.as_posix()}/ "
    "and do not write anywhere else."
)

# 4. Pass the string into the agent execution loop
print("Starting single-cell RNA-seq analysis via Claude API LLM agent...")
# Agent code runs in this process and saves plots with relative paths, so
# working from the run folder keeps them (and the report) out of the project root.
os.chdir(RUN_DIR)
result = go(sc_rna_task)
if result is None:
    # Stopped by the step cap instead of looping; keep what the agent produced so far
    print(f"Stopped after {MAX_ITERATIONS} iterations without a final answer.")
    result = "\n\n".join(text for _, text in transcript[-3:])

# 5. Output the textual insights and interpretation generated by the agent
print("\n--- Analysis Report Summary ---")
print(result)

# Save every AnnData the agent left in its namespace, even if it never wrote one itself
from anndata import AnnData

saved_ids = set()
for name, obj in list(_namespace.items()):
    # The same object is often bound to several names; save it once
    if isinstance(obj, AnnData) and id(obj) not in saved_ids:
        saved_ids.add(id(obj))
        out_path = RUN_DIR / f"final_{name}.h5ad"
        try:
            obj.write_h5ad(out_path)
            print(f"Saved AnnData '{name}' {obj.shape} to {out_path}")
        except Exception as e:
            print(f"Could not save AnnData '{name}': {e}")
if not saved_ids:
    print("No AnnData object found in the agent's namespace.")

# 6. Save the full execution history (task, reasoning, code, outputs) as a markdown report
report_md = "scrna_seq_analysis_report.md"
with open(report_md, "w", encoding="utf-8") as f:
    f.write("# scRNA-seq analysis report\n\n")
    for heading, text in transcript:
        f.write(f"## {heading}\n\n{text}\n\n")
    f.write(f"## Final summary\n\n{result}\n")
print(f"Analysis complete. Detailed report saved to {report_md}")
print()
