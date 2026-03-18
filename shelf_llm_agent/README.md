# SHELF LLM Agent

An LLM-assisted prior elicitation agent for the
[SHELF](https://github.com/OakleyJ/SHELF) R package. This agent
bridges large language models and SHELF's distribution fitting
functions, enabling automated Bayesian prior elicitation **without
modifying any existing R scripts**.

## Architecture

```
┌─────────┐     ┌──────────────┐     ┌────────────────┐
│  User   │────▶│ Python Agent │────▶│   LLM API      │
│  (CLI)  │     │ (Orchestrator│     │ (OpenAI/Ollama) │
└─────────┘     └──────┬───────┘     └────────────────┘
                       │
                       ▼
                ┌──────────────┐     ┌────────────────┐
                │  R Bridge    │────▶│ SHELF R Package │
                │ (subprocess) │     │  (unmodified)   │
                └──────────────┘     └────────────────┘
```

**Data flow**: User → Agent → LLM (structured prompts) → Agent
parses JSON response → R Bridge calls `fitdist()` via subprocess →
Agent returns fitted distributions + audit log.

## Module Reference

| Module | Description |
|--------|-------------|
| `config.py` | `LlmConfig` and `AgentConfig` dataclasses for LLM provider, R path, seeds. `AuditLogger` for full session traceability. |
| `prompts.py` | Prompt templates for single, precision, and Dirichlet elicitation. Includes JSON response parsing (handles markdown fences) and validation functions. |
| `serializer.py` | JSON serialization/deserialization between Python dicts and R-compatible format. Handles `Inf`/`-Inf` conversion. |
| `r_bridge.py` | `RBridge` class that calls SHELF R functions (`fitdist`, `feedback`, `sampleFit`, `fitprecision`) via `subprocess` with JSON stdin/stdout. |
| `agent.py` | `ShelfLlmAgent` orchestrator. Chains: prompt building → LLM call → response validation (with retries) → R fitting → audit logging. |
| `cli.py` | Command-line entry point. Supports `--mode traditional` (Shiny) and `--mode llm` (LLM-assisted). |
| `shelf_bridge.R` | Thin R script that reads JSON from stdin, calls SHELF functions, writes JSON to stdout. **Does not modify any SHELF source.** |

## Installation

```bash
# Inside your project environment
pip install langchain-openai

# Verify R + SHELF are available
python3 -m shelf_llm_agent.cli --dry-run
```

## Configuration

Edit `config.py` to set your defaults:

```python
@dataclass
class LlmConfig:
    provider: str = "ollama"           # or "openai"
    model: str = "gpt-oss:120b-cloud"  # your model
    temperature: float = 0.2
    seed: int = 42
    api_key: Optional[str] = None      # or set env var
    base_url: Optional[str] = "https://ollama.com/v1"
    max_tokens: int = 2048
```

Set your API key via environment variable:

```bash
export OLLAMA_API_KEY="your-key-here"
```

## Usage

### LLM-Assisted Single Distribution

```bash
python3 -m shelf_llm_agent.cli \
  --mode llm \
  --elicitation-type single \
  --parameter "5-year survival rate" \
  --context "phase III clinical trial"
```

### Example Output

```
Starting single distribution elicitation...
Parameter: 5-year survival rate
Context: phase III clinical trial

==================================================
ELICITATION RESULTS
==================================================

Elicited judgements:
{
  "lower": 0.05,
  "upper": 0.95,
  "values": [0.3, 0.55, 0.75],
  "probabilities": [0.25, 0.5, 0.75],
  "reasoning": "In a phase III trial the 5-year survival rate
    is a proportion between 0 and 1. Extremely low (<5%) or
    near-perfect (>95%) outcomes are unlikely for most
    indications, so 5% and 95% are set as plausible extremes."
}

=== Fitted Distribution Summary ===
Best fitting: skewnormal

Normal: mean: 0.5360, sd: 0.3349
Student.t: location: 0.5376, scale: 0.2951, df: 3.0000
Gamma: shape: 1.9878, rate: 3.5993
Log.normal: mean_log_X: -0.7954, sd_log_X: 0.7927
Beta: shape1: 1.0545, shape2: 0.9303

Sum of squared errors:
  skewnormal: 0.000000
  beta: 0.000300
  normal: 0.000500

Audit log saved to: elicitation_logs/elicitation_32d8f106_....json
```

### Precision Elicitation

```bash
python3 -m shelf_llm_agent.cli \
  --mode llm \
  --elicitation-type precision \
  --parameter "patient heights in cm" \
  --interval 160 180 \
  --median 170
```

### Dirichlet Elicitation

```bash
python3 -m shelf_llm_agent.cli \
  --mode llm \
  --elicitation-type dirichlet \
  --categories "responders" "partial" "non-responders" \
  --context "cancer immunotherapy trial"
```

### Traditional SHELF (Shiny App)

```bash
python3 -m shelf_llm_agent.cli --mode traditional
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `llm` | `llm` or `traditional` |
| `--elicitation-type` | `single` | `single`, `precision`, or `dirichlet` |
| `--parameter` | — | Description of the uncertain quantity |
| `--context` | — | Domain-specific background |
| `--categories` | — | Category names (Dirichlet only) |
| `--interval` | — | Two floats: lower upper (precision only) |
| `--median` | — | Hypothetical median (precision only) |
| `--llm-provider` | from config | Override LLM provider |
| `--llm-model` | from config | Override model name |
| `--temperature` | `0.2` | LLM sampling temperature |
| `--seed` | `42` | Random seed (LLM + R) |
| `--base-url` | from config | Override API base URL |
| `--api-key` | from config | Override API key |
| `--r-executable` | `Rscript` | Path to Rscript |
| `--log-dir` | `./elicitation_logs` | Audit log directory |
| `--dry-run` | — | Check R + SHELF availability only |
| `--verbose` | — | Enable debug logging |

## How It Works

1. **Prompt Generation** (`prompts.py`): Builds a structured prompt
   asking the LLM to provide probability judgements as JSON
   (values, probabilities, limits, reasoning).

2. **LLM Call** (`agent.py`): Sends the prompt via
   `langchain_openai.ChatOpenAI` with `response_format: json_object`.
   If the response fails validation, retries up to 2 times with
   error feedback.

3. **R Bridge** (`r_bridge.py` + `shelf_bridge.R`): Serializes the
   parsed judgements as JSON, pipes them to `Rscript` via stdin. The
   R script calls `SHELF::fitdist()` and returns fitted parameters
   as JSON via stdout.

4. **Results**: The agent displays fitted distributions (Normal,
   Student-t, Gamma, Log-normal, Beta, etc.), identifies the best
   fit, and saves a full audit log.

## Audit Log

Every session is saved as a JSON file in `elicitation_logs/`:

```json
{
  "session_id": "uuid",
  "config": {
    "llm_model": "gpt-oss:120b-cloud",
    "llm_temperature": 0.2,
    "llm_seed": 42,
    "shelf_version": "1.12.1"
  },
  "steps": [
    {
      "step": 1,
      "type": "llm_elicit_attempt_1",
      "prompt": "...",
      "response_raw": "...",
      "parsed_values": {"vals": [...], "probs": [...]}
    },
    {
      "step": 2,
      "type": "r_fitdist",
      "r_result": {"best_fitting": "skewnormal", ...}
    }
  ]
}
```

## Tests

```bash
python3 -m unittest discover -s shelf_llm_agent/tests -v
```

32 unit tests covering serialization, prompt building, JSON parsing,
validation, and R bridge (mocked subprocess).

## Supported Elicitation Types

| Type | SHELF Function | Description |
|------|---------------|-------------|
| `single` | `fitdist()` | Fit parametric distributions to quartile judgements |
| `precision` | `fitprecision()` | Elicit population variability via interval proportions |
| `dirichlet` | `fitdist()` per marginal | Elicit proportions across multiple categories |

## Programmatic Usage

```python
from shelf_llm_agent.agent import ShelfLlmAgent
from shelf_llm_agent.config import AgentConfig

agent = ShelfLlmAgent(config=AgentConfig())

result = agent.elicit_single(
    parameter_description="5-year survival rate",
    domain_context="phase III clinical trial",
)

print(result["fit_result"]["best_fitting"])
# => "skewnormal"

agent.save_session()
```
