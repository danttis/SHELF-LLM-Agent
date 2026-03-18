# SHELF — Tools to Support the Sheffield Elicitation Framework

[![R Version](https://img.shields.io/badge/R-%E2%89%A5%203.5.0-blue)](https://www.r-project.org/)
[![SHELF Version](https://img.shields.io/badge/SHELF-1.12.1-green)](https://cran.r-project.org/package=SHELF)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)

This repository contains the **SHELF** R package for expert probability
elicitation, extended with an **LLM-assisted elicitation agent** that
automates the process using large language models.

---

## What Is SHELF?

SHELF (Sheffield Elicitation Framework) is an R package that provides
tools for eliciting probability distributions from domain experts. It
supports:

- **Single expert elicitation** — fit distributions (Normal, Gamma,
  Beta, Log-normal, etc.) to quantile judgements
- **Multiple expert elicitation** — aggregate beliefs from multiple
  experts via linear pooling
- **Dirichlet elicitation** — elicit proportions across categories
- **Precision elicitation** — assess population variability
- **Interactive Shiny apps** — visual, guided elicitation interfaces

## What Is SHELF LLM Agent?

The [`shelf_llm_agent/`](shelf_llm_agent/) directory contains a
**Python agent** that replaces or supplements human experts with an LLM
for prior elicitation. It acts as a bridge between language models and
SHELF's R functions — **without modifying any existing R source code**.

### Why?

Traditional expert elicitation requires scheduling sessions with domain
experts, which can be time-consuming and expensive. The LLM agent:

- **Automates elicitation** by prompting an LLM for structured
  probability judgements
- **Preserves SHELF's fitting engine** — all distribution fitting
  is still done by SHELF's validated R functions
- **Enables reproducibility** — every session is logged with prompts,
  responses, seeds, and fitted parameters
- **Supports comparison** — run both LLM and traditional elicitation
  to compare results

### Architecture

```
User (CLI) ──▶ Python Agent ──▶ LLM API (OpenAI/Ollama)
                    │
                    ▼
               R Bridge ──▶ SHELF R Package (unmodified)
              (subprocess)     fitdist(), fitprecision()
                    │
                    ▼
              Audit Log (JSON)
```

The agent sends structured prompts to the LLM, parses JSON responses,
validates them, and pipes the judgements to SHELF via `Rscript`
subprocess calls. All communication uses JSON stdin/stdout.

### Quick Start

```bash
# Install dependencies
pip install langchain-openai

# Set your API key
export OLLAMA_API_KEY="your-key"

# Check R + SHELF availability
python3 -m shelf_llm_agent.cli --dry-run

# Run LLM-assisted elicitation
python3 -m shelf_llm_agent.cli \
  --mode llm \
  --elicitation-type single \
  --parameter "5-year survival rate" \
  --context "phase III clinical trial"
```

### Example Output

```
Elicited judgements:
{
  "lower": 0.05,
  "upper": 0.95,
  "values": [0.3, 0.55, 0.75],
  "probabilities": [0.25, 0.5, 0.75],
  "reasoning": "In a phase III trial the 5-year survival rate
    is a proportion between 0 and 1..."
}

=== Fitted Distribution Summary ===
Best fitting: skewnormal

Normal: mean: 0.5360, sd: 0.3349
Beta:   shape1: 1.0545, shape2: 0.9303
Gamma:  shape: 1.9878, rate: 3.5993
```

### Supported Elicitation Types

| Type | Command | SHELF Function |
|------|---------|---------------|
| Single distribution | `--elicitation-type single` | `fitdist()` |
| Precision | `--elicitation-type precision` | `fitprecision()` |
| Dirichlet | `--elicitation-type dirichlet` | `fitdist()` per marginal |
| Traditional (Shiny) | `--mode traditional` | `elicit()` |

### Agent Modules

| File | Role |
|------|------|
| `config.py` | LLM and R configuration, audit logging |
| `prompts.py` | Prompt templates, JSON parsing, validation |
| `serializer.py` | Python ↔ R JSON conversion |
| `r_bridge.py` | Subprocess calls to SHELF R functions |
| `agent.py` | Orchestrator with retry logic |
| `cli.py` | Command-line interface |
| `shelf_bridge.R` | R helper script (read-only, no SHELF changes) |

📖 Full documentation: [`shelf_llm_agent/README.md`](shelf_llm_agent/README.md)

---

## Repository Structure

```
SHELF/
├── R/                    # SHELF R package source (unmodified)
├── man/                  # R documentation
├── inst/                 # Shiny app files, help pages
├── tests/                # R unit tests (testthat)
├── vignettes/            # R vignettes
├── shelf_llm_agent/      # LLM elicitation agent (Python)
│   ├── config.py
│   ├── prompts.py
│   ├── serializer.py
│   ├── r_bridge.py
│   ├── agent.py
│   ├── cli.py
│   ├── shelf_bridge.R
│   ├── tests/
│   └── README.md
├── elicitation_logs/     # Audit logs (auto-generated)
├── DESCRIPTION           # R package metadata
├── NAMESPACE             # R exports
└── README.md             # This file
```

## Requirements

- **R** ≥ 3.5.0 with SHELF package installed
- **Python** ≥ 3.10 with `langchain-openai`
- An LLM API key (OpenAI, Ollama, or compatible provider)

## License

See [LICENSE.txt](LICENSE.txt).
