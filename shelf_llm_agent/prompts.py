"""Prompt templates for LLM-assisted prior elicitation."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SINGLE_ELICITATION_TEMPLATE = """You are a domain expert \
being consulted about the uncertain quantity: \
"{parameter_description}".

Context: {domain_context}

Please provide your probability judgements. Think carefully \
about each value based on the domain context provided.

1. What is the LOWEST plausible value for this quantity? \
(lower limit)
2. What is the HIGHEST plausible value for this quantity? \
(upper limit)
3. For each cumulative probability, provide the value X such \
that P(quantity <= X) = p:
   - P(quantity <= X) = 0.25  ->  X = ?
   - P(quantity <= X) = 0.50  ->  X = ?
   - P(quantity <= X) = 0.75  ->  X = ?

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "lower": <number>,
  "upper": <number>,
  "values": [<Q25>, <Q50>, <Q75>],
  "probabilities": [0.25, 0.5, 0.75],
  "reasoning": "<brief justification for your choices>"
}}"""

PRECISION_TEMPLATE = """You are assessing population \
variability for: "{population_description}".

Assume the population median is {median_value}.
Consider the proportion of the population that falls in the \
interval [{lower_bound}, {upper_bound}].

Provide two probability judgements about this proportion:
1. You are 5% sure the proportion is at most ___
2. You are 95% sure the proportion is at most ___

These values must be between 0 and 0.5 (exclusive).

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "interval": [{lower_bound}, {upper_bound}],
  "prop_values": [<5th_pct_proportion>, <95th_pct_proportion>],
  "prop_probs": [0.05, 0.95],
  "reasoning": "<brief justification>"
}}"""

DIRICHLET_TEMPLATE = """You are assessing proportions across \
{n_categories} categories: {category_list}.

These proportions must sum to 1.0.

For EACH category, provide quartile judgements for the \
proportion in that category:
   - P(proportion <= X) = 0.25  ->  X = ?
   - P(proportion <= X) = 0.50  ->  X = ?
   - P(proportion <= X) = 0.75  ->  X = ?

Context: {domain_context}

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "categories": {category_json},
  "marginals": [
    {{
      "category": "<name>",
      "values": [<Q25>, <Q50>, <Q75>],
      "probabilities": [0.25, 0.5, 0.75],
      "lower": 0,
      "upper": 1
    }}
  ],
  "reasoning": "<brief justification>"
}}"""

CUSTOM_QUANTILES_TEMPLATE = """You are a domain expert \
being consulted about the uncertain quantity: \
"{parameter_description}".

Context: {domain_context}

Please provide your probability judgements.

1. Lower plausible limit: ?
2. Upper plausible limit: ?
3. For each cumulative probability below, provide the value X:
{quantile_lines}

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "lower": <number>,
  "upper": <number>,
  "values": [{value_placeholders}],
  "probabilities": [{prob_values}],
  "reasoning": "<brief justification>"
}}"""

FEEDBACK_TEMPLATE = """Based on your previous elicitation, \
the best fitting distribution is: {best_fitting}.

Fitted parameters: {fitted_params}

The fitted quantiles are:
{quantile_table}

Sum of squared errors: {ssq_best:.6f}

Review these results. If you believe the fit is reasonable, \
respond with:
{{"accept": true, "reasoning": "<why this is acceptable>"}}

If you want to revise your judgements, respond with:
{{"accept": false, "revised_values": [<new Q25>, <new Q50>, \
<new Q75>], "reasoning": "<why you are revising>"}}"""


def build_single_elicitation_prompt(
    parameter_description: str,
    domain_context: str = "",
) -> str:
    """Build a prompt for single distribution elicitation.

    Args:
        parameter_description: What the uncertain quantity is.
        domain_context: Additional domain-specific context.

    Returns:
        Formatted prompt string.
    """
    return SINGLE_ELICITATION_TEMPLATE.format(
        parameter_description=parameter_description,
        domain_context=domain_context or "No additional context.",
    )


def build_custom_quantiles_prompt(
    parameter_description: str,
    probabilities: List[float],
    domain_context: str = "",
) -> str:
    """Build a prompt with custom quantile probabilities.

    Args:
        parameter_description: What the uncertain quantity is.
        probabilities: List of cumulative probabilities.
        domain_context: Additional domain-specific context.

    Returns:
        Formatted prompt string.
    """
    quantile_lines = "\n".join(
        "   - P(quantity <= X) = {}  ->  X = ?".format(p)
        for p in probabilities
    )
    value_placeholders = ", ".join(
        "<Q{}>".format(int(p * 100)) for p in probabilities
    )
    prob_values = ", ".join(str(p) for p in probabilities)

    return CUSTOM_QUANTILES_TEMPLATE.format(
        parameter_description=parameter_description,
        domain_context=domain_context or "No additional context.",
        quantile_lines=quantile_lines,
        value_placeholders=value_placeholders,
        prob_values=prob_values,
    )


def build_precision_prompt(
    population_description: str,
    median_value: float,
    lower_bound: float,
    upper_bound: float,
) -> str:
    """Build a prompt for precision elicitation.

    Args:
        population_description: Description of the population.
        median_value: Hypothetical population median.
        lower_bound: Lower endpoint of the interval.
        upper_bound: Upper endpoint of the interval.

    Returns:
        Formatted prompt string.
    """
    return PRECISION_TEMPLATE.format(
        population_description=population_description,
        median_value=median_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def build_dirichlet_prompt(
    categories: List[str],
    domain_context: str = "",
) -> str:
    """Build a prompt for Dirichlet distribution elicitation.

    Args:
        categories: List of category names.
        domain_context: Additional domain-specific context.

    Returns:
        Formatted prompt string.
    """
    return DIRICHLET_TEMPLATE.format(
        n_categories=len(categories),
        category_list=", ".join(categories),
        category_json=json.dumps(categories),
        domain_context=domain_context or "No additional context.",
    )


def build_feedback_prompt(
    fit_result: Dict[str, Any],
) -> str:
    """Build a feedback prompt from a fitdist result.

    Args:
        fit_result: Dict from R bridge fitdist call.

    Returns:
        Formatted feedback prompt string.
    """
    best = fit_result.get("best_fitting", "unknown")
    ssq = fit_result.get("ssq", {})
    ssq_best = ssq.get(best, 0.0) if ssq else 0.0

    fitted_params = fit_result.get(best, {})
    param_str = json.dumps(fitted_params, indent=2)

    quantile_lines = []
    for dist_name in ["Normal", "Gamma", "Log.normal", "Beta"]:
        params = fit_result.get(dist_name, {})
        if params and any(v is not None for v in params.values()):
            quantile_lines.append(
                "  {}: {}".format(dist_name, params)
            )

    return FEEDBACK_TEMPLATE.format(
        best_fitting=best,
        fitted_params=param_str,
        quantile_table="\n".join(quantile_lines),
        ssq_best=ssq_best if ssq_best else 0.0,
    )


def parse_llm_response(
    response_text: str,
) -> Optional[Dict[str, Any]]:
    """Extract structured JSON from an LLM response.

    Handles responses with or without markdown code fences.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        Parsed dictionary, or None if parsing fails.
    """
    text = response_text.strip()

    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        text,
        re.DOTALL,
    )
    if json_match:
        text = json_match.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        text = brace_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse LLM response as JSON: %s",
            response_text[:200],
        )
        return None


def validate_single_response(
    parsed: Dict[str, Any],
) -> List[str]:
    """Validate a parsed single elicitation response.

    Args:
        parsed: Parsed JSON from LLM response.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    for key in ["lower", "upper", "values", "probabilities"]:
        if key not in parsed:
            errors.append("Missing key: '{}'".format(key))

    if not errors:
        if parsed["lower"] >= parsed["upper"]:
            errors.append("lower must be less than upper")

        vals = parsed["values"]
        probs = parsed["probabilities"]
        if len(vals) != len(probs):
            errors.append(
                "values and probabilities must have same length"
            )

        if vals != sorted(vals):
            errors.append("values must be in ascending order")

        if probs != sorted(probs):
            errors.append(
                "probabilities must be in ascending order"
            )

        if any(v < parsed["lower"] for v in vals):
            errors.append(
                "values must be >= lower limit"
            )

        if any(v > parsed["upper"] for v in vals):
            errors.append(
                "values must be <= upper limit"
            )

        if any(p <= 0 or p >= 1 for p in probs):
            errors.append(
                "probabilities must be strictly between 0 and 1"
            )

    return errors


def validate_precision_response(
    parsed: Dict[str, Any],
) -> List[str]:
    """Validate a parsed precision elicitation response.

    Args:
        parsed: Parsed JSON from LLM response.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    for key in ["interval", "prop_values", "prop_probs"]:
        if key not in parsed:
            errors.append("Missing key: '{}'".format(key))

    if not errors:
        pv = parsed["prop_values"]
        if any(v <= 0 or v >= 0.5 for v in pv):
            errors.append(
                "prop_values must be between 0 and 0.5"
            )

    return errors
