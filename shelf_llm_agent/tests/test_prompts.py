"""Unit tests for the prompts module."""

import json
import unittest

from shelf_llm_agent.prompts import (
    build_dirichlet_prompt,
    build_feedback_prompt,
    build_precision_prompt,
    build_single_elicitation_prompt,
    parse_llm_response,
    validate_precision_response,
    validate_single_response,
)


class TestBuildSinglePrompt(unittest.TestCase):
    """Tests for build_single_elicitation_prompt."""

    def test_contains_parameter(self) -> None:
        """Include parameter description in prompt."""
        prompt = build_single_elicitation_prompt(
            parameter_description="mortality rate",
            domain_context="ICU patients",
        )
        if "mortality rate" not in prompt:
            raise ValueError("Missing parameter in prompt")
        if "ICU patients" not in prompt:
            raise ValueError("Missing context in prompt")

    def test_default_context(self) -> None:
        """Use default context when none provided."""
        prompt = build_single_elicitation_prompt(
            parameter_description="effect size",
        )
        if "No additional context" not in prompt:
            raise ValueError("Missing default context")


class TestBuildPrecisionPrompt(unittest.TestCase):
    """Tests for build_precision_prompt."""

    def test_includes_interval(self) -> None:
        """Include interval values in prompt."""
        prompt = build_precision_prompt(
            population_description="heights",
            median_value=170.0,
            lower_bound=160.0,
            upper_bound=180.0,
        )
        if "160.0" not in prompt:
            raise ValueError("Missing lower bound")
        if "180.0" not in prompt:
            raise ValueError("Missing upper bound")


class TestBuildDirichletPrompt(unittest.TestCase):
    """Tests for build_dirichlet_prompt."""

    def test_includes_categories(self) -> None:
        """Include all category names in prompt."""
        prompt = build_dirichlet_prompt(
            categories=["cat_a", "cat_b", "cat_c"],
        )
        if "cat_a" not in prompt:
            raise ValueError("Missing category cat_a")
        if "3 categories" not in prompt:
            raise ValueError("Missing category count")


class TestParseLlmResponse(unittest.TestCase):
    """Tests for parse_llm_response."""

    def test_plain_json(self) -> None:
        """Parse plain JSON without fences."""
        data = {"lower": 0, "upper": 100, "values": [25, 50, 75]}
        result = parse_llm_response(json.dumps(data))
        if result is None:
            raise ValueError("Expected parsed result")
        if result["lower"] != 0:
            raise ValueError("Expected lower=0")

    def test_json_with_fences(self) -> None:
        """Parse JSON wrapped in markdown code fences."""
        text = '```json\n{"lower": 0, "upper": 50}\n```'
        result = parse_llm_response(text)
        if result is None:
            raise ValueError("Expected parsed result")
        if result["upper"] != 50:
            raise ValueError("Expected upper=50")

    def test_json_with_surrounding_text(self) -> None:
        """Extract JSON from response with extra text."""
        text = 'Here is my answer:\n{"lower": 5}\nThank you.'
        result = parse_llm_response(text)
        if result is None:
            raise ValueError("Expected parsed result")
        if result["lower"] != 5:
            raise ValueError("Expected lower=5")

    def test_invalid_text_returns_none(self) -> None:
        """Return None for unparseable responses."""
        result = parse_llm_response("no json here")
        if result is not None:
            raise ValueError("Expected None")


class TestValidateSingleResponse(unittest.TestCase):
    """Tests for validate_single_response."""

    def test_valid_response(self) -> None:
        """Accept valid single elicitation response."""
        parsed = {
            "lower": 0,
            "upper": 100,
            "values": [25, 50, 75],
            "probabilities": [0.25, 0.5, 0.75],
        }
        errors = validate_single_response(parsed)
        if errors:
            raise ValueError(
                "Expected no errors: {}".format(errors)
            )

    def test_missing_key(self) -> None:
        """Detect missing required keys."""
        parsed = {"lower": 0, "upper": 100}
        errors = validate_single_response(parsed)
        if not errors:
            raise ValueError("Expected errors for missing keys")

    def test_lower_gt_upper(self) -> None:
        """Detect lower >= upper."""
        parsed = {
            "lower": 100,
            "upper": 0,
            "values": [50],
            "probabilities": [0.5],
        }
        errors = validate_single_response(parsed)
        if not errors:
            raise ValueError("Expected error for bad limits")

    def test_unsorted_values(self) -> None:
        """Detect unsorted values."""
        parsed = {
            "lower": 0,
            "upper": 100,
            "values": [75, 50, 25],
            "probabilities": [0.25, 0.5, 0.75],
        }
        errors = validate_single_response(parsed)
        if not errors:
            raise ValueError("Expected error for unsorted")

    def test_values_out_of_bounds(self) -> None:
        """Detect values outside limits."""
        parsed = {
            "lower": 10,
            "upper": 100,
            "values": [5, 50, 75],
            "probabilities": [0.25, 0.5, 0.75],
        }
        errors = validate_single_response(parsed)
        if not errors:
            raise ValueError("Expected out-of-bounds error")


class TestValidatePrecisionResponse(unittest.TestCase):
    """Tests for validate_precision_response."""

    def test_valid_precision(self) -> None:
        """Accept valid precision response."""
        parsed = {
            "interval": [60, 70],
            "prop_values": [0.1, 0.3],
            "prop_probs": [0.05, 0.95],
        }
        errors = validate_precision_response(parsed)
        if errors:
            raise ValueError(
                "Expected no errors: {}".format(errors)
            )

    def test_prop_out_of_range(self) -> None:
        """Detect proportions outside (0, 0.5)."""
        parsed = {
            "interval": [60, 70],
            "prop_values": [0.6, 0.8],
            "prop_probs": [0.05, 0.95],
        }
        errors = validate_precision_response(parsed)
        if not errors:
            raise ValueError("Expected range error")


class TestBuildFeedbackPrompt(unittest.TestCase):
    """Tests for build_feedback_prompt."""

    def test_includes_best_fitting(self) -> None:
        """Include best fitting name in feedback prompt."""
        fit = {
            "best_fitting": "gamma",
            "gamma": {"shape": 2.0, "rate": 0.1},
            "ssq": {"gamma": 0.001},
        }
        prompt = build_feedback_prompt(fit)
        if "gamma" not in prompt:
            raise ValueError("Missing best fitting name")


if __name__ == "__main__":
    unittest.main()
