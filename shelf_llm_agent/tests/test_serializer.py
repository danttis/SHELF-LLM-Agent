"""Unit tests for the serializer module."""

import json
import unittest

from shelf_llm_agent.serializer import (
    dirichlet_to_json,
    elicitation_to_json,
    fit_result_to_summary,
    json_to_fit_result,
    precision_to_json,
    sample_to_json,
)


class TestElicitationToJson(unittest.TestCase):
    """Tests for elicitation_to_json function."""

    def test_basic_fitdist_payload(self) -> None:
        """Produce valid JSON with fitdist action."""
        result = elicitation_to_json(
            action="fitdist",
            vals=[20, 30, 50],
            probs=[0.25, 0.5, 0.75],
            lower=0,
            upper=100,
        )
        parsed = json.loads(result)
        if parsed["action"] != "fitdist":
            raise ValueError("Expected action 'fitdist'")
        if parsed["params"]["vals"] != [20, 30, 50]:
            raise ValueError("Unexpected vals")
        if parsed["params"]["lower"] != 0:
            raise ValueError("Unexpected lower")

    def test_infinite_limits_converted(self) -> None:
        """Convert Python infinities to R strings."""
        result = elicitation_to_json(
            action="fitdist",
            vals=[20, 30, 50],
            probs=[0.25, 0.5, 0.75],
            lower=float("-inf"),
            upper=float("inf"),
        )
        parsed = json.loads(result)
        if parsed["params"]["lower"] != "-Inf":
            raise ValueError("Expected -Inf string")
        if parsed["params"]["upper"] != "Inf":
            raise ValueError("Expected Inf string")

    def test_extra_params_included(self) -> None:
        """Include extra parameters in payload."""
        result = elicitation_to_json(
            action="fitdist",
            vals=[10],
            probs=[0.5],
            extra={"weights": [1]},
        )
        parsed = json.loads(result)
        if "weights" not in parsed["params"]:
            raise ValueError("Expected weights in params")


class TestPrecisionToJson(unittest.TestCase):
    """Tests for precision_to_json function."""

    def test_basic_precision_payload(self) -> None:
        """Produce valid JSON for fitprecision."""
        result = precision_to_json(
            interval=[60, 70],
            propvals=[0.2, 0.4],
        )
        parsed = json.loads(result)
        if parsed["action"] != "fitprecision":
            raise ValueError("Expected fitprecision action")
        if parsed["params"]["propprobs"] != [0.05, 0.95]:
            raise ValueError("Default propprobs wrong")


class TestDirichletToJson(unittest.TestCase):
    """Tests for dirichlet_to_json function."""

    def test_basic_dirichlet_payload(self) -> None:
        """Produce valid JSON for Dirichlet fitting."""
        result = dirichlet_to_json(
            marginals=[
                {"vals": [0.2, 0.3, 0.4], "probs": [0.25, 0.5, 0.75]},
            ],
            categories=["A"],
        )
        parsed = json.loads(result)
        if parsed["action"] != "fitdirichlet":
            raise ValueError("Expected fitdirichlet action")


class TestJsonToFitResult(unittest.TestCase):
    """Tests for json_to_fit_result function."""

    def test_valid_json_parsed(self) -> None:
        """Parse valid JSON string to dict."""
        data = {"Normal": {"mean": 30, "sd": 10}}
        result = json_to_fit_result(json.dumps(data))
        if result["Normal"]["mean"] != 30:
            raise ValueError("Expected mean 30")

    def test_invalid_json_raises(self) -> None:
        """Raise ValueError on invalid JSON."""
        try:
            json_to_fit_result("not valid json")
            raise RuntimeError("Should have raised ValueError")
        except ValueError:
            pass


class TestFitResultToSummary(unittest.TestCase):
    """Tests for fit_result_to_summary function."""

    def test_summary_includes_best_fitting(self) -> None:
        """Include best fitting distribution in summary."""
        result = {
            "best_fitting": "gamma",
            "Gamma": {"shape": 3.0, "rate": 0.1},
            "ssq": {"gamma": 0.001},
        }
        summary = fit_result_to_summary(result)
        if "gamma" not in summary:
            raise ValueError("Expected 'gamma' in summary")


class TestSampleToJson(unittest.TestCase):
    """Tests for sample_to_json function."""

    def test_basic_sample_payload(self) -> None:
        """Produce valid JSON for sampleFit."""
        result = sample_to_json(
            fit_json={"vals": [20, 30, 50]},
            n=100,
        )
        parsed = json.loads(result)
        if parsed["action"] != "sampleFit":
            raise ValueError("Expected sampleFit action")
        if parsed["params"]["n"] != 100:
            raise ValueError("Expected n=100")


if __name__ == "__main__":
    unittest.main()
