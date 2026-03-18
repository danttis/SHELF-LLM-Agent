"""JSON serialization between Python dicts and R-compatible format."""

import json
from typing import Any, Dict, List, Optional, Union

Number = Union[int, float]


def elicitation_to_json(
    action: str,
    vals: List[Number],
    probs: List[Number],
    lower: Number = float("-inf"),
    upper: Number = float("inf"),
    tdf: int = 3,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Build JSON string for sending to R bridge.

    Args:
        action: The R function to call (e.g. 'fitdist').
        vals: Elicited parameter values.
        probs: Cumulative probabilities.
        lower: Lower limit for the quantity.
        upper: Upper limit for the quantity.
        tdf: Degrees of freedom for Student-t.
        extra: Additional parameters for specific actions.

    Returns:
        JSON string ready for R subprocess stdin.
    """
    params: Dict[str, Any] = {
        "vals": vals,
        "probs": probs,
        "lower": _safe_number(lower),
        "upper": _safe_number(upper),
        "tdf": tdf,
    }
    if extra:
        params.update(extra)

    payload = {"action": action, "params": params}
    return json.dumps(payload, ensure_ascii=False)


def precision_to_json(
    interval: List[Number],
    propvals: List[Number],
    propprobs: Optional[List[Number]] = None,
    med: Optional[Number] = None,
    trans: str = "identity",
    tdf: int = 3,
) -> str:
    """Build JSON string for fitprecision call.

    Args:
        interval: Endpoints of the interval [k1, k2].
        propvals: Two proportion values.
        propprobs: Two probabilities (default [0.05, 0.95]).
        med: Hypothetical population median.
        trans: Transform ('identity', 'log', 'logit').
        tdf: Degrees of freedom for log Student-t.

    Returns:
        JSON string for R subprocess stdin.
    """
    params: Dict[str, Any] = {
        "interval": interval,
        "propvals": propvals,
        "propprobs": propprobs or [0.05, 0.95],
        "trans": trans,
        "tdf": tdf,
    }
    if med is not None:
        params["med"] = med

    payload = {"action": "fitprecision", "params": params}
    return json.dumps(payload, ensure_ascii=False)


def dirichlet_to_json(
    marginals: List[Dict[str, Any]],
    categories: List[str],
    n_fitted: str = "opt",
) -> str:
    """Build JSON string for Dirichlet fitting.

    Args:
        marginals: List of dicts with vals/probs/lower/upper.
        categories: Category labels.
        n_fitted: Fitting method ('opt', 'min', 'med', 'mean').

    Returns:
        JSON string for R subprocess stdin.
    """
    payload = {
        "action": "fitdirichlet",
        "params": {
            "marginals": marginals,
            "categories": categories,
            "n_fitted": n_fitted,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def feedback_to_json(
    fit_json: Dict[str, Any],
    quantiles: Optional[List[Number]] = None,
    values: Optional[List[Number]] = None,
) -> str:
    """Build JSON string for feedback call.

    Args:
        fit_json: The fit result from a previous fitdist call.
        quantiles: Quantiles to report.
        values: Values to report P(X <= value).

    Returns:
        JSON string for R subprocess stdin.
    """
    payload = {
        "action": "feedback",
        "params": {
            "fit": fit_json,
            "quantiles": quantiles or [0.1, 0.9],
            "values": values,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def sample_to_json(
    fit_json: Dict[str, Any],
    n: int = 1000,
    expert: int = 1,
) -> str:
    """Build JSON string for sampleFit call.

    Args:
        fit_json: The fit result from a previous fitdist call.
        n: Number of samples to draw.
        expert: Expert index (1-based).

    Returns:
        JSON string for R subprocess stdin.
    """
    payload = {
        "action": "sampleFit",
        "params": {
            "fit": fit_json,
            "n": n,
            "expert": expert,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def json_to_fit_result(json_str: str) -> Dict[str, Any]:
    """Parse JSON output from R into a Python dict.

    Args:
        json_str: JSON string from R stdout.

    Returns:
        Parsed dictionary with fitted distribution params.

    Raises:
        ValueError: If JSON parsing fails.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Failed to parse R output as JSON: {}".format(
                str(exc)
            )
        ) from exc


def fit_result_to_summary(result: Dict[str, Any]) -> str:
    """Format a fit result dict as a readable summary.

    Args:
        result: Parsed fit result dictionary.

    Returns:
        Human-readable multi-line summary string.
    """
    lines = ["=== Fitted Distribution Summary ==="]

    best = result.get("best_fitting", "unknown")
    lines.append("Best fitting: {}".format(best))
    lines.append("")

    dist_keys = [
        "Normal", "Student.t", "Gamma",
        "Log.normal", "Beta", "mirrorgamma",
        "mirrorlognormal",
    ]
    for key in dist_keys:
        params = result.get(key)
        if params and any(
            v is not None for v in params.values()
        ):
            param_str = ", ".join(
                "{}: {:.4f}".format(k, v)
                for k, v in params.items()
                if v is not None
            )
            lines.append("{}: {}".format(key, param_str))

    ssq = result.get("ssq", {})
    if ssq:
        lines.append("")
        lines.append("Sum of squared errors:")
        for dist_name, val in ssq.items():
            if val is not None:
                lines.append(
                    "  {}: {:.6f}".format(dist_name, val)
                )

    return "\n".join(lines)


def _safe_number(value: Number) -> Any:
    """Convert infinity to R-compatible string.

    Args:
        value: Numeric value, possibly infinite.

    Returns:
        The value itself, or a string 'Inf'/'-Inf'.
    """
    if value == float("inf"):
        return "Inf"
    if value == float("-inf"):
        return "-Inf"
    return value
