"""R Bridge — calls SHELF R functions via subprocess + JSON."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BRIDGE_SCRIPT = Path(__file__).parent / "shelf_bridge.R"


class RBridgeError(Exception):
    """Raised when the R subprocess fails."""


class RBridge:
    """Execute SHELF R functions through a subprocess.

    Attributes:
        r_executable: Path to the Rscript binary.
        bridge_script: Path to the R helper script.
        r_seed: Random seed for R sampling operations.
        timeout: Subprocess timeout in seconds.
    """

    def __init__(
        self,
        r_executable: str = "Rscript",
        r_seed: int = 42,
        timeout: int = 60,
    ) -> None:
        """Initialize the R bridge.

        Args:
            r_executable: Path or name of Rscript binary.
            r_seed: Random seed for reproducible R calls.
            timeout: Maximum seconds to wait for R process.
        """
        self.r_executable = r_executable
        self.bridge_script = str(_BRIDGE_SCRIPT)
        self.r_seed = r_seed
        self.timeout = timeout

    def _call_r(self, json_input: str) -> Dict[str, Any]:
        """Send JSON to R subprocess, return parsed output.

        Args:
            json_input: JSON string to pass via stdin.

        Returns:
            Parsed JSON dictionary from R stdout.

        Raises:
            RBridgeError: If R process fails or returns
                invalid JSON.
        """
        cmd = [self.r_executable, "--vanilla", self.bridge_script]
        logger.debug("Running R: %s", " ".join(cmd))
        logger.debug("R input: %s", json_input)

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                input=json_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RBridgeError(
                "R process timed out after {} seconds".format(
                    self.timeout
                )
            ) from exc
        except FileNotFoundError as exc:
            raise RBridgeError(
                "Rscript not found at '{}'".format(
                    self.r_executable
                )
            ) from exc

        if result.returncode != 0:
            raise RBridgeError(
                "R process failed (code {}):\n{}".format(
                    result.returncode, result.stderr
                )
            )

        stdout = result.stdout.strip()
        if not stdout:
            raise RBridgeError(
                "R process returned empty output.\n"
                "stderr: {}".format(result.stderr)
            )

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RBridgeError(
                "R output is not valid JSON:\n{}\n"
                "stderr: {}".format(stdout, result.stderr)
            ) from exc

    def call_fitdist(
        self,
        vals: list,
        probs: list,
        lower: float = float("-inf"),
        upper: float = float("inf"),
        tdf: int = 3,
    ) -> Dict[str, Any]:
        """Fit distributions to elicited probabilities.

        Args:
            vals: Elicited parameter values.
            probs: Cumulative probabilities.
            lower: Lower parameter limit.
            upper: Upper parameter limit.
            tdf: Student-t degrees of freedom.

        Returns:
            Dictionary with fitted distribution parameters.
        """
        payload = {
            "action": "fitdist",
            "params": {
                "vals": vals,
                "probs": probs,
                "lower": _safe_num(lower),
                "upper": _safe_num(upper),
                "tdf": tdf,
            },
            "seed": self.r_seed,
        }
        return self._call_r(json.dumps(payload))

    def call_feedback(
        self,
        fit_result: Dict[str, Any],
        quantiles: Optional[list] = None,
        values: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Get feedback quantiles/probabilities from fit.

        Args:
            fit_result: Result from a previous fitdist call.
            quantiles: Quantiles to report.
            values: Values for P(X <= value) reporting.

        Returns:
            Dictionary with fitted quantiles and probabilities.
        """
        payload = {
            "action": "feedback",
            "params": {
                "fit": fit_result,
                "quantiles": quantiles or [0.1, 0.9],
                "values": values,
            },
            "seed": self.r_seed,
        }
        return self._call_r(json.dumps(payload))

    def call_sample_fit(
        self,
        fit_result: Dict[str, Any],
        n: int = 1000,
        expert: int = 1,
    ) -> Dict[str, Any]:
        """Sample from fitted distributions.

        Args:
            fit_result: Result from a previous fitdist call.
            n: Number of samples.
            expert: Expert index (1-based).

        Returns:
            Dictionary with sampled values per distribution.
        """
        payload = {
            "action": "sampleFit",
            "params": {
                "fit": fit_result,
                "n": n,
                "expert": expert,
            },
            "seed": self.r_seed,
        }
        return self._call_r(json.dumps(payload))

    def call_fitprecision(
        self,
        interval: list,
        propvals: list,
        propprobs: Optional[list] = None,
        med: Optional[float] = None,
        trans: str = "identity",
        tdf: int = 3,
    ) -> Dict[str, Any]:
        """Fit distributions to precision judgements.

        Args:
            interval: Interval endpoints [k1, k2].
            propvals: Two proportion values.
            propprobs: Two probabilities.
            med: Hypothetical population median.
            trans: Transform type.
            tdf: Degrees of freedom for log Student-t.

        Returns:
            Dictionary with fitted precision distributions.
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

        payload = {
            "action": "fitprecision",
            "params": params,
            "seed": self.r_seed,
        }
        return self._call_r(json.dumps(payload))

    def check_r_available(self) -> bool:
        """Test if R and SHELF are available.

        Returns:
            True if both Rscript and SHELF are accessible.
        """
        try:
            result = subprocess.run(  # noqa: S603
                [
                    self.r_executable, "--vanilla", "-e",
                    'library(SHELF); cat("OK")',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() == "OK"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


def _safe_num(value: float) -> Any:
    """Convert Python infinities to R-compatible strings.

    Args:
        value: Numeric value, possibly infinite.

    Returns:
        The value or an R-compatible infinity string.
    """
    if value == float("inf"):
        return "Inf"
    if value == float("-inf"):
        return "-Inf"
    return value
