"""Unit tests for the R bridge module."""

import json
import unittest
from unittest.mock import MagicMock, patch

from shelf_llm_agent.r_bridge import RBridge, RBridgeError


class TestRBridgeCallR(unittest.TestCase):
    """Tests for RBridge._call_r method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.bridge = RBridge(r_executable="Rscript")

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_successful_call(
        self, mock_run: MagicMock,
    ) -> None:
        """Return parsed JSON on success."""
        expected = {"Normal": {"mean": 30, "sd": 10}}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(expected),
            stderr="",
        )

        result = self.bridge._call_r('{"action": "test"}')
        if result["Normal"]["mean"] != 30:
            raise ValueError("Expected mean=30")

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_nonzero_exit_raises(
        self, mock_run: MagicMock,
    ) -> None:
        """Raise RBridgeError on non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error in R",
        )

        try:
            self.bridge._call_r('{"action": "test"}')
            raise RuntimeError("Should have raised")
        except RBridgeError:
            pass

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_empty_output_raises(
        self, mock_run: MagicMock,
    ) -> None:
        """Raise RBridgeError on empty stdout."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        try:
            self.bridge._call_r('{"action": "test"}')
            raise RuntimeError("Should have raised")
        except RBridgeError:
            pass

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_invalid_json_raises(
        self, mock_run: MagicMock,
    ) -> None:
        """Raise RBridgeError on invalid JSON output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not json",
            stderr="",
        )

        try:
            self.bridge._call_r('{"action": "test"}')
            raise RuntimeError("Should have raised")
        except RBridgeError:
            pass


class TestRBridgeFitdist(unittest.TestCase):
    """Tests for RBridge.call_fitdist method."""

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_fitdist_passes_correct_params(
        self, mock_run: MagicMock,
    ) -> None:
        """Send correct JSON payload to R."""
        expected = {
            "best_fitting": "gamma",
            "Gamma": {"shape": 3, "rate": 0.1},
        }
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(expected),
            stderr="",
        )

        bridge = RBridge()
        result = bridge.call_fitdist(
            vals=[20, 30, 50],
            probs=[0.25, 0.5, 0.75],
            lower=0,
            upper=100,
        )

        if result["best_fitting"] != "gamma":
            raise ValueError("Expected gamma")

        call_args = mock_run.call_args
        input_json = json.loads(call_args.kwargs["input"])
        if input_json["action"] != "fitdist":
            raise ValueError("Expected fitdist action")
        if input_json["params"]["vals"] != [20, 30, 50]:
            raise ValueError("Wrong vals passed")


class TestRBridgeAvailability(unittest.TestCase):
    """Tests for RBridge.check_r_available method."""

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_available_returns_true(
        self, mock_run: MagicMock,
    ) -> None:
        """Return True when R and SHELF are available."""
        mock_run.return_value = MagicMock(
            stdout="OK",
            stderr="",
        )

        bridge = RBridge()
        if not bridge.check_r_available():
            raise ValueError("Expected True")

    @patch("shelf_llm_agent.r_bridge.subprocess.run")
    def test_not_found_returns_false(
        self, mock_run: MagicMock,
    ) -> None:
        """Return False when Rscript not found."""
        mock_run.side_effect = FileNotFoundError()

        bridge = RBridge()
        if bridge.check_r_available():
            raise ValueError("Expected False")


if __name__ == "__main__":
    unittest.main()
