"""Configuration and audit logging for shelf_llm_agent."""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LlmConfig:
    """Configuration for the LLM provider.

    Attributes:
        provider: LLM provider name (openai, anthropic, ollama).
        model: Model identifier string.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility.
        api_key: API key (read from env if not set).
        base_url: Base URL for API endpoint.
        max_tokens: Maximum tokens in response.
    """

    provider: str = "ollama"
    model: str = "gpt-oss:120b-cloud"
    temperature: float = 0.2
    seed: int = 42
    api_key: Optional[str] = None
    base_url: Optional[str] = "https://ollama.com/v1"
    max_tokens: int = 2048

    def resolve_api_key(self) -> str:
        """Resolve API key from config or environment.

        Returns:
            The resolved API key string.

        Raises:
            ValueError: If no API key is found.
        """
        if self.api_key:
            return self.api_key
        env_map = {
            "ollama": "OLLAMA_API_KEY",
        }
        env_var = env_map.get(self.provider, "OLLAMA_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                "No API key found. Set {} or pass api_key.".format(
                    env_var
                )
            )
        return key


@dataclass
class AgentConfig:
    """Top-level configuration for the agent.

    Attributes:
        llm: LLM provider configuration.
        r_executable: Path to the Rscript binary.
        r_seed: Random seed passed to R for sampling.
        audit_log_dir: Directory for audit log files.
        shelf_version: SHELF package version string.
        agent_version: Agent version string.
    """

    llm: LlmConfig = field(default_factory=LlmConfig)
    r_executable: str = "Rscript"
    r_seed: int = 42
    audit_log_dir: str = "./elicitation_logs"
    shelf_version: str = "1.12.1"
    agent_version: str = "0.1.0"


class AuditLogger:
    """Logs every step of an elicitation session for traceability.

    Attributes:
        session_id: Unique session identifier.
        config: Agent configuration snapshot.
        steps: List of recorded step dictionaries.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the audit logger.

        Args:
            config: Agent configuration to snapshot.
        """
        self.session_id: str = str(uuid.uuid4())
        self.config = config
        self.steps: List[Dict[str, Any]] = []
        self._start_time = datetime.now(timezone.utc).isoformat()

    def log_step(
        self,
        step_type: str,
        prompt: str = "",
        response_raw: str = "",
        parsed_values: Optional[Dict[str, Any]] = None,
        r_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one step of the elicitation session.

        Args:
            step_type: Kind of step (e.g. 'llm_call', 'r_call').
            prompt: The prompt sent to the LLM.
            response_raw: Raw LLM response text.
            parsed_values: Parsed values from the response.
            r_result: Result from R subprocess call.
        """
        step = {
            "step": len(self.steps) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": step_type,
            "prompt": prompt,
            "response_raw": response_raw,
            "parsed_values": parsed_values or {},
            "r_result": r_result or {},
        }
        self.steps.append(step)
        logger.info(
            "Audit step %d: %s", step["step"], step_type
        )

    def save(self, path: Optional[str] = None) -> str:
        """Save the full audit log to a JSON file.

        Args:
            path: Optional file path. Auto-generated if None.

        Returns:
            The path where the log was saved.
        """
        log_dir = Path(
            path or self.config.audit_log_dir
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = "elicitation_{session}_{ts}.json".format(
            session=self.session_id[:8],
            ts=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        )
        filepath = log_dir / filename

        payload = {
            "session_id": self.session_id,
            "start_time": self._start_time,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "config": {
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "llm_temperature": self.config.llm.temperature,
                "llm_seed": self.config.llm.seed,
                "r_seed": self.config.r_seed,
                "shelf_version": self.config.shelf_version,
                "agent_version": self.config.agent_version,
            },
            "steps": self.steps,
        }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Audit log saved to %s", filepath)
        return str(filepath)
