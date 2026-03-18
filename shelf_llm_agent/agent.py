"""Main orchestrator for LLM-assisted SHELF elicitation."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from shelf_llm_agent.config import AgentConfig, AuditLogger
from shelf_llm_agent.prompts import (
    build_dirichlet_prompt,
    build_feedback_prompt,
    build_precision_prompt,
    build_single_elicitation_prompt,
    parse_llm_response,
    validate_precision_response,
    validate_single_response,
)
from shelf_llm_agent.r_bridge import RBridge, RBridgeError

logger = logging.getLogger(__name__)

LlmCallable = Callable[[str], str]


class ElicitationError(Exception):
    """Raised when the elicitation process fails."""


class ShelfLlmAgent:
    """Orchestrates LLM-assisted prior elicitation with SHELF.

    This agent:
    1. Sends structured prompts to an LLM
    2. Parses and validates the responses
    3. Passes valid judgements to SHELF R functions
    4. Optionally provides feedback and allows refinement

    Attributes:
        config: Agent configuration.
        r_bridge: Interface to R subprocess.
        audit: Audit logger instance.
        llm_call: Callable that takes a prompt, returns text.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_call: Optional[LlmCallable] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration. Uses defaults if None.
            llm_call: Function that sends a prompt to LLM and
                returns the response text. If None, a default
                OpenAI-compatible caller is created.
        """
        self.config = config or AgentConfig()
        self.r_bridge = RBridge(
            r_executable=self.config.r_executable,
            r_seed=self.config.r_seed,
        )
        self.audit = AuditLogger(self.config)

        if llm_call is not None:
            self.llm_call = llm_call
        else:
            self.llm_call = self._default_llm_call

    def _default_llm_call(self, prompt: str) -> str:
        """Default LLM call using LangChain ChatOpenAI.

        Uses langchain_openai with JSON response format
        for structured output from the LLM.

        Args:
            prompt: The prompt to send.

        Returns:
            The response text from the LLM.

        Raises:
            ElicitationError: If the API call fails.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ElicitationError(
                "langchain-openai package required. "
                "Install with: pip install langchain-openai"
            ) from exc

        api_key = self.config.llm.resolve_api_key()

        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            "model_kwargs": {
                "response_format": {"type": "json_object"},
            },
        }
        if self.config.llm.base_url:
            llm_kwargs["base_url"] = self.config.llm.base_url
        if self.config.llm.seed is not None:
            llm_kwargs["seed"] = self.config.llm.seed

        try:
            llm = ChatOpenAI(**llm_kwargs)
            response = llm.invoke(prompt)
            return response.content or ""
        except Exception as exc:
            raise ElicitationError(
                "LLM API call failed: {}".format(str(exc))
            ) from exc

    def _call_llm_with_audit(
        self,
        prompt: str,
        step_type: str = "llm_call",
    ) -> str:
        """Call LLM and log the interaction.

        Args:
            prompt: The prompt to send.
            step_type: Type label for the audit log.

        Returns:
            Raw LLM response text.
        """
        response = self.llm_call(prompt)
        self.audit.log_step(
            step_type=step_type,
            prompt=prompt,
            response_raw=response,
        )
        return response

    def elicit_single(
        self,
        parameter_description: str,
        domain_context: str = "",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Run a single-distribution elicitation session.

        Args:
            parameter_description: What the quantity represents.
            domain_context: Domain-specific background info.
            max_retries: Max attempts if LLM response invalid.

        Returns:
            Dictionary with fitted distributions and metadata.

        Raises:
            ElicitationError: If elicitation fails after retries.
        """
        prompt = build_single_elicitation_prompt(
            parameter_description=parameter_description,
            domain_context=domain_context,
        )

        parsed = self._get_valid_response(
            prompt=prompt,
            validator=validate_single_response,
            max_retries=max_retries,
        )

        try:
            fit_result = self.r_bridge.call_fitdist(
                vals=parsed["values"],
                probs=parsed["probabilities"],
                lower=parsed["lower"],
                upper=parsed["upper"],
            )
        except RBridgeError as exc:
            raise ElicitationError(
                "R fitting failed: {}".format(str(exc))
            ) from exc

        self.audit.log_step(
            step_type="r_fitdist",
            parsed_values=parsed,
            r_result=fit_result,
        )

        return {
            "elicitation_type": "single",
            "elicited_judgements": parsed,
            "fit_result": fit_result,
            "session_id": self.audit.session_id,
        }

    def elicit_precision(
        self,
        population_description: str,
        median_value: float,
        lower_bound: float,
        upper_bound: float,
        trans: str = "identity",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Run a precision elicitation session.

        Args:
            population_description: Description of population.
            median_value: Hypothetical population median.
            lower_bound: Lower interval endpoint.
            upper_bound: Upper interval endpoint.
            trans: Transform type.
            max_retries: Max attempts if LLM response invalid.

        Returns:
            Dictionary with fitted precision distributions.

        Raises:
            ElicitationError: If elicitation fails.
        """
        prompt = build_precision_prompt(
            population_description=population_description,
            median_value=median_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        parsed = self._get_valid_response(
            prompt=prompt,
            validator=validate_precision_response,
            max_retries=max_retries,
        )

        try:
            fit_result = self.r_bridge.call_fitprecision(
                interval=parsed["interval"],
                propvals=parsed["prop_values"],
                propprobs=parsed.get("prop_probs", [0.05, 0.95]),
                trans=trans,
            )
        except RBridgeError as exc:
            raise ElicitationError(
                "R precision fitting failed: {}".format(
                    str(exc)
                )
            ) from exc

        self.audit.log_step(
            step_type="r_fitprecision",
            parsed_values=parsed,
            r_result=fit_result,
        )

        return {
            "elicitation_type": "precision",
            "elicited_judgements": parsed,
            "fit_result": fit_result,
            "session_id": self.audit.session_id,
        }

    def elicit_dirichlet(
        self,
        categories: List[str],
        domain_context: str = "",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Run a Dirichlet distribution elicitation session.

        Each category's marginal is elicited via the LLM, then
        each marginal is fitted individually using fitdist.

        Args:
            categories: List of category names.
            domain_context: Domain-specific background info.
            max_retries: Max attempts if LLM response invalid.

        Returns:
            Dictionary with marginal fits per category.

        Raises:
            ElicitationError: If elicitation fails.
        """
        prompt = build_dirichlet_prompt(
            categories=categories,
            domain_context=domain_context,
        )

        response_text = self._call_llm_with_audit(
            prompt, step_type="llm_dirichlet"
        )
        parsed = parse_llm_response(response_text)
        if parsed is None:
            raise ElicitationError(
                "Failed to parse Dirichlet LLM response"
            )

        marginals = parsed.get("marginals", [])
        if len(marginals) != len(categories):
            raise ElicitationError(
                "Expected {} marginals, got {}".format(
                    len(categories), len(marginals)
                )
            )

        marginal_fits = []
        for i, marginal in enumerate(marginals):
            try:
                fit = self.r_bridge.call_fitdist(
                    vals=marginal["values"],
                    probs=marginal["probabilities"],
                    lower=marginal.get("lower", 0),
                    upper=marginal.get("upper", 1),
                )
                marginal_fits.append({
                    "category": categories[i],
                    "judgements": marginal,
                    "fit": fit,
                })
            except RBridgeError as exc:
                logger.warning(
                    "Failed to fit marginal %d: %s", i, exc
                )
                marginal_fits.append({
                    "category": categories[i],
                    "judgements": marginal,
                    "fit": None,
                    "error": str(exc),
                })

        self.audit.log_step(
            step_type="r_fitdirichlet",
            parsed_values=parsed,
            r_result={"marginal_fits": marginal_fits},
        )

        return {
            "elicitation_type": "dirichlet",
            "categories": categories,
            "elicited_judgements": parsed,
            "marginal_fits": marginal_fits,
            "session_id": self.audit.session_id,
        }

    def provide_feedback(
        self,
        fit_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Show fit results to LLM and get acceptance/revision.

        Args:
            fit_result: Result dictionary from a fitting call.

        Returns:
            Dict with 'accept' boolean and optional revisions.

        Raises:
            ElicitationError: If feedback interaction fails.
        """
        prompt = build_feedback_prompt(fit_result)
        response_text = self._call_llm_with_audit(
            prompt, step_type="llm_feedback"
        )
        parsed = parse_llm_response(response_text)
        if parsed is None:
            return {"accept": True, "reasoning": "No response"}

        self.audit.log_step(
            step_type="feedback_response",
            parsed_values=parsed,
        )
        return parsed

    def save_session(self, path: Optional[str] = None) -> str:
        """Save the full audit log to disk.

        Args:
            path: Directory for the log file.

        Returns:
            Path to the saved audit log file.
        """
        return self.audit.save(path)

    def _get_valid_response(
        self,
        prompt: str,
        validator: Callable[
            [Dict[str, Any]], List[str]
        ],
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Call LLM repeatedly until valid response received.

        Args:
            prompt: The prompt to send.
            validator: Function returning error list.
            max_retries: Maximum retry attempts.

        Returns:
            Parsed and validated response dictionary.

        Raises:
            ElicitationError: If all retries exhausted.
        """
        for attempt in range(max_retries + 1):
            response_text = self._call_llm_with_audit(
                prompt,
                step_type="llm_elicit_attempt_{}".format(
                    attempt + 1
                ),
            )
            parsed = parse_llm_response(response_text)
            if parsed is None:
                logger.warning(
                    "Attempt %d: could not parse response",
                    attempt + 1,
                )
                continue

            errors = validator(parsed)
            if not errors:
                return parsed

            logger.warning(
                "Attempt %d validation errors: %s",
                attempt + 1,
                errors,
            )

            prompt = (
                "Your previous response had errors:\n"
                "{}\n\n"
                "Please fix and respond again with ONLY "
                "valid JSON.".format(
                    "\n".join("- {}".format(e) for e in errors)
                )
            )

        raise ElicitationError(
            "Failed to get valid LLM response after {} attempts"
            .format(max_retries + 1)
        )
