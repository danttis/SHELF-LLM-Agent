"""CLI entry point for shelf_llm_agent."""

import argparse
import json
import logging
import subprocess
import sys
from typing import List, Optional

from shelf_llm_agent.agent import ShelfLlmAgent
from shelf_llm_agent.config import AgentConfig, LlmConfig
from shelf_llm_agent.r_bridge import RBridge
from shelf_llm_agent.serializer import fit_result_to_summary

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="shelf-llm-agent",
        description=(
            "LLM-assisted prior elicitation for SHELF. "
            "Choose between traditional R Shiny elicitation "
            "or LLM-assisted mode."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["traditional", "llm"],
        default="llm",
        help="Elicitation mode (default: llm)",
    )
    parser.add_argument(
        "--elicitation-type",
        choices=["single", "precision", "dirichlet"],
        default="single",
        help="Type of elicitation to perform (default: single)",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        default="",
        help="Description of the uncertain parameter",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Domain context for the elicitation",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Category names for Dirichlet elicitation",
    )
    parser.add_argument(
        "--interval",
        type=float,
        nargs=2,
        default=None,
        help="Interval [lower, upper] for precision",
    )
    parser.add_argument(
        "--median",
        type=float,
        default=None,
        help="Hypothetical median for precision",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        help="LLM provider (openai, anthropic, ollama)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o",
        help="LLM model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--r-executable",
        type=str,
        default="Rscript",
        help="Path to Rscript binary",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./elicitation_logs",
        help="Directory for audit log files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check R availability without running elicitation",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for LLM API endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def run_traditional(r_executable: str = "Rscript") -> None:
    """Launch the traditional SHELF Shiny app.

    Args:
        r_executable: Path to Rscript binary.
    """
    print("Launching traditional SHELF Shiny app...")
    cmd = [
        r_executable, "--vanilla", "-e",
        'library(SHELF); elicit()',
    ]
    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except FileNotFoundError:
        print(
            "Error: Rscript not found at '{}'".format(
                r_executable
            )
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print("R process failed: {}".format(exc))
        sys.exit(1)


def run_llm_elicitation(args: argparse.Namespace) -> None:
    """Run LLM-assisted elicitation.

    Args:
        args: Parsed command-line arguments.
    """
    llm_config = LlmConfig()

    if args.llm_provider != "openai":
        llm_config.provider = args.llm_provider
    if args.llm_model != "gpt-4o":
        llm_config.model = args.llm_model
    if args.temperature != 0.2:
        llm_config.temperature = args.temperature
    if args.seed != 42:
        llm_config.seed = args.seed
    if args.base_url is not None:
        llm_config.base_url = args.base_url
    if args.api_key is not None:
        llm_config.api_key = args.api_key

    config = AgentConfig(
        llm=llm_config,
        r_executable=args.r_executable,
        r_seed=args.seed,
        audit_log_dir=args.log_dir,
    )

    agent = ShelfLlmAgent(config=config)

    if args.dry_run:
        _check_availability(agent)
        return

    if args.elicitation_type == "single":
        _run_single(agent, args)
    elif args.elicitation_type == "precision":
        _run_precision(agent, args)
    elif args.elicitation_type == "dirichlet":
        _run_dirichlet(agent, args)


def _check_availability(agent: ShelfLlmAgent) -> None:
    """Check if R and SHELF are available.

    Args:
        agent: The agent to check with.
    """
    print("Checking R + SHELF availability...")
    if agent.r_bridge.check_r_available():
        print("OK: R and SHELF are available.")
    else:
        print("FAIL: R or SHELF is not available.")
        sys.exit(1)


def _run_single(
    agent: ShelfLlmAgent,
    args: argparse.Namespace,
) -> None:
    """Execute a single distribution elicitation.

    Args:
        agent: Configured ShelfLlmAgent.
        args: Parsed command-line arguments.
    """
    if not args.parameter:
        print("Error: --parameter is required for single")
        sys.exit(1)

    print("Starting single distribution elicitation...")
    print(
        "Parameter: {}".format(args.parameter)
    )
    print("Context: {}".format(args.context or "(none)"))
    print("")

    result = agent.elicit_single(
        parameter_description=args.parameter,
        domain_context=args.context,
    )

    _display_result(result)
    log_path = agent.save_session()
    print("\nAudit log saved to: {}".format(log_path))


def _run_precision(
    agent: ShelfLlmAgent,
    args: argparse.Namespace,
) -> None:
    """Execute a precision elicitation.

    Args:
        agent: Configured ShelfLlmAgent.
        args: Parsed command-line arguments.
    """
    if not args.interval or args.median is None:
        print(
            "Error: --interval and --median required "
            "for precision"
        )
        sys.exit(1)

    print("Starting precision elicitation...")
    result = agent.elicit_precision(
        population_description=args.parameter or "population",
        median_value=args.median,
        lower_bound=args.interval[0],
        upper_bound=args.interval[1],
    )

    _display_result(result)
    log_path = agent.save_session()
    print("\nAudit log saved to: {}".format(log_path))


def _run_dirichlet(
    agent: ShelfLlmAgent,
    args: argparse.Namespace,
) -> None:
    """Execute a Dirichlet distribution elicitation.

    Args:
        agent: Configured ShelfLlmAgent.
        args: Parsed command-line arguments.
    """
    if not args.categories:
        print(
            "Error: --categories required for dirichlet"
        )
        sys.exit(1)

    print("Starting Dirichlet elicitation...")
    print(
        "Categories: {}".format(", ".join(args.categories))
    )

    result = agent.elicit_dirichlet(
        categories=args.categories,
        domain_context=args.context,
    )

    _display_result(result)
    log_path = agent.save_session()
    print("\nAudit log saved to: {}".format(log_path))


def _display_result(result: dict) -> None:
    """Display elicitation results to console.

    Args:
        result: Elicitation result dictionary.
    """
    print("\n" + "=" * 50)
    print("ELICITATION RESULTS")
    print("=" * 50)

    judgements = result.get("elicited_judgements", {})
    print("\nElicited judgements:")
    print(json.dumps(judgements, indent=2))

    fit = result.get("fit_result")
    if fit:
        print("\n" + fit_result_to_summary(fit))

    reasoning = judgements.get("reasoning", "")
    if reasoning:
        print("\nReasoning: {}".format(reasoning))


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point.

    Args:
        argv: Command-line arguments (uses sys.argv if None).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: "
               "%(message)s",
    )

    if args.mode == "traditional":
        run_traditional(args.r_executable)
    else:
        try:
            run_llm_elicitation(args)
        except Exception as exc:
            logger.error("Elicitation failed: %s", exc)
            print("\nError: {}".format(exc))
            sys.exit(1)


if __name__ == "__main__":
    main()
