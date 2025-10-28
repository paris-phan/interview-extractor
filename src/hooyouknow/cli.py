"""
Command-line interface for the HooYouKnow interview extraction pipeline.

This module provides a user-friendly CLI for processing interviews.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from .pipeline import InterviewPipeline
from .utils import load_config, load_env_vars


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print a header with formatting."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print("=" * len(text))


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}", file=sys.stderr)


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def setup_cli_logging(verbose: bool = False, quiet: bool = False):
    """
    Set up logging for CLI.

    Args:
        verbose: Enable verbose logging.
        quiet: Enable quiet mode (errors only).
    """
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_process(args):
    """
    Handle the 'process' command.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        print_header("HooYouKnow Interview Processor")

        # Initialize pipeline
        print_info("Initializing pipeline...")
        pipeline = InterviewPipeline(config_path=args.config, env_path=args.env)

        # Dry run - estimate costs
        if args.dry_run:
            print_info("Performing dry run (cost estimation only)...")
            costs = pipeline.estimate_cost(
                audio_path=args.audio_path if not args.transcript_path else None,
                transcript=None,  # Would need to load transcript for accurate estimate
            )
            print("\nEstimated Costs:")
            print(f"  Transcription: ${costs['transcription']:.4f}")
            print(f"  Extraction:    ${costs['extraction']:.4f}")
            print(f"  Total:         ${costs['total']:.4f}")
            return 0

        # Process the interview
        print_info(f"Processing interview for: {args.guest_name}")

        result = pipeline.process_interview(
            audio_path=args.audio_path,
            transcript_path=args.transcript_path,
            guest_name=args.guest_name,
            guest_title=args.guest_title or "",
            episode_number=args.episode_number,
            output_dir=args.output_dir,
        )

        # Display results
        if result["success"]:
            print_success("Interview processed successfully!")

            print("\nOutput Files:")
            for format_name, file_path in result["output_files"].items():
                print(f"  {format_name}: {file_path}")

            stats = result["stats"]
            print("\nProcessing Statistics:")
            print(f"  Total time: {stats['timing']['total']:.2f}s")
            print(f"  Total cost: ${stats['costs']['total']:.4f}")
            print(f"  Tokens:     {stats['tokens']['total']:,}")

            if result.get("transcript_path"):
                print(f"\nTranscript saved to: {result['transcript_path']}")

            return 0
        else:
            print_error(f"Processing failed: {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        print_warning("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_batch(args):
    """
    Handle the 'batch' command.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        print_header("HooYouKnow Batch Processor")

        # Initialize pipeline
        print_info("Initializing pipeline...")
        pipeline = InterviewPipeline(config_path=args.config, env_path=args.env)

        # Process batch
        print_info(f"Processing interviews from: {args.input_dir}")
        print_info(f"Using guest info from: {args.guest_info_file}")

        results = pipeline.batch_process(
            input_dir=args.input_dir,
            guest_info_file=args.guest_info_file,
            output_dir=args.output_dir,
            save_intermediates=not args.no_intermediates,
            continue_on_error=not args.stop_on_error,
        )

        # Generate and display report
        print("\n")
        report = pipeline.generate_batch_report(results, report_path=args.report_file)

        if not args.report_file:
            # Print report to console if not saving to file
            print(report)

        # Summary
        if results["success_rate"] == 1.0:
            print_success(f"All {results['successful']} interview(s) processed successfully!")
        elif results["successful"] > 0:
            print_warning(
                f"Processed {results['successful']}/{results['total_processed']} interviews successfully"
            )
        else:
            print_error("All interviews failed to process")

        # Return appropriate exit code
        if results["failed"] > 0:
            return 1
        return 0

    except KeyboardInterrupt:
        print_warning("\nBatch processing interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate_config(args):
    """
    Handle the 'validate-config' command.

    Args:
        args: Parsed command-line arguments.
    """
    try:
        print_header("Configuration Validator")

        # Load configuration
        print_info("Loading configuration...")
        config = load_config(args.config)
        print_success("Configuration file loaded successfully")

        # Load environment variables
        print_info("Loading environment variables...")
        env_vars = load_env_vars(args.env)

        # Validate required sections
        print_info("Validating configuration structure...")

        required_sections = ["sections", "extraction", "transcription", "output"]
        missing_sections = [sec for sec in required_sections if sec not in config]

        if missing_sections:
            print_error(f"Missing required sections: {missing_sections}")
            return 1

        print_success("All required configuration sections present")

        # Validate sections
        sections = config.get("sections", [])
        print_info(f"Found {len(sections)} newsletter section(s)")

        for i, section in enumerate(sections, 1):
            section_name = section.get("name", f"Section {i}")
            required_fields = ["name", "title", "extraction_prompt"]
            missing_fields = [f for f in required_fields if not section.get(f)]

            if missing_fields:
                print_warning(f"  Section '{section_name}' missing fields: {missing_fields}")
            else:
                print(f"  ✓ {section_name}")

        # Validate API keys
        print_info("Checking API keys...")

        openai_key = env_vars.get("openai_api_key")
        anthropic_key = env_vars.get("anthropic_api_key")
        provider = env_vars.get("llm_provider", "claude")

        if not openai_key:
            print_warning("OpenAI API key not set (required for transcription)")
        else:
            print_success("OpenAI API key is set")

        if provider == "claude":
            if not anthropic_key:
                print_error("Anthropic API key not set (required when using Claude)")
                return 1
            else:
                print_success("Anthropic API key is set")
        elif provider == "openai":
            if not openai_key:
                print_error("OpenAI API key not set (required when using OpenAI)")
                return 1

        # Validate extraction settings
        print_info("Validating extraction settings...")
        extraction = config.get("extraction", {})

        if provider not in extraction:
            print_warning(f"No configuration found for provider '{provider}'")
        else:
            provider_config = extraction.get(provider, {})
            model = provider_config.get("model")
            if model:
                print(f"  Model: {model}")
            else:
                print_warning(f"  No model specified for {provider}")

        # Summary
        print("\n")
        print_success("Configuration validation complete!")
        print_info(f"Configuration is valid and ready to use")

        return 0

    except FileNotFoundError as e:
        print_error(f"Configuration file not found: {e}")
        return 1
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="hooyouknow",
        description="HooYouKnow Interview Extraction Pipeline - Convert interview audio to newsletters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline configuration file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file (default: config/.env)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (errors only)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a single interview",
        description="Process a single interview audio file into a newsletter",
    )
    process_parser.add_argument(
        "audio_path",
        type=str,
        nargs="?",
        help="Path to audio file (required unless --transcript-path is provided)",
    )
    process_parser.add_argument(
        "--guest-name",
        "-n",
        type=str,
        required=True,
        help="Name of the interview guest",
    )
    process_parser.add_argument(
        "--guest-title",
        "-t",
        type=str,
        default="",
        help="Title/position of the guest",
    )
    process_parser.add_argument(
        "--episode-number",
        "-e",
        type=int,
        help="Episode number",
    )
    process_parser.add_argument(
        "--transcript-path",
        type=str,
        help="Path to existing transcript (skip transcription)",
    )
    process_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: outputs)",
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs without processing",
    )
    process_parser.set_defaults(func=cmd_process)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple interviews in batch",
        description="Process multiple interview audio files from a directory",
    )
    batch_parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing audio files",
    )
    batch_parser.add_argument(
        "--guest-info-file",
        "-g",
        type=str,
        required=True,
        help="Path to guest information JSON file",
    )
    batch_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory (default: outputs)",
    )
    batch_parser.add_argument(
        "--report-file",
        "-r",
        type=str,
        help="Path to save batch processing report",
    )
    batch_parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Don't save intermediate files (transcripts, sections)",
    )
    batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error (default: continue)",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # Validate-config command
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate configuration files",
        description="Validate pipeline configuration and check API keys",
    )
    validate_parser.set_defaults(func=cmd_validate_config)

    return parser


def main():
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    setup_cli_logging(verbose=args.verbose, quiet=args.quiet)

    # If no command specified, print help
    if not args.command:
        parser.print_help()
        return 1

    # Execute the command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print_warning("\n\nOperation interrupted by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
