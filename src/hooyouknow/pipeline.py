"""
Pipeline orchestrator for the HooYouKnow interview extraction system.

This module coordinates the entire workflow from audio transcription through
content extraction to newsletter formatting.
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .utils import (
    load_config,
    load_env_vars,
    setup_logging,
    ProcessingStats,
    ensure_directory,
    sanitize_filename,
)
from .transcription import AudioTranscriber
from .extraction import ContentExtractor
from .formatting import NewsletterFormatter


logger = logging.getLogger("hooyouknow.pipeline")


class InterviewPipeline:
    """
    Main pipeline orchestrator for processing interview audio into newsletters.

    This class coordinates the three-stage pipeline:
    1. Transcription (audio -> text)
    2. Extraction (text -> structured sections)
    3. Formatting (sections -> newsletter files)
    """

    def __init__(self, config_path: Optional[str] = None, env_path: Optional[str] = None):
        """
        Initialize the interview processing pipeline.

        Args:
            config_path: Optional path to configuration file.
            env_path: Optional path to .env file.

        Raises:
            ValueError: If required configuration or API keys are missing.
        """
        # Load configuration
        logger.info("Initializing InterviewPipeline...")
        self.config = load_config(config_path)
        self.env_vars = load_env_vars(env_path)

        # Set up logging
        log_config = self.config.get("logging", {})
        log_level = self.env_vars.get("log_level", log_config.get("level", "INFO"))
        log_file = log_config.get("log_file") if log_config.get("log_to_file") else None
        setup_logging(log_level, log_file)

        # Get output directory
        self.output_dir = self.env_vars.get("output_dir", "outputs")
        ensure_directory(self.output_dir)

        # Initialize components
        self.transcriber = None
        self.extractor = None
        self.formatter = None
        self._initialize_components()

        logger.info("InterviewPipeline initialized successfully")

    def _initialize_components(self):
        """
        Initialize all pipeline components (transcriber, extractor, formatter).

        Raises:
            ValueError: If required API keys are missing.
        """
        logger.info("Initializing pipeline components...")

        # Get API keys
        openai_key = self.env_vars.get("openai_api_key")
        anthropic_key = self.env_vars.get("anthropic_api_key")
        provider = self.env_vars.get("llm_provider", "claude")

        # Initialize transcriber
        if not openai_key:
            raise ValueError("OpenAI API key is required for transcription (OPENAI_API_KEY)")
        self.transcriber = AudioTranscriber(openai_key, self.config)
        logger.info("AudioTranscriber initialized")

        # Initialize extractor
        if provider == "claude" and not anthropic_key:
            raise ValueError("Anthropic API key is required when using Claude (ANTHROPIC_API_KEY)")
        elif provider == "openai" and not openai_key:
            raise ValueError("OpenAI API key is required when using OpenAI (OPENAI_API_KEY)")

        self.extractor = ContentExtractor(openai_key, anthropic_key, provider, self.config)
        logger.info(f"ContentExtractor initialized with provider: {provider}")

        # Initialize formatter
        self.formatter = NewsletterFormatter(self.config)
        logger.info("NewsletterFormatter initialized")

    def process_interview(
        self,
        audio_path: Optional[str] = None,
        transcript_path: Optional[str] = None,
        guest_name: str = "Unknown Guest",
        guest_title: str = "",
        episode_number: Optional[int] = None,
        output_dir: Optional[str] = None,
        save_intermediates: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single interview through the complete pipeline.

        Args:
            audio_path: Path to audio file (required unless transcript_path provided).
            transcript_path: Optional path to existing transcript (skip transcription).
            guest_name: Name of the interview guest.
            guest_title: Title/position of the guest.
            episode_number: Optional episode number.
            output_dir: Optional output directory (overrides default).
            save_intermediates: If True, save intermediate files (transcript, sections).

        Returns:
            Dictionary with processing results:
            - success: Boolean indicating overall success
            - output_files: Dict of format names to file paths
            - stats: ProcessingStats object with metrics
            - transcript_path: Path to saved transcript
            - sections: Extracted sections dict

        Raises:
            ValueError: If neither audio_path nor transcript_path provided.
        """
        if not audio_path and not transcript_path:
            raise ValueError("Either audio_path or transcript_path must be provided")

        logger.info(f"Starting interview processing for: {guest_name}")
        stats = ProcessingStats()
        overall_start = time.time()

        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir

        # Sanitize guest name for file naming
        safe_name = sanitize_filename(guest_name)

        try:
            # Stage 1: Transcription
            if transcript_path:
                logger.info(f"Using existing transcript: {transcript_path}")
                transcript_text = self._load_transcript(transcript_path)
                transcript_file = transcript_path
            else:
                logger.info("Stage 1/3: Transcribing audio...")
                transcript_result, transcript_cost, transcript_time = self._transcribe_stage(audio_path)
                transcript_text = transcript_result["text"]
                stats.add_transcription_stats(
                    duration=transcript_result.get("duration", 0) / 60.0,
                    cost=transcript_cost,
                    time_elapsed=transcript_time,
                )

                # Save transcript if requested
                if save_intermediates:
                    transcript_dir = Path(output_dir) / "transcripts"
                    ensure_directory(transcript_dir)
                    transcript_file = str(transcript_dir / f"{safe_name}_transcript.txt")
                    self.transcriber.save_transcript(
                        transcript_result,
                        transcript_file,
                        include_timestamps=False,
                    )
                    logger.info(f"Saved transcript to: {transcript_file}")
                else:
                    transcript_file = None

            # Stage 2: Extraction
            logger.info("Stage 2/3: Extracting sections...")
            sections, extraction_metadata, extraction_time = self._extract_stage(
                transcript_text, guest_name, guest_title
            )
            stats.add_extraction_stats(
                tokens_in=extraction_metadata.get("tokens_input", 0),
                tokens_out=extraction_metadata.get("tokens_output", 0),
                cost=extraction_metadata.get("cost", 0),
                time_elapsed=extraction_time,
            )

            # Save sections JSON if requested
            if save_intermediates:
                import json

                sections_dir = Path(output_dir) / "metadata"
                ensure_directory(sections_dir)
                sections_file = sections_dir / f"{safe_name}_sections.json"
                with open(sections_file, "w", encoding="utf-8") as f:
                    json.dump(sections, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved sections to: {sections_file}")

            # Stage 3: Formatting
            logger.info("Stage 3/3: Formatting newsletter...")
            format_start = time.time()
            formatted_content = self.formatter.format_newsletter(
                sections=sections,
                guest_name=guest_name,
                guest_title=guest_title,
                episode_number=episode_number,
                output_dir=output_dir,
            )
            output_files = self.formatter.save_newsletter(formatted_content, guest_name, output_dir)
            format_time = time.time() - format_start
            stats.add_formatting_stats(len(sections), format_time)

            # Mark success
            stats.mark_success()
            stats.total_time = time.time() - overall_start

            logger.info(f"Pipeline completed successfully in {stats.total_time:.2f}s")
            logger.info(f"Total cost: ${stats.total_cost:.4f}")

            return {
                "success": True,
                "output_files": output_files,
                "stats": stats.to_dict(),
                "transcript_path": transcript_file,
                "sections": sections,
                "guest_name": guest_name,
                "guest_title": guest_title,
            }

        except Exception as e:
            stats.add_error(str(e))
            stats.total_time = time.time() - overall_start
            logger.error(f"Pipeline failed: {e}")

            return {
                "success": False,
                "output_files": {},
                "stats": stats.to_dict(),
                "error": str(e),
                "guest_name": guest_name,
                "guest_title": guest_title,
            }

    def _transcribe_stage(self, audio_path: str) -> Tuple[Dict[str, Any], float, float]:
        """
        Execute transcription stage.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (transcript_result, cost, time_elapsed).
        """
        start_time = time.time()
        transcript_result, cost = self.transcriber.transcribe_with_retry(audio_path)
        elapsed_time = time.time() - start_time

        logger.info(f"Transcription completed in {elapsed_time:.2f}s, cost: ${cost:.4f}")
        return transcript_result, cost, elapsed_time

    def _extract_stage(
        self, transcript: str, guest_name: Optional[str], guest_title: Optional[str]
    ) -> Tuple[Dict[str, str], Dict[str, Any], float]:
        """
        Execute extraction stage.

        Args:
            transcript: Transcript text.
            guest_name: Guest name for context.
            guest_title: Guest title for context.

        Returns:
            Tuple of (sections_dict, metadata, time_elapsed).
        """
        start_time = time.time()
        sections, metadata = self.extractor.extract_sections(transcript, guest_name, guest_title)
        elapsed_time = time.time() - start_time

        logger.info(
            f"Extraction completed in {elapsed_time:.2f}s, "
            f"cost: ${metadata.get('cost', 0):.4f}, "
            f"tokens: {metadata.get('tokens_total', 0)}"
        )
        return sections, metadata, elapsed_time

    def _format_stage(
        self,
        sections: Dict[str, str],
        guest_name: str,
        guest_title: str,
        episode_number: Optional[int],
        output_dir: str,
    ) -> Tuple[Dict[str, str], float]:
        """
        Execute formatting stage.

        Args:
            sections: Extracted sections.
            guest_name: Guest name.
            guest_title: Guest title.
            episode_number: Episode number.
            output_dir: Output directory.

        Returns:
            Tuple of (output_files_dict, time_elapsed).
        """
        start_time = time.time()

        formatted_content = self.formatter.format_newsletter(
            sections=sections,
            guest_name=guest_name,
            guest_title=guest_title,
            episode_number=episode_number,
            output_dir=output_dir,
        )

        output_files = self.formatter.save_newsletter(formatted_content, guest_name, output_dir)

        elapsed_time = time.time() - start_time
        logger.info(f"Formatting completed in {elapsed_time:.2f}s")

        return output_files, elapsed_time

    def _load_transcript(self, transcript_path: str) -> str:
        """
        Load transcript from file.

        Args:
            transcript_path: Path to transcript file.

        Returns:
            Transcript text.

        Raises:
            FileNotFoundError: If transcript file doesn't exist.
        """
        path = Path(transcript_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract just the transcript text (skip header comments)
        lines = content.split("\n")
        text_lines = [line for line in lines if not line.startswith("#")]
        transcript = "\n".join(text_lines).strip()

        logger.info(f"Loaded transcript from {transcript_path} ({len(transcript)} characters)")
        return transcript

    def estimate_cost(
        self, audio_path: Optional[str] = None, transcript: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate processing cost before running pipeline.

        Args:
            audio_path: Path to audio file (for transcription cost).
            transcript: Transcript text (for extraction cost estimation).

        Returns:
            Dictionary with cost estimates:
            - transcription: Estimated transcription cost
            - extraction: Estimated extraction cost
            - total: Total estimated cost
        """
        costs = {"transcription": 0.0, "extraction": 0.0, "total": 0.0}

        # Estimate transcription cost
        if audio_path:
            costs["transcription"] = self.transcriber.estimate_cost(audio_path)

        # Estimate extraction cost (rough approximation)
        if transcript:
            # Rough token estimate: ~4 characters per token
            estimated_tokens = len(transcript) // 4
            # Assume input tokens = transcript + prompts (~2000), output tokens = ~1500
            estimated_input = estimated_tokens + 2000
            estimated_output = 1500

            from .utils import calculate_llm_cost

            provider = self.env_vars.get("llm_provider", "claude")
            model = (
                self.config.get("extraction", {}).get(provider, {}).get("model")
                if provider in ["claude", "openai"]
                else None
            )
            costs["extraction"] = calculate_llm_cost(
                estimated_input, estimated_output, provider, model
            )

        costs["total"] = costs["transcription"] + costs["extraction"]
        logger.info(f"Estimated cost: ${costs['total']:.4f}")

        return costs

    def get_config(self) -> Dict[str, Any]:
        """
        Get current pipeline configuration.

        Returns:
            Configuration dictionary.
        """
        return self.config

    def get_sections_config(self) -> list:
        """
        Get sections configuration.

        Returns:
            List of section definitions.
        """
        return self.config.get("sections", [])

    def batch_process(
        self,
        input_dir: str,
        guest_info_file: str,
        output_dir: Optional[str] = None,
        save_intermediates: bool = True,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Process multiple interviews in batch mode.

        Args:
            input_dir: Directory containing audio files.
            guest_info_file: Path to JSON file with guest information.
            output_dir: Optional output directory (overrides default).
            save_intermediates: If True, save intermediate files.
            continue_on_error: If True, continue processing on individual failures.

        Returns:
            Dictionary with batch processing results:
            - total_processed: Number of interviews processed
            - successful: Number of successful interviews
            - failed: Number of failed interviews
            - success_rate: Success rate (0-1)
            - results: List of individual results
            - aggregate_stats: Aggregated statistics
            - errors: List of errors encountered

        Raises:
            FileNotFoundError: If input directory or guest info file not found.
            ValueError: If guest info file is invalid JSON.
        """
        logger.info("=" * 80)
        logger.info("Starting batch processing")
        logger.info("=" * 80)

        batch_start = time.time()

        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Load guest information
        guest_info = self._load_guest_info(guest_info_file)

        # Find all audio files in directory
        audio_extensions = {".m4a", ".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".webm"}
        audio_files = [
            f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "results": [],
                "aggregate_stats": {},
                "errors": [],
            }

        logger.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")

        # Process each interview
        results = []
        errors = []
        successful_count = 0
        failed_count = 0

        # Aggregate statistics
        total_cost = 0.0
        total_tokens = 0
        total_duration = 0.0

        for i, audio_file in enumerate(audio_files, 1):
            filename = audio_file.name
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Processing {i}/{len(audio_files)}: {filename}")
            logger.info("=" * 80)

            # Get guest information for this file
            file_info = guest_info.get(filename)

            if not file_info:
                logger.warning(f"No guest information found for {filename} in guest_info.json")
                if continue_on_error:
                    errors.append({
                        "file": filename,
                        "error": "No guest information in guest_info.json",
                    })
                    failed_count += 1
                    continue
                else:
                    raise ValueError(f"Missing guest information for {filename}")

            guest_name = file_info.get("guest_name", "Unknown")
            guest_title = file_info.get("guest_title", "")
            episode_number = file_info.get("episode_number")

            try:
                # Process the interview
                result = self.process_interview(
                    audio_path=str(audio_file),
                    guest_name=guest_name,
                    guest_title=guest_title,
                    episode_number=episode_number,
                    output_dir=output_dir,
                    save_intermediates=save_intermediates,
                )

                results.append(result)

                if result["success"]:
                    successful_count += 1
                    # Aggregate statistics
                    stats = result["stats"]
                    total_cost += stats.get("costs", {}).get("total", 0)
                    total_tokens += stats.get("tokens", {}).get("total", 0)
                    total_duration += stats.get("content", {}).get("audio_duration_minutes", 0)

                    logger.info(f"✓ Successfully processed: {filename}")
                else:
                    failed_count += 1
                    error_msg = result.get("error", "Unknown error")
                    errors.append({"file": filename, "error": error_msg})
                    logger.error(f"✗ Failed to process: {filename} - {error_msg}")

            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                errors.append({"file": filename, "error": error_msg})
                logger.error(f"✗ Exception processing {filename}: {error_msg}")

                if not continue_on_error:
                    raise

        # Calculate batch statistics
        batch_elapsed = time.time() - batch_start
        total_processed = successful_count + failed_count
        success_rate = successful_count / total_processed if total_processed > 0 else 0.0

        # Create aggregate statistics
        aggregate_stats = {
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_audio_duration_minutes": round(total_duration, 2),
            "total_processing_time_seconds": round(batch_elapsed, 2),
            "average_cost_per_interview": round(total_cost / successful_count, 4)
            if successful_count > 0
            else 0.0,
            "average_tokens_per_interview": total_tokens // successful_count
            if successful_count > 0
            else 0,
        }

        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total processed:        {total_processed}")
        logger.info(f"Successful:             {successful_count}")
        logger.info(f"Failed:                 {failed_count}")
        logger.info(f"Success rate:           {success_rate * 100:.1f}%")
        logger.info(f"Total time:             {batch_elapsed:.2f}s")
        logger.info(f"Total cost:             ${total_cost:.4f}")
        logger.info(f"Total tokens:           {total_tokens}")
        logger.info(f"Total audio duration:   {total_duration:.2f} minutes")
        if successful_count > 0:
            logger.info(f"Avg cost per interview: ${total_cost / successful_count:.4f}")
            logger.info(f"Avg time per interview: {batch_elapsed / successful_count:.2f}s")
        logger.info("=" * 80)

        if errors:
            logger.warning(f"Encountered {len(errors)} error(s):")
            for error in errors:
                logger.warning(f"  - {error['file']}: {error['error']}")

        return {
            "total_processed": total_processed,
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": success_rate,
            "results": results,
            "aggregate_stats": aggregate_stats,
            "errors": errors,
            "batch_elapsed_seconds": batch_elapsed,
        }

    def _load_guest_info(self, guest_info_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load guest information from JSON file.

        Args:
            guest_info_file: Path to guest info JSON file.

        Returns:
            Dictionary mapping filenames to guest information.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file is not valid JSON.
        """
        guest_info_path = Path(guest_info_file)

        if not guest_info_path.exists():
            raise FileNotFoundError(f"Guest info file not found: {guest_info_file}")

        try:
            with open(guest_info_path, "r", encoding="utf-8") as f:
                guest_info = json.load(f)

            logger.info(f"Loaded guest information for {len(guest_info)} file(s)")
            return guest_info

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in guest info file: {e}")

    def generate_batch_report(
        self, batch_results: Dict[str, Any], report_path: Optional[str] = None
    ) -> str:
        """
        Generate a detailed batch processing report.

        Args:
            batch_results: Results from batch_process().
            report_path: Optional path to save report file.

        Returns:
            Report text as string.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HOOYOUKNOW BATCH PROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary section
        report_lines.append("## Summary")
        report_lines.append(f"Total Interviews Processed: {batch_results['total_processed']}")
        report_lines.append(f"Successful: {batch_results['successful']}")
        report_lines.append(f"Failed: {batch_results['failed']}")
        report_lines.append(f"Success Rate: {batch_results['success_rate'] * 100:.1f}%")
        report_lines.append(
            f"Total Processing Time: {batch_results['batch_elapsed_seconds']:.2f}s"
        )
        report_lines.append("")

        # Cost statistics
        agg_stats = batch_results.get("aggregate_stats", {})
        report_lines.append("## Cost & Usage Statistics")
        report_lines.append(f"Total Cost: ${agg_stats.get('total_cost', 0):.4f}")
        report_lines.append(f"Average Cost per Interview: ${agg_stats.get('average_cost_per_interview', 0):.4f}")
        report_lines.append(f"Total Tokens: {agg_stats.get('total_tokens', 0):,}")
        report_lines.append(f"Average Tokens per Interview: {agg_stats.get('average_tokens_per_interview', 0):,}")
        report_lines.append(
            f"Total Audio Duration: {agg_stats.get('total_audio_duration_minutes', 0):.2f} minutes"
        )
        report_lines.append("")

        # Individual results
        report_lines.append("## Individual Results")
        report_lines.append("")

        for i, result in enumerate(batch_results.get("results", []), 1):
            guest_name = result.get("guest_name", "Unknown")
            success = result.get("success", False)
            status = "✓ SUCCESS" if success else "✗ FAILED"

            report_lines.append(f"### {i}. {guest_name} - {status}")

            if success:
                stats = result.get("stats", {})
                costs = stats.get("costs", {})
                timing = stats.get("timing", {})
                tokens = stats.get("tokens", {})

                report_lines.append(f"  - Cost: ${costs.get('total', 0):.4f}")
                report_lines.append(f"  - Time: {timing.get('total', 0):.2f}s")
                report_lines.append(f"  - Tokens: {tokens.get('total', 0):,}")

                output_files = result.get("output_files", {})
                if output_files:
                    report_lines.append("  - Output files:")
                    for format_name, file_path in output_files.items():
                        report_lines.append(f"    - {format_name}: {file_path}")
            else:
                error = result.get("error", "Unknown error")
                report_lines.append(f"  - Error: {error}")

            report_lines.append("")

        # Errors section
        if batch_results.get("errors"):
            report_lines.append("## Errors")
            report_lines.append("")
            for error in batch_results["errors"]:
                report_lines.append(f"- {error['file']}: {error['error']}")
            report_lines.append("")

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save to file if path provided
        if report_path:
            report_file = Path(report_path)
            ensure_directory(report_file.parent)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Batch report saved to: {report_path}")

        return report_text
