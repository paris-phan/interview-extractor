"""
Pipeline orchestrator for the HooYouKnow interview extraction system.

This module coordinates the entire workflow from audio transcription through
content extraction to newsletter formatting.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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
