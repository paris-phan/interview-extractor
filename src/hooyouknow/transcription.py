"""
Transcription module for the HooYouKnow interview extraction pipeline.

This module handles audio transcription using OpenAI's Whisper API.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import openai
from openai import OpenAI

from .utils import validate_audio_file, calculate_whisper_cost, ensure_directory


logger = logging.getLogger("hooyouknow.transcription")


class AudioTranscriber:
    """
    Handles audio transcription using OpenAI Whisper API.

    This class provides methods for transcribing audio files with retry logic,
    progress tracking, and cost estimation.
    """

    def __init__(self, api_key: str, config: Dict[str, Any]):
        """
        Initialize the AudioTranscriber.

        Args:
            api_key: OpenAI API key.
            config: Configuration dictionary containing transcription settings.

        Raises:
            ValueError: If API key is missing or config is invalid.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for transcription")

        self.api_key = api_key
        self.config = config.get("transcription", {})

        # Extract configuration settings
        self.model = self.config.get("model", "whisper-1")
        self.language = self.config.get("language", "en")
        self.response_format = self.config.get("response_format", "verbose_json")
        self.temperature = self.config.get("temperature", 0.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)
        self.timeout = self.config.get("timeout", 300)  # Default 5 minutes

        # Initialize client with timeout
        self.client = OpenAI(api_key=api_key, timeout=self.timeout)

        logger.info(f"AudioTranscriber initialized with model: {self.model}, timeout: {self.timeout}s")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file (single attempt).

        Args:
            audio_path: Path to the audio file.
            language: Optional language code (overrides config).
            response_format: Optional response format (overrides config).

        Returns:
            Dictionary containing transcription results with keys:
            - text: The transcribed text
            - segments: List of segments with timestamps (if verbose_json format)
            - language: Detected or specified language
            - duration: Audio duration in seconds

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            ValueError: If audio file is invalid.
            openai.OpenAIError: If API call fails.
        """
        # Validate audio file
        if not validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        # Log file information
        from pathlib import Path
        file_path = Path(audio_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Starting transcription for: {audio_path} (size: {file_size_mb:.2f}MB)")

        if file_size_mb > 20:
            logger.warning(
                f"Large audio file ({file_size_mb:.2f}MB). "
                f"Transcription may take several minutes. Timeout set to {self.timeout}s."
            )

        # Use provided parameters or fall back to config
        lang = language or self.language
        fmt = response_format or self.response_format

        # Open and transcribe the audio file
        with open(audio_path, "rb") as audio_file:
            try:
                # Make API call
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=lang if lang else None,
                    response_format=fmt,
                    temperature=self.temperature,
                )

                # Parse response based on format
                if fmt == "verbose_json":
                    result = {
                        "text": response.text,
                        "segments": response.segments if hasattr(response, "segments") else [],
                        "language": response.language if hasattr(response, "language") else lang,
                        "duration": response.duration if hasattr(response, "duration") else 0,
                    }
                elif fmt == "json":
                    result = {
                        "text": response.text,
                        "segments": [],
                        "language": lang,
                        "duration": 0,
                    }
                else:
                    # Plain text format
                    result = {
                        "text": str(response),
                        "segments": [],
                        "language": lang,
                        "duration": 0,
                    }

                logger.info(f"Transcription completed successfully. Length: {len(result['text'])} characters")
                return result

            except openai.APIError as e:
                logger.error(f"OpenAI API error during transcription: {e}")
                raise
            except openai.APIConnectionError as e:
                logger.error(f"Network error during transcription: {e}")
                raise
            except openai.RateLimitError as e:
                logger.error(f"Rate limit exceeded during transcription: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during transcription: {e}")
                raise

    def transcribe_with_retry(
        self,
        audio_path: str,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Transcribe audio file with exponential backoff retry logic.

        Args:
            audio_path: Path to the audio file.
            language: Optional language code (overrides config).
            response_format: Optional response format (overrides config).

        Returns:
            Tuple of (transcription_result, cost_usd)

        Raises:
            Exception: If all retry attempts fail.
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                logger.info(f"Transcription attempt {attempt + 1}/{self.max_retries}")

                # Attempt transcription
                start_time = time.time()
                result = self.transcribe(audio_path, language, response_format)
                elapsed_time = time.time() - start_time

                # Calculate cost
                duration_minutes = result.get("duration", 0) / 60.0
                cost = calculate_whisper_cost(duration_minutes)

                logger.info(
                    f"Transcription successful on attempt {attempt + 1}. "
                    f"Time: {elapsed_time:.2f}s, Cost: ${cost:.4f}"
                )

                return result, cost

            except openai.RateLimitError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"Rate limit hit. Retrying in {wait_time}s... "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached due to rate limiting")
                    raise

            except openai.APIConnectionError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"Network error. Retrying in {wait_time}s... "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached due to network errors")
                    raise

            except openai.APIError as e:
                # For other API errors, don't retry
                logger.error(f"API error (not retrying): {e}")
                raise

            except Exception as e:
                # For unexpected errors, don't retry
                logger.error(f"Unexpected error (not retrying): {e}")
                raise

        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise Exception("Transcription failed after all retry attempts")

    def save_transcript(
        self,
        transcript_result: Dict[str, Any],
        output_path: str,
        include_timestamps: bool = False,
    ) -> str:
        """
        Save transcript to file.

        Args:
            transcript_result: Transcription result from transcribe() or transcribe_with_retry().
            output_path: Path where transcript should be saved.
            include_timestamps: If True and segments available, include timestamps.

        Returns:
            Path to saved transcript file.
        """
        output_path = Path(output_path)

        # Ensure output directory exists
        ensure_directory(output_path.parent)

        # Build transcript content
        content_lines = []

        # Add metadata header
        content_lines.append("# Interview Transcript")
        content_lines.append(f"# Language: {transcript_result.get('language', 'unknown')}")
        if transcript_result.get('duration'):
            duration_min = transcript_result['duration'] / 60.0
            content_lines.append(f"# Duration: {duration_min:.2f} minutes")
        content_lines.append("")

        # Add transcript text
        if include_timestamps and transcript_result.get("segments"):
            content_lines.append("## Transcript with Timestamps")
            content_lines.append("")

            for segment in transcript_result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()

                # Format timestamp as [MM:SS - MM:SS]
                start_str = self._format_timestamp(start_time)
                end_str = self._format_timestamp(end_time)
                content_lines.append(f"[{start_str} - {end_str}] {text}")
        else:
            content_lines.append("## Transcript")
            content_lines.append("")
            content_lines.append(transcript_result.get("text", ""))

        # Write to file
        content = "\n".join(content_lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Transcript saved to: {output_path}")
        return str(output_path)

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into MM:SS string.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted timestamp string.
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def estimate_cost(self, audio_path: str) -> float:
        """
        Estimate transcription cost for an audio file.

        This attempts to determine audio duration and calculate cost.
        Requires pydub for accurate duration detection.

        Args:
            audio_path: Path to audio file.

        Returns:
            Estimated cost in USD.
        """
        try:
            # Try to get duration using pydub
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            duration_minutes = len(audio) / 1000.0 / 60.0  # Convert ms to minutes
            cost = calculate_whisper_cost(duration_minutes)
            logger.debug(f"Estimated cost for {audio_path}: ${cost:.4f} ({duration_minutes:.2f} min)")
            return cost

        except ImportError:
            logger.warning("pydub not installed. Cannot estimate audio duration. Assuming 10 minutes.")
            # Fallback: assume 10 minutes
            return calculate_whisper_cost(10.0)

        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}. Assuming 10 minutes.")
            # Fallback: assume 10 minutes
            return calculate_whisper_cost(10.0)

    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Get audio file duration in minutes.

        Args:
            audio_path: Path to audio file.

        Returns:
            Duration in minutes, or None if cannot be determined.
        """
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            duration_minutes = len(audio) / 1000.0 / 60.0
            return duration_minutes

        except ImportError:
            logger.warning("pydub not installed. Cannot determine audio duration.")
            return None

        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return None
