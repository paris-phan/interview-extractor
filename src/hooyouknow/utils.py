"""
Utility functions for the HooYouKnow interview extraction pipeline.

This module provides:
- Configuration loading from YAML and environment variables
- Logging setup
- Cost calculation for API usage
- File management utilities
- Processing statistics tracking
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv


class ProcessingStats:
    """Track statistics for interview processing."""

    def __init__(self):
        """Initialize empty processing statistics."""
        self.transcription_time: float = 0.0
        self.extraction_time: float = 0.0
        self.formatting_time: float = 0.0
        self.total_time: float = 0.0

        self.transcription_cost: float = 0.0
        self.extraction_cost: float = 0.0
        self.total_cost: float = 0.0

        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.tokens_total: int = 0

        self.audio_duration: float = 0.0  # minutes
        self.sections_extracted: int = 0
        self.success: bool = False
        self.errors: List[str] = []

    def add_transcription_stats(self, duration: float, cost: float, time_elapsed: float):
        """Add transcription statistics."""
        self.audio_duration = duration
        self.transcription_cost = cost
        self.transcription_time = time_elapsed
        self.total_cost += cost
        self.total_time += time_elapsed

    def add_extraction_stats(
        self, tokens_in: int, tokens_out: int, cost: float, time_elapsed: float
    ):
        """Add extraction statistics."""
        self.tokens_input = tokens_in
        self.tokens_output = tokens_out
        self.tokens_total = tokens_in + tokens_out
        self.extraction_cost = cost
        self.extraction_time = time_elapsed
        self.total_cost += cost
        self.total_time += time_elapsed

    def add_formatting_stats(self, sections_count: int, time_elapsed: float):
        """Add formatting statistics."""
        self.sections_extracted = sections_count
        self.formatting_time = time_elapsed
        self.total_time += time_elapsed

    def add_error(self, error: str):
        """Add error message."""
        self.errors.append(error)
        self.success = False

    def mark_success(self):
        """Mark processing as successful."""
        self.success = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "success": self.success,
            "timing": {
                "transcription": round(self.transcription_time, 2),
                "extraction": round(self.extraction_time, 2),
                "formatting": round(self.formatting_time, 2),
                "total": round(self.total_time, 2),
            },
            "costs": {
                "transcription": round(self.transcription_cost, 4),
                "extraction": round(self.extraction_cost, 4),
                "total": round(self.total_cost, 4),
            },
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_total,
            },
            "content": {
                "audio_duration_minutes": round(self.audio_duration, 2),
                "sections_extracted": self.sections_extracted,
            },
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """Return human-readable statistics summary."""
        return (
            f"Processing {'succeeded' if self.success else 'failed'}\n"
            f"Time: {self.total_time:.2f}s "
            f"(transcription: {self.transcription_time:.2f}s, "
            f"extraction: {self.extraction_time:.2f}s, "
            f"formatting: {self.formatting_time:.2f}s)\n"
            f"Cost: ${self.total_cost:.4f} "
            f"(transcription: ${self.transcription_cost:.4f}, "
            f"extraction: ${self.extraction_cost:.4f})\n"
            f"Tokens: {self.tokens_total} "
            f"(input: {self.tokens_input}, output: {self.tokens_output})\n"
            f"Audio duration: {self.audio_duration:.2f} minutes\n"
            f"Sections extracted: {self.sections_extracted}"
        )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Dictionary containing configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if config_path is None:
        # Default to config/pipeline_config.yaml relative to project root
        config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    # Validate required sections
    required_keys = ["sections", "extraction", "transcription", "output"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")

    return config


def load_env_vars(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, uses default location.

    Returns:
        Dictionary containing environment variables.
    """
    if env_path is None:
        # Default to config/.env relative to project root
        env_path = Path(__file__).parent.parent.parent / "config" / ".env"
    else:
        env_path = Path(env_path)

    # Load .env file if it exists
    if env_path.exists():
        load_dotenv(env_path)

    # Extract relevant environment variables
    env_vars = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "llm_provider": os.getenv("LLM_PROVIDER", "claude"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "output_dir": os.getenv("OUTPUT_DIR", "outputs"),
    }

    return env_vars


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        log_format: Optional custom log format.

    Returns:
        Configured logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("hooyouknow")
    logger.setLevel(numeric_level)

    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def calculate_whisper_cost(audio_duration_minutes: float) -> float:
    """
    Calculate cost for Whisper transcription.

    Args:
        audio_duration_minutes: Duration of audio in minutes.

    Returns:
        Estimated cost in USD.
    """
    # Whisper pricing: $0.006 per minute
    cost_per_minute = 0.006
    return audio_duration_minutes * cost_per_minute


def calculate_llm_cost(
    tokens_input: int,
    tokens_output: int,
    provider: str = "claude",
    model: Optional[str] = None,
) -> float:
    """
    Calculate cost for LLM API usage.

    Args:
        tokens_input: Number of input tokens.
        tokens_output: Number of output tokens.
        provider: LLM provider ("claude" or "openai").
        model: Specific model name (optional, uses defaults).

    Returns:
        Estimated cost in USD.
    """
    # Pricing per 1000 tokens (as of 2024)
    pricing = {
        "claude": {
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        },
        "openai": {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
        },
    }

    # Get pricing for provider
    if provider not in pricing:
        provider = "claude"  # Default fallback

    provider_pricing = pricing[provider]

    # If model specified, use its pricing, otherwise use first available
    if model and model in provider_pricing:
        model_pricing = provider_pricing[model]
    else:
        model_pricing = next(iter(provider_pricing.values()))

    # Calculate cost
    input_cost = (tokens_input / 1000) * model_pricing["input"]
    output_cost = (tokens_output / 1000) * model_pricing["output"]

    return input_cost + output_cost


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that audio file exists and has valid format.

    Args:
        file_path: Path to audio file.

    Returns:
        True if valid, False otherwise.
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        return False

    # Check if it's a file (not directory)
    if not path.is_file():
        return False

    # Check file extension
    valid_extensions = {".m4a", ".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".webm"}
    if path.suffix.lower() not in valid_extensions:
        return False

    # Check file size (must be > 0 and < 25MB for Whisper API)
    file_size = path.stat().st_size
    if file_size == 0 or file_size > 25 * 1024 * 1024:
        return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename.
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    # Ensure filename is not empty
    if not filename:
        filename = "untitled"

    return filename


def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Path to directory.

    Returns:
        Path object for the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as string.

    Returns:
        ISO format timestamp string.
    """
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., "2m 34s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
