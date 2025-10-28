"""
HooYouKnow Interview Extraction Pipeline

An automated pipeline for converting interview audio files into structured,
formatted newsletter content using OpenAI Whisper and Claude/GPT-4.

Usage:
    from hooyouknow import InterviewPipeline

    pipeline = InterviewPipeline()
    result = pipeline.process_interview(
        audio_path="interview.m4a",
        guest_name="Jane Doe",
        guest_title="CEO at TechCorp"
    )
"""

__version__ = "1.0.0"
__author__ = "HooYouKnow Team"
__email__ = "contact@hooyouknow.com"
__license__ = "MIT"

# Import main classes for easy access
from .pipeline import InterviewPipeline
from .transcription import AudioTranscriber
from .extraction import ContentExtractor
from .formatting import NewsletterFormatter

# Import utility classes and functions
from .utils import (
    ProcessingStats,
    load_config,
    load_env_vars,
    setup_logging,
    calculate_whisper_cost,
    calculate_llm_cost,
)

# Define public API
__all__ = [
    # Main classes
    "InterviewPipeline",
    "AudioTranscriber",
    "ContentExtractor",
    "NewsletterFormatter",
    # Utility classes
    "ProcessingStats",
    # Utility functions
    "load_config",
    "load_env_vars",
    "setup_logging",
    "calculate_whisper_cost",
    "calculate_llm_cost",
    # Package metadata
    "__version__",
]
