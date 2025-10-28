"""
Extraction module for the HooYouKnow interview extraction pipeline.

This module handles content extraction from transcripts using LLMs (Claude or GPT-4).
"""

import json
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import anthropic
from anthropic import Anthropic
import openai
from openai import OpenAI

from .utils import calculate_llm_cost


logger = logging.getLogger("hooyouknow.extraction")


class ContentExtractor:
    """
    Handles content extraction from interview transcripts using LLMs.

    This class uses Claude or GPT-4 to extract structured sections from
    interview transcripts based on configured prompts.
    """

    def __init__(
        self,
        openai_api_key: Optional[str],
        anthropic_api_key: Optional[str],
        provider: str,
        config: Dict[str, Any],
    ):
        """
        Initialize the ContentExtractor.

        Args:
            openai_api_key: OpenAI API key (required if provider is "openai").
            anthropic_api_key: Anthropic API key (required if provider is "claude").
            provider: LLM provider to use ("claude" or "openai").
            config: Configuration dictionary containing extraction settings.

        Raises:
            ValueError: If API key is missing for selected provider or config is invalid.
        """
        self.provider = provider.lower()
        self.config = config.get("extraction", {})

        # Extract timeout setting
        self.timeout = self.config.get("timeout", 120)  # Default 2 minutes

        # Initialize appropriate client based on provider
        if self.provider == "claude":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required when using Claude")
            self.anthropic_client = Anthropic(api_key=anthropic_api_key, timeout=self.timeout)
            self.model = self.config.get("claude", {}).get("model", "claude-3-5-sonnet-20241022")
            self.temperature = self.config.get("claude", {}).get("temperature", 0.7)
            self.max_tokens = self.config.get("claude", {}).get("max_tokens", 4000)
        elif self.provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI")
            self.openai_client = OpenAI(api_key=openai_api_key, timeout=self.timeout)
            self.model = self.config.get("openai", {}).get("model", "gpt-4-turbo-preview")
            self.temperature = self.config.get("openai", {}).get("temperature", 0.7)
            self.max_tokens = self.config.get("openai", {}).get("max_tokens", 4000)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'claude' or 'openai'")

        # Extract other settings
        self.system_prompt = self.config.get("system_prompt", "")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)

        # Store sections configuration
        self.sections_config = config.get("sections", [])
        self.quality_config = config.get("quality_check", {})

        logger.info(f"ContentExtractor initialized with provider: {self.provider}, model: {self.model}, timeout: {self.timeout}s")

    def extract_sections(
        self,
        transcript: str,
        guest_name: Optional[str] = None,
        guest_title: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Extract all configured sections from transcript.

        Args:
            transcript: Full interview transcript text.
            guest_name: Optional guest name for context.
            guest_title: Optional guest title for context.

        Returns:
            Tuple of (sections_dict, metadata_dict) where:
            - sections_dict: Dictionary mapping section names to extracted content
            - metadata_dict: Dictionary with tokens, cost, and other metadata

        Raises:
            Exception: If extraction fails after all retries.
        """
        logger.info(f"Starting content extraction using {self.provider}")

        # Build the extraction prompt
        prompt = self._build_extraction_prompt(transcript, guest_name, guest_title)

        # Log prompt size for debugging
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # Rough estimate: 4 chars per token
        logger.info(
            f"Extraction prompt size: {prompt_chars} characters (~{estimated_tokens} tokens). "
            f"Timeout: {self.timeout}s"
        )

        # Call appropriate LLM
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{self.max_retries}")

                if self.provider == "claude":
                    response, metadata = self._call_claude(prompt)
                else:
                    response, metadata = self._call_openai(prompt)

                # Parse JSON response
                sections = self._parse_json_response(response)

                # Validate sections
                validation_result = self.validate_sections(sections)

                if not validation_result["valid"]:
                    logger.warning(f"Section validation failed: {validation_result['errors']}")
                    # If validation fails but we have some content, continue with warnings
                    if validation_result["warnings"]:
                        logger.warning(f"Validation warnings: {validation_result['warnings']}")

                logger.info(f"Successfully extracted {len(sections)} sections")
                return sections, metadata

            except (anthropic.APIError, openai.APIError) as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"API error. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached due to API errors")
                    raise

            except json.JSONDecodeError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"JSON parsing error. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached due to JSON parsing errors")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error during extraction: {e}")
                raise

        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise Exception("Extraction failed after all retry attempts")

    def _build_extraction_prompt(
        self,
        transcript: str,
        guest_name: Optional[str] = None,
        guest_title: Optional[str] = None,
    ) -> str:
        """
        Build the extraction prompt from configuration and transcript.

        Args:
            transcript: Interview transcript.
            guest_name: Optional guest name.
            guest_title: Optional guest title.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = []

        # Add context if available
        if guest_name or guest_title:
            prompt_parts.append("## Interview Context")
            if guest_name:
                prompt_parts.append(f"Guest: {guest_name}")
            if guest_title:
                prompt_parts.append(f"Title: {guest_title}")
            prompt_parts.append("")

        # Add transcript
        prompt_parts.append("## Interview Transcript")
        prompt_parts.append(transcript)
        prompt_parts.append("")

        # Add extraction instructions
        prompt_parts.append("## Extraction Task")
        prompt_parts.append(
            "Extract the following sections from the interview transcript. "
            "For each section, follow the specific guidelines provided."
        )
        prompt_parts.append("")

        # Add section definitions
        for section in self.sections_config:
            section_name = section.get("name")
            section_title = section.get("title", section_name)
            question_pattern = section.get("question_pattern", "")
            extraction_prompt = section.get("extraction_prompt", "")
            optional = section.get("optional", False)

            prompt_parts.append(f"### {section_title} ({section_name})")
            if question_pattern:
                prompt_parts.append(f"**Question Pattern:** {question_pattern}")
            if optional:
                prompt_parts.append("**Note:** This section is optional. Only include if relevant content is found.")
            prompt_parts.append("")
            prompt_parts.append(extraction_prompt)
            prompt_parts.append("")

        # Add output format instructions
        prompt_parts.append("## Output Format")
        prompt_parts.append("Return your response as a JSON object with the following structure:")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        for i, section in enumerate(self.sections_config):
            section_name = section.get("name")
            comma = "," if i < len(self.sections_config) - 1 else ""
            prompt_parts.append(f'  "{section_name}": "extracted content here"{comma}')
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append(
            "Important: Return ONLY the JSON object, no additional text or formatting. "
            "If a section cannot be extracted, use an empty string."
        )

        return "\n".join(prompt_parts)

    def _call_claude(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call Anthropic Claude API.

        Args:
            prompt: The prompt to send.

        Returns:
            Tuple of (response_text, metadata_dict).

        Raises:
            anthropic.APIError: If API call fails.
        """
        logger.debug(f"Calling Claude API with model: {self.model}")

        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response text
            response_text = response.content[0].text

            # Extract token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost = calculate_llm_cost(input_tokens, output_tokens, "claude", self.model)

            metadata = {
                "provider": "claude",
                "model": self.model,
                "tokens_input": input_tokens,
                "tokens_output": output_tokens,
                "tokens_total": input_tokens + output_tokens,
                "cost": cost,
            }

            logger.info(
                f"Claude API call successful. Tokens: {input_tokens + output_tokens}, Cost: ${cost:.4f}"
            )

            return response_text, metadata

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _call_openai(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call OpenAI API.

        Args:
            prompt: The prompt to send.

        Returns:
            Tuple of (response_text, metadata_dict).

        Raises:
            openai.APIError: If API call fails.
        """
        logger.debug(f"Calling OpenAI API with model: {self.model}")

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate cost
            cost = calculate_llm_cost(input_tokens, output_tokens, "openai", self.model)

            metadata = {
                "provider": "openai",
                "model": self.model,
                "tokens_input": input_tokens,
                "tokens_output": output_tokens,
                "tokens_total": input_tokens + output_tokens,
                "cost": cost,
            }

            logger.info(
                f"OpenAI API call successful. Tokens: {input_tokens + output_tokens}, Cost: ${cost:.4f}"
            )

            return response_text, metadata

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, str]:
        """
        Parse JSON response from LLM, handling malformed JSON.

        Args:
            response: Raw response text from LLM.

        Returns:
            Dictionary of section names to content.

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed.
        """
        # Try to extract JSON from response
        # Sometimes LLMs wrap JSON in markdown code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response

        try:
            sections = json.loads(json_text)
            logger.debug(f"Successfully parsed JSON response with {len(sections)} sections")
            return sections
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response[:500]}...")
            raise

    def validate_sections(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate extracted sections for quality and completeness.

        Args:
            sections: Dictionary of section names to content.

        Returns:
            Dictionary with validation results:
            - valid: Boolean indicating if sections pass validation
            - errors: List of error messages
            - warnings: List of warning messages
        """
        errors = []
        warnings = []

        # Check required sections
        required_sections = self.quality_config.get("required_sections", [])
        for required in required_sections:
            if required not in sections or not sections[required]:
                errors.append(f"Required section missing or empty: {required}")

        # Check for placeholder/rejection text
        reject_patterns = self.quality_config.get("reject_if_contains", [])
        for section_name, content in sections.items():
            if not content:
                continue

            for pattern in reject_patterns:
                if pattern.lower() in content.lower():
                    errors.append(f"Section {section_name} contains placeholder text: {pattern}")

        # Check word counts
        word_count_config = self.quality_config.get("word_count", {})
        min_words = word_count_config.get("min", 50)
        max_words = word_count_config.get("max", 500)
        warn_below = word_count_config.get("warn_below", 100)
        warn_above = word_count_config.get("warn_above", 350)

        for section_name, content in sections.items():
            if not content:
                continue

            word_count = len(content.split())

            if word_count < min_words:
                errors.append(f"Section {section_name} too short: {word_count} words (min: {min_words})")
            elif word_count < warn_below:
                warnings.append(f"Section {section_name} is short: {word_count} words (recommended: {warn_below}+)")

            if word_count > max_words:
                errors.append(f"Section {section_name} too long: {word_count} words (max: {max_words})")
            elif word_count > warn_above:
                warnings.append(f"Section {section_name} is long: {word_count} words (recommended: <{warn_above})")

        # Check content quality (not empty, has substance)
        for section_name, content in sections.items():
            if not content:
                continue

            # Check if content is suspiciously repetitive
            words = content.split()
            if len(words) > 10:
                unique_words = len(set(words))
                if unique_words / len(words) < 0.3:  # Less than 30% unique words
                    warnings.append(f"Section {section_name} may be repetitive or low quality")

        # Overall validation result
        valid = len(errors) == 0

        result = {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
        }

        if errors:
            logger.warning(f"Validation failed with {len(errors)} errors")
        if warnings:
            logger.info(f"Validation warnings: {len(warnings)}")

        return result

    def get_section_word_counts(self, sections: Dict[str, str]) -> Dict[str, int]:
        """
        Get word counts for all sections.

        Args:
            sections: Dictionary of section names to content.

        Returns:
            Dictionary mapping section names to word counts.
        """
        return {name: len(content.split()) for name, content in sections.items() if content}
