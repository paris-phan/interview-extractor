"""
Formatting module for the HooYouKnow interview extraction pipeline.

This module handles formatting of extracted sections into newsletter formats
(Markdown, HTML, JSON).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .utils import ensure_directory, sanitize_filename


logger = logging.getLogger("hooyouknow.formatting")


class NewsletterFormatter:
    """
    Handles formatting of extracted interview sections into newsletter formats.

    Supports multiple output formats: Markdown, HTML, and JSON.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NewsletterFormatter.

        Args:
            config: Configuration dictionary containing output settings.
        """
        self.config = config
        self.output_config = config.get("output", {})
        self.newsletter_config = config.get("newsletter", {})
        self.sections_config = config.get("sections", [])

        # Get output formats
        self.formats = self.output_config.get("formats", ["markdown", "json"])

        # Get template
        self.markdown_template = self.output_config.get("markdown_template", self._default_template())

        # Get newsletter metadata
        self.newsletter_name = self.newsletter_config.get("name", "HooYouKnow")
        self.tagline = self.newsletter_config.get("tagline", "Real talk from real professionals")
        self.website = self.newsletter_config.get("website", "https://hooyouknow.com")

        logger.info(f"NewsletterFormatter initialized with formats: {self.formats}")

    def format_newsletter(
        self,
        sections: Dict[str, str],
        guest_name: str,
        guest_title: str,
        episode_number: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Format extracted sections into newsletter(s) in specified format(s).

        Args:
            sections: Dictionary of section names to content.
            guest_name: Name of the interview guest.
            guest_title: Title/position of the guest.
            episode_number: Optional episode number.
            output_dir: Optional output directory (overrides config).

        Returns:
            Dictionary with keys:
            - markdown: Formatted markdown content (if format enabled)
            - html: Formatted HTML content (if format enabled)
            - json: Structured JSON content (if format enabled)
        """
        logger.info(f"Formatting newsletter for {guest_name}")

        formatted_content = {}

        # Format metadata
        metadata = {
            "guest_name": guest_name,
            "guest_title": guest_title,
            "episode_number": episode_number,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

        # Generate each requested format
        if "markdown" in self.formats:
            formatted_content["markdown"] = self._format_markdown(sections, metadata)
            logger.debug("Markdown format generated")

        if "html" in self.formats:
            # HTML is generated from markdown
            if "markdown" not in formatted_content:
                markdown_content = self._format_markdown(sections, metadata)
            else:
                markdown_content = formatted_content["markdown"]
            formatted_content["html"] = self._format_html(markdown_content)
            logger.debug("HTML format generated")

        if "json" in self.formats:
            formatted_content["json"] = self._format_json(sections, metadata)
            logger.debug("JSON format generated")

        logger.info(f"Successfully formatted newsletter in {len(formatted_content)} format(s)")
        return formatted_content

    def save_newsletter(
        self,
        formatted_content: Dict[str, Any],
        guest_name: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save formatted newsletter content to files.

        Args:
            formatted_content: Dictionary of format names to content.
            guest_name: Name of the guest (used for filename).
            output_dir: Optional output directory (overrides config).

        Returns:
            Dictionary mapping format names to output file paths.
        """
        if output_dir is None:
            output_dir = self.output_config.get("directory_structure", {}).get("base", "outputs")

        # Ensure output directory exists
        output_path = ensure_directory(output_dir)

        # Create subdirectories
        newsletters_dir = output_path / self.output_config.get("directory_structure", {}).get(
            "newsletters", "newsletters"
        )
        ensure_directory(newsletters_dir)

        # Sanitize guest name for filename
        safe_name = sanitize_filename(guest_name)

        output_files = {}

        # Save each format
        if "markdown" in formatted_content:
            md_file = newsletters_dir / f"{safe_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(formatted_content["markdown"])
            output_files["markdown"] = str(md_file)
            logger.info(f"Saved markdown to: {md_file}")

        if "html" in formatted_content:
            html_file = newsletters_dir / f"{safe_name}.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(formatted_content["html"])
            output_files["html"] = str(html_file)
            logger.info(f"Saved HTML to: {html_file}")

        if "json" in formatted_content:
            json_file = newsletters_dir / f"{safe_name}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(formatted_content["json"], f, indent=2, ensure_ascii=False)
            output_files["json"] = str(json_file)
            logger.info(f"Saved JSON to: {json_file}")

        return output_files

    def _format_markdown(self, sections: Dict[str, str], metadata: Dict[str, str]) -> str:
        """
        Format sections as Markdown newsletter.

        Args:
            sections: Dictionary of section names to content.
            metadata: Dictionary with guest_name, guest_title, etc.

        Returns:
            Formatted markdown string.
        """
        # Start with template
        content = self.markdown_template

        # Replace metadata placeholders
        content = content.replace("{guest_name}", metadata.get("guest_name", "Unknown"))
        content = content.replace("{guest_title}", metadata.get("guest_title", ""))
        content = content.replace("{date}", metadata.get("date", ""))

        episode_num = metadata.get("episode_number")
        if episode_num:
            content = content.replace("{episode_number}", str(episode_num))
        else:
            content = content.replace("**Episode:** {episode_number}\n", "")

        # Build sections content
        sections_text = []

        for section_config in self.sections_config:
            section_name = section_config.get("name")
            section_title = section_config.get("title", section_name)
            optional = section_config.get("optional", False)

            # Get section content
            section_content = sections.get(section_name, "")

            # Skip optional sections if not present
            if optional and not section_content:
                logger.debug(f"Skipping optional section: {section_name}")
                continue

            # Skip empty sections
            if not section_content:
                logger.warning(f"Section {section_name} is empty")
                continue

            # Format section
            sections_text.append(f"## {section_title}")
            sections_text.append("")
            sections_text.append(section_content)
            sections_text.append("")

        # Replace sections placeholder
        content = content.replace("{sections}", "\n".join(sections_text))

        return content

    def _format_html(self, markdown_content: str) -> str:
        """
        Convert Markdown to HTML.

        Args:
            markdown_content: Markdown formatted content.

        Returns:
            HTML formatted content.
        """
        try:
            import markdown

            # Convert markdown to HTML
            html_body = markdown.markdown(
                markdown_content,
                extensions=["extra", "nl2br"],
            )

            # Wrap in basic HTML template
            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.newsletter_name} Newsletter</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""

            return html_template

        except ImportError:
            logger.warning("markdown package not installed. HTML generation disabled.")
            return f"<pre>{markdown_content}</pre>"

    def _format_json(self, sections: Dict[str, str], metadata: Dict[str, str]) -> Dict[str, Any]:
        """
        Format sections as structured JSON.

        Args:
            sections: Dictionary of section names to content.
            metadata: Dictionary with guest_name, guest_title, etc.

        Returns:
            Dictionary with structured newsletter data.
        """
        # Build sections list with metadata
        sections_list = []

        for section_config in self.sections_config:
            section_name = section_config.get("name")
            section_title = section_config.get("title", section_name)
            optional = section_config.get("optional", False)

            # Get section content
            section_content = sections.get(section_name, "")

            # Skip optional sections if not present
            if optional and not section_content:
                continue

            # Skip empty sections
            if not section_content:
                continue

            sections_list.append({
                "name": section_name,
                "title": section_title,
                "content": section_content,
                "word_count": len(section_content.split()),
                "optional": optional,
            })

        # Build complete JSON structure
        json_data = {
            "newsletter": {
                "name": self.newsletter_name,
                "tagline": self.tagline,
                "website": self.website,
            },
            "episode": {
                "guest_name": metadata.get("guest_name"),
                "guest_title": metadata.get("guest_title"),
                "episode_number": metadata.get("episode_number"),
                "date": metadata.get("date"),
            },
            "sections": sections_list,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_sections": len(sections_list),
                "total_words": sum(s["word_count"] for s in sections_list),
            },
        }

        return json_data

    def _default_template(self) -> str:
        """
        Return default Markdown template.

        Returns:
            Default template string.
        """
        return """# HooYouKnow Newsletter: {guest_name}

**Guest:** {guest_name}
**Title:** {guest_title}
**Episode:** {episode_number}
**Date:** {date}

---

{sections}

---

*Want to share your story? Visit [hooyouknow.com](https://hooyouknow.com)*
"""

    def preview_newsletter(
        self,
        sections: Dict[str, str],
        guest_name: str,
        guest_title: str,
        max_chars: int = 500,
    ) -> str:
        """
        Generate a preview of the newsletter (first N characters).

        Args:
            sections: Dictionary of section names to content.
            guest_name: Name of the guest.
            guest_title: Title of the guest.
            max_chars: Maximum characters to include in preview.

        Returns:
            Preview string.
        """
        metadata = {
            "guest_name": guest_name,
            "guest_title": guest_title,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

        full_content = self._format_markdown(sections, metadata)
        preview = full_content[:max_chars]

        if len(full_content) > max_chars:
            preview += "..."

        return preview

    def get_section_count(self, sections: Dict[str, str]) -> Dict[str, int]:
        """
        Get count of sections by type.

        Args:
            sections: Dictionary of section names to content.

        Returns:
            Dictionary with counts:
            - total: Total sections
            - required: Required sections present
            - optional: Optional sections present
            - missing: Missing required sections
        """
        total = 0
        required = 0
        optional = 0
        missing_required = 0

        for section_config in self.sections_config:
            section_name = section_config.get("name")
            is_optional = section_config.get("optional", False)
            has_content = bool(sections.get(section_name))

            if has_content:
                total += 1
                if is_optional:
                    optional += 1
                else:
                    required += 1
            elif not is_optional:
                missing_required += 1

        return {
            "total": total,
            "required": required,
            "optional": optional,
            "missing": missing_required,
        }
