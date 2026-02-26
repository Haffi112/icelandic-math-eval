"""
Prompt Builder for Math Problems
Creates Icelandic prompts for different evaluation modes.
"""

from typing import List, Dict, Optional
from .data_loader import Problem
from .image_handler import ImageHandler


class PromptBuilder:
    """Builds prompts for LLM evaluation in Icelandic."""

    def __init__(self, config: Dict):
        """
        Initialize prompt builder with configuration.

        Args:
            config: Configuration dictionary with prompt templates
        """
        self.config = config
        self.prompts = config.get("prompts", {})

    def build_messages(
        self,
        problem: Problem,
        evaluation_mode: str,
        image_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Build message list for API call.

        Args:
            problem: Problem object
            evaluation_mode: "with_choices" or "without_choices"
            image_path: Path to image file if problem has image

        Returns:
            List of message dictionaries for API call
        """
        messages = []

        # System message
        if evaluation_mode == "with_choices":
            system_prompt = self.prompts.get("with_choices_system", "")
        else:
            system_prompt = self.prompts.get("without_choices_system", "")

        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # User message
        user_content = self._build_user_content(problem, evaluation_mode, image_path)
        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _build_user_content(
        self,
        problem: Problem,
        evaluation_mode: str,
        image_path: Optional[str] = None
    ) -> List[Dict] | str:
        """
        Build user message content (text or multimodal).

        Args:
            problem: Problem object
            evaluation_mode: "with_choices" or "without_choices"
            image_path: Path to image file if problem has image

        Returns:
            String for text-only, or list of content items for multimodal
        """
        # Build text prompt
        if evaluation_mode == "with_choices":
            if problem.is_multiple_choice:
                text_prompt = self._format_with_choices(problem)
            else:
                # Numeric problem in with_choices mode - shouldn't happen
                # but handle gracefully
                text_prompt = self._format_without_choices(problem)
        else:
            text_prompt = self._format_without_choices(problem)

        # If no image, return simple text
        if not image_path:
            return text_prompt

        # If image exists, create multimodal content
        image_content = ImageHandler.create_image_content(image_path)
        if not image_content:
            # Image failed to load, return text only
            return text_prompt

        # Return multimodal content
        return [
            {
                "type": "text",
                "text": text_prompt
            },
            image_content
        ]

    def _format_with_choices(self, problem: Problem) -> str:
        """Format problem with multiple choices."""
        template = self.prompts.get("with_choices_user", "")

        return template.format(
            problem_text=problem.problem_text,
            choice_a=problem.choice_a or "",
            choice_b=problem.choice_b or "",
            choice_c=problem.choice_c or "",
            choice_d=problem.choice_d or ""
        )

    def _format_without_choices(self, problem: Problem) -> str:
        """Format problem without choices."""
        if problem.answer_type == "numeric":
            template = self.prompts.get("numeric_without_choices_user", "")
        else:
            template = self.prompts.get("without_choices_user", "")

        return template.format(problem_text=problem.problem_text)

    def get_prompt_summary(
        self,
        problem: Problem,
        evaluation_mode: str,
        image_path: Optional[str] = None
    ) -> str:
        """
        Get a text summary of the prompt for caching purposes.

        Args:
            problem: Problem object
            evaluation_mode: "with_choices" or "without_choices"
            image_path: Path to image file if problem has image

        Returns:
            String summary of the prompt
        """
        messages = self.build_messages(problem, evaluation_mode, image_path)

        summary_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                summary_parts.append(f"{role.upper()}: {content}")
            elif isinstance(content, list):
                # Multimodal content
                text_parts = [
                    item["text"] for item in content
                    if item["type"] == "text"
                ]
                has_image = any(
                    item["type"] == "image_url" for item in content
                )
                summary_parts.append(
                    f"{role.upper()}: {' '.join(text_parts)}"
                )
                if has_image:
                    summary_parts.append(f"[IMAGE: {image_path}]")

        return "\n\n".join(summary_parts)
