#!/usr/bin/env python3
"""
Test script to verify image input works with OpenRouter vision models.
Tests multiple models to see which ones properly handle multimodal inputs.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_handler import ImageHandler
from openai import OpenAI


def test_image_with_models(image_path: str):
    """Test image input with multiple vision models."""

    # Models to test
    models = [
        "openai/gpt-5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.1",
        "google/gemini-2.5-pro",
        "x-ai/grok-4"
    ]

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        return

    # Validate image
    if not ImageHandler.validate_image(image_path):
        print(f"Error: Invalid or missing image: {image_path}")
        return

    print(f"Testing image: {image_path}\n")
    print("=" * 80)

    # Create image content
    image_content = ImageHandler.create_image_content(image_path)
    if not image_content:
        print("Error: Failed to encode image")
        return

    # Create OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Test each model
    for model in models:
        print(f"\nTesting model: {model}")
        print("-" * 80)

        # Build messages with multimodal content
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is shown in this image? Describe it briefly."
                    },
                    image_content
                ]
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                timeout=60,
                extra_headers={
                    "HTTP-Referer": "https://github.com/icelandic-math-eval",
                    "X-Title": "Icelandic Math Evaluation - Image Test"
                }
            )

            # Print response
            print(f"✓ Success!")
            print(f"\nFull response object:")
            print(response)
            print(f"\nModel used: {response.model}")
            print(f"Response content: {response.choices[0].message.content}")
            print(f"Finish reason: {response.choices[0].finish_reason}")
            print(f"Tokens - Prompt: {response.usage.prompt_tokens}, "
                  f"Completion: {response.usage.completion_tokens}, "
                  f"Total: {response.usage.total_tokens}")

        except Exception as e:
            print(f"✗ Error: {e}")

        print("-" * 80)

    print("\n" + "=" * 80)
    print("Testing complete!")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_image_input.py <image_path>")
        print("\nExample:")
        print("  python test_image_input.py data/images/problem_001.png")
        sys.exit(1)

    image_path = sys.argv[1]
    test_image_with_models(image_path)


if __name__ == "__main__":
    main()
