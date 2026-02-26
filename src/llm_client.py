"""
LLM Client for OpenRouter API
Handles communication with language models via OpenRouter.
"""

import time
import asyncio
import json
from typing import List, Dict, Optional
from openai import OpenAI
import aiohttp
import os


class LLMClient:
    """Client for interacting with LLMs through OpenRouter."""

    def __init__(self, api_key: str, config: Dict):
        """
        Initialize LLM client.

        Args:
            api_key: OpenRouter API key
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout_seconds", 60)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.5)

    def query_model(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Query a model through OpenRouter.

        Args:
            model: Model identifier (e.g., "openai/gpt-4")
            messages: List of message dictionaries
            temperature: Sampling temperature (1.0 is default, lower for more deterministic)
            max_tokens: Maximum tokens in response (defaults to config value)

        Returns:
            Dictionary with response data, or None if failed
        """
        if max_tokens is None:
            max_tokens = self.config.get("max_tokens", 500)

        for attempt in range(self.max_retries):
            try:
                # Add rate limiting delay
                if attempt > 0:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/icelandic-math-eval",
                        "X-Title": "Icelandic Math Evaluation"
                    }
                )

                # Extract response
                result = {
                    "response": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }

                # Add small delay between successful requests
                time.sleep(self.rate_limit_delay)

                return result

            except Exception as e:
                print(f"Error querying {model} (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt == self.max_retries - 1:
                    return None

                # Wait before retry
                time.sleep(self.rate_limit_delay * (2 ** attempt))

        return None

    async def query_model_async(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[Dict]:
        """
        Query a model through OpenRouter asynchronously.

        Args:
            model: Model identifier (e.g., "openai/gpt-4")
            messages: List of message dictionaries
            temperature: Sampling temperature (1.0 is default, lower for more deterministic)
            max_tokens: Maximum tokens in response (defaults to config value)
            session: Optional aiohttp session (for connection pooling)

        Returns:
            Dictionary with response data, or None if failed
        """
        if max_tokens is None:
            max_tokens = self.config.get("max_tokens", 500)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/icelandic-math-eval",
            "X-Title": "Icelandic Math Evaluation",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Use provided session or create temporary one
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            for attempt in range(self.max_retries):
                try:
                    # Add exponential backoff delay for retries
                    if attempt > 0:
                        await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))

                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Extract response in OpenAI format
                            result = {
                                "response": data["choices"][0]["message"]["content"],
                                "model": data.get("model", model),
                                "usage": {
                                    "prompt_tokens": data["usage"]["prompt_tokens"],
                                    "completion_tokens": data["usage"]["completion_tokens"],
                                    "total_tokens": data["usage"]["total_tokens"]
                                },
                                "finish_reason": data["choices"][0].get("finish_reason")
                            }

                            # Add small delay between successful requests
                            await asyncio.sleep(self.rate_limit_delay)

                            return result
                        else:
                            error_text = await response.text()
                            print(f"Error querying {model} (attempt {attempt + 1}/{self.max_retries}): "
                                  f"HTTP {response.status} - {error_text}")

                            if attempt == self.max_retries - 1:
                                return None

                except asyncio.TimeoutError:
                    print(f"Timeout querying {model} (attempt {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return None

                except Exception as e:
                    print(f"Error querying {model} (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        return None

        finally:
            if close_session:
                await session.close()

        return None

    def extract_answer(
        self,
        response_text: str,
        evaluation_mode: str,
        answer_type: str = "multiple_choice"
    ) -> Optional[str]:
        """
        Extract answer from LLM response.

        Args:
            response_text: Raw response from LLM
            evaluation_mode: "with_choices" or "without_choices"
            answer_type: "multiple_choice" or "numeric"

        Returns:
            Extracted answer string, or None if extraction failed
        """
        if not response_text:
            return None

        response_text = response_text.strip()

        if evaluation_mode == "with_choices" and answer_type == "multiple_choice":
            # Extract letter (A, B, C, or D)
            response_upper = response_text.upper()

            # Look for A, B, C, or D
            for letter in ["A", "B", "C", "D"]:
                if letter in response_upper:
                    # Return first occurrence
                    return letter

            return None

        else:
            # For "without_choices" or numeric, return cleaned response
            # Try to extract a number if possible
            import re

            # Look for numbers in the response
            numbers = re.findall(r'-?\d+\.?\d*', response_text)
            if numbers:
                return numbers[0]  # Return first number found

            # If no number found, return the cleaned text
            return response_text

    def validate_models(self, models: List[str]) -> List[str]:
        """
        Validate that models are accessible.

        Args:
            models: List of model identifiers

        Returns:
            List of valid model identifiers
        """
        # For now, just return all models
        # In production, you might want to check if models are available
        # via OpenRouter API
        return models
