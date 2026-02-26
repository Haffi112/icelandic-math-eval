"""
Unified Model Client
Routes requests to appropriate API provider (OpenAI direct or OpenRouter).
"""

import time
import asyncio
from typing import List, Dict, Optional
from openai import OpenAI, AsyncOpenAI
import aiohttp
import os


class OpenAIDirectClient:
    """Client for direct OpenAI API access."""

    def __init__(self, api_key: str, config: Dict):
        """
        Initialize OpenAI direct client.

        Args:
            api_key: OpenAI API key
            config: Configuration dictionary
        """
        self.api_key = api_key
        self.config = config

        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)

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
        Query a model through OpenAI direct API.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-5")
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
                    max_completion_tokens=max_tokens,
                    timeout=self.timeout,
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

                # Add cached tokens info if available (prompt caching)
                if hasattr(response.usage, 'prompt_tokens_details'):
                    result["usage"]["cached_tokens"] = getattr(
                        response.usage.prompt_tokens_details,
                        'cached_tokens',
                        0
                    )

                # Add small delay between successful requests
                time.sleep(self.rate_limit_delay)

                return result

            except Exception as e:
                print(f"Error querying {model} via OpenAI (attempt {attempt + 1}/{self.max_retries}): {e}")

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
        Query a model through OpenAI direct API asynchronously.

        Args:
            model: Model identifier (e.g., "gpt-4o", "gpt-5")
            messages: List of message dictionaries
            temperature: Sampling temperature (1.0 is default, lower for more deterministic)
            max_tokens: Maximum tokens in response (defaults to config value)
            session: Optional aiohttp session (ignored, kept for compatibility)

        Returns:
            Dictionary with response data, or None if failed
        """
        if max_tokens is None:
            max_tokens = self.config.get("max_tokens", 500)

        for attempt in range(self.max_retries):
            try:
                # Add exponential backoff delay for retries
                if attempt > 0:
                    await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))

                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    timeout=self.timeout,
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

                # Add cached tokens info if available (prompt caching)
                if hasattr(response.usage, 'prompt_tokens_details'):
                    result["usage"]["cached_tokens"] = getattr(
                        response.usage.prompt_tokens_details,
                        'cached_tokens',
                        0
                    )

                # Add small delay between successful requests
                await asyncio.sleep(self.rate_limit_delay)

                return result

            except Exception as e:
                print(f"Error querying {model} via OpenAI (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None

        return None


class OpenRouterClient:
    """Client for OpenRouter API access."""

    def __init__(self, api_key: str, config: Dict):
        """
        Initialize OpenRouter client.

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
            model: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
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
                print(f"Error querying {model} via OpenRouter (attempt {attempt + 1}/{self.max_retries}): {e}")

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
            model: Model identifier (e.g., "anthropic/claude-sonnet-4.5")
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
                            print(f"Error querying {model} via OpenRouter (attempt {attempt + 1}/{self.max_retries}): "
                                  f"HTTP {response.status} - {error_text}")

                            if attempt == self.max_retries - 1:
                                return None

                except asyncio.TimeoutError:
                    print(f"Timeout querying {model} via OpenRouter (attempt {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return None

                except Exception as e:
                    print(f"Error querying {model} via OpenRouter (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        return None

        finally:
            if close_session:
                await session.close()

        return None


class ModelClient:
    """
    Unified client that routes requests to appropriate API provider.
    Automatically detects whether to use OpenAI direct API or OpenRouter
    based on the model identifier.
    """

    def __init__(self, openai_api_key: Optional[str], openrouter_api_key: Optional[str], config: Dict):
        """
        Initialize unified model client.

        Args:
            openai_api_key: OpenAI API key (optional)
            openrouter_api_key: OpenRouter API key (optional)
            config: Configuration dictionary
        """
        self.config = config
        self.openai_client = None
        self.openrouter_client = None

        # Initialize OpenAI client if key provided
        if openai_api_key:
            self.openai_client = OpenAIDirectClient(openai_api_key, config)

        # Initialize OpenRouter client if key provided
        if openrouter_api_key:
            self.openrouter_client = OpenRouterClient(openrouter_api_key, config)

    def _get_client_for_model(self, model: str):
        """
        Determine which client to use based on model identifier.

        Args:
            model: Model identifier

        Returns:
            Tuple of (client, model_name) where model_name has prefix stripped if needed
        """
        # Check if model starts with "openai/"
        if model.startswith("openai/"):
            if not self.openai_client:
                raise ValueError(
                    f"Model '{model}' requires OpenAI API key, but none provided. "
                    "Please set OPENAI_API_KEY in your .env file."
                )
            # Strip the "openai/" prefix for direct API
            model_name = model[7:]  # Remove "openai/" prefix
            return self.openai_client, model_name
        else:
            # Use OpenRouter for all other models
            if not self.openrouter_client:
                raise ValueError(
                    f"Model '{model}' requires OpenRouter API key, but none provided. "
                    "Please set OPENROUTER_API_KEY in your .env file."
                )
            return self.openrouter_client, model

    def query_model(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Query a model through the appropriate API.

        Args:
            model: Model identifier (e.g., "openai/gpt-5", "anthropic/claude-sonnet-4.5")
            messages: List of message dictionaries
            temperature: Sampling temperature (1.0 is default, lower for more deterministic)
            max_tokens: Maximum tokens in response (defaults to config value)

        Returns:
            Dictionary with response data, or None if failed
        """
        client, model_name = self._get_client_for_model(model)
        return client.query_model(model_name, messages, temperature, max_tokens)

    async def query_model_async(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[Dict]:
        """
        Query a model through the appropriate API asynchronously.

        Args:
            model: Model identifier (e.g., "openai/gpt-5", "anthropic/claude-sonnet-4.5")
            messages: List of message dictionaries
            temperature: Sampling temperature (1.0 is default, lower for more deterministic)
            max_tokens: Maximum tokens in response (defaults to config value)
            session: Optional aiohttp session (for connection pooling)

        Returns:
            Dictionary with response data, or None if failed
        """
        client, model_name = self._get_client_for_model(model)
        return await client.query_model_async(model_name, messages, temperature, max_tokens, session)

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
        return models
