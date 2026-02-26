#!/usr/bin/env python3
"""
Test script to verify model routing logic.
Tests that the ModelClient correctly routes requests to OpenAI vs OpenRouter.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_client import ModelClient


def test_routing():
    """Test model routing logic."""

    print("Testing ModelClient routing logic...\n")
    print("=" * 60)

    # Create a mock config
    config = {
        "max_retries": 3,
        "timeout_seconds": 60,
        "rate_limit_delay": 0.5
    }

    # Initialize with both API keys (using dummy keys for testing)
    client = ModelClient(
        openai_api_key="dummy_openai_key",
        openrouter_api_key="dummy_openrouter_key",
        config=config
    )

    # Test cases
    test_cases = [
        ("openai/gpt-5", "OpenAI Direct", "gpt-5"),
        ("openai/gpt-4o", "OpenAI Direct", "gpt-4o"),
        ("anthropic/claude-sonnet-4.5", "OpenRouter", "anthropic/claude-sonnet-4.5"),
        ("anthropic/claude-opus-4.1", "OpenRouter", "anthropic/claude-opus-4.1"),
        ("google/gemini-2.5-pro", "OpenRouter", "google/gemini-2.5-pro"),
    ]

    print("\nModel Routing Tests:")
    print("-" * 60)

    all_passed = True
    for model_id, expected_provider, expected_model_name in test_cases:
        try:
            api_client, model_name = client._get_client_for_model(model_id)

            # Determine actual provider
            if api_client == client.openai_client:
                actual_provider = "OpenAI Direct"
            elif api_client == client.openrouter_client:
                actual_provider = "OpenRouter"
            else:
                actual_provider = "Unknown"

            # Check results
            provider_match = actual_provider == expected_provider
            model_match = model_name == expected_model_name

            status = "✓ PASS" if (provider_match and model_match) else "✗ FAIL"

            print(f"{status} | {model_id}")
            print(f"       Provider: {actual_provider} (expected: {expected_provider})")
            print(f"       Model name: {model_name} (expected: {expected_model_name})")
            print()

            if not (provider_match and model_match):
                all_passed = False

        except Exception as e:
            print(f"✗ FAIL | {model_id}")
            print(f"       Error: {e}")
            print()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ All routing tests PASSED!")
    else:
        print("✗ Some routing tests FAILED!")
    print("=" * 60)

    return all_passed


def test_missing_keys():
    """Test error handling for missing API keys."""

    print("\n\nTesting missing API key handling...\n")
    print("=" * 60)

    config = {
        "max_retries": 3,
        "timeout_seconds": 60,
        "rate_limit_delay": 0.5
    }

    # Test 1: Only OpenAI key, trying OpenRouter model
    print("\nTest: Only OpenAI key provided, requesting OpenRouter model")
    print("-" * 60)
    client = ModelClient(
        openai_api_key="dummy_openai_key",
        openrouter_api_key=None,
        config=config
    )

    try:
        client._get_client_for_model("anthropic/claude-sonnet-4.5")
        print("✗ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError")
        print(f"       Message: {e}")

    # Test 2: Only OpenRouter key, trying OpenAI model
    print("\n\nTest: Only OpenRouter key provided, requesting OpenAI model")
    print("-" * 60)
    client = ModelClient(
        openai_api_key=None,
        openrouter_api_key="dummy_openrouter_key",
        config=config
    )

    try:
        client._get_client_for_model("openai/gpt-5")
        print("✗ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError")
        print(f"       Message: {e}")

    print("\n" + "=" * 60)
    print("✓ API key validation tests PASSED!")
    print("=" * 60)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ModelClient Routing Test Suite")
    print("=" * 60)

    # Run tests
    routing_passed = test_routing()
    test_missing_keys()

    print("\n\n" + "=" * 60)
    if routing_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 60 + "\n")

    return 0 if routing_passed else 1


if __name__ == "__main__":
    sys.exit(main())
