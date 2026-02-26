#!/usr/bin/env python3
"""
Main script for evaluating LLMs on Icelandic math problems.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

from src.data_loader import DataLoader
from src.model_client import ModelClient
from src.cache_manager import CacheManager
from src.evaluator import Evaluator


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_statistics(model: str, stats: dict):
    """Print statistics for a model."""
    print(f"\n{'='*60}")
    print(f"Results for: {model}")
    print(f"{'='*60}")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Failed: {stats['failed']}")
    print(f"Accuracy: {stats['accuracy']:.2%}")

    if stats.get('by_problem_type'):
        print(f"\nBy Problem Type:")
        for ptype, pstats in sorted(stats['by_problem_type'].items()):
            print(f"  {ptype}: {pstats['correct']}/{pstats['total']} "
                  f"({pstats['accuracy']:.2%})")

    if stats.get('by_level'):
        print(f"\nBy Level:")
        for level, lstats in sorted(stats['by_level'].items()):
            print(f"  Level {level}: {lstats['correct']}/{lstats['total']} "
                  f"({lstats['accuracy']:.2%})")


def save_results(results: dict, output_dir: str = "results"):
    """Save results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model, model_results in results.items():
        safe_model_name = model.replace("/", "_").replace(":", "_")
        output_file = output_path / f"{safe_model_name}_results.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)

        print(f"Saved results for {model} to {output_file}")


def main():
    """Main execution function."""
    print("Icelandic Math LLM Evaluation")
    print("="*60)

    # Load environment variables
    load_dotenv()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check that at least one API key is provided
    if not openrouter_api_key and not openai_api_key:
        print("Error: No API keys found in environment")
        print("Please create a .env file with at least one API key:")
        print("  OPENROUTER_API_KEY=your_key_here")
        print("  OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Evaluation mode: {config['evaluation_mode']}")
    print(f"  Models: {', '.join(config['models'])}")
    print(f"  Data directory: {config['data_dir']}")
    print(f"  Cache directory: {config['cache_dir']}")

    # Load data
    print(f"\nLoading problems...")
    data_loader = DataLoader(config['data_dir'])
    problems = data_loader.load_all_problems()

    if not problems:
        print("Error: No problems loaded")
        sys.exit(1)

    # Print data statistics
    stats = data_loader.get_statistics(problems)
    print(f"\nDataset Statistics:")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  Multiple choice: {stats['multiple_choice']}")
    print(f"  Numeric: {stats['numeric']}")
    print(f"  With images: {stats['with_images']}")

    # Initialize components
    model_client = ModelClient(openai_api_key, openrouter_api_key, config)
    cache_manager = CacheManager(config['cache_dir'])
    evaluator = Evaluator(config, model_client, cache_manager)

    # Check cache statistics
    cache_stats = cache_manager.get_cache_statistics()
    if cache_stats['total_cached'] > 0:
        print(f"\nCache Statistics:")
        print(f"  Total cached results: {cache_stats['total_cached']}")
        for model, count in cache_stats['by_model'].items():
            print(f"    {model}: {count}")

    # Run evaluation
    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}\n")

    all_results = evaluator.evaluate_all(problems, config['models'])

    # Compute and print statistics for each model
    for model in config['models']:
        model_results = all_results[model]
        stats = evaluator.compute_statistics(model_results)
        print_statistics(model, stats)

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    save_results(all_results)

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
