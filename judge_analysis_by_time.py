#!/usr/bin/env python3
"""
Judge Analysis by Time Script
Analyzes judge results and computes performance metrics over time (by year).
Generates publication-quality figures showing temporal trends.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a publication-quality backend
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')

# Set colorblind-friendly palette (same as judge_analysis.py)
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494']
MODEL_COLORS = {'anthropic/claude-sonnet-4.5': COLORS[0],
                'google/gemini-2.5-pro': COLORS[1],
                'openai/gpt-5': COLORS[2]}


class JudgeAnalyzerByTime:
    """Analyzes judge results from cache_judge directory over time."""

    def __init__(self, cache_judge_dir: str = "cache_judge"):
        """
        Initialize analyzer.

        Args:
            cache_judge_dir: Path to cache_judge directory
        """
        self.cache_judge_dir = Path(cache_judge_dir)
        self.results = []

    def load_all_results(self) -> int:
        """
        Load all judge results from cache_judge directory.

        Returns:
            Number of results loaded
        """
        print(f"Loading results from {self.cache_judge_dir}...")

        for judge_file in self.cache_judge_dir.rglob("*.json"):
            try:
                with open(judge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {judge_file}: {e}")

        print(f"Loaded {len(self.results)} results")
        return len(self.results)

    def parse_year_from_problem_id(self, problem_id: str) -> int:
        """
        Parse year from problem_id.

        Examples:
            keppni_8485_p001 -> 1984
            keppni_9900_p012 -> 1999
            keppni_0001_p005 -> 2000
            keppni_2425_p010 -> 2024

        Args:
            problem_id: Problem ID string

        Returns:
            Year as integer, or None if parsing fails
        """
        try:
            # Extract the year code (e.g., "8485" from "keppni_8485_p001")
            parts = problem_id.split('_')
            if len(parts) < 2 or not parts[0] == 'keppni':
                return None

            year_code = parts[1]
            if len(year_code) != 4:
                return None

            # Parse first two digits
            first_two = int(year_code[:2])

            # Determine century
            if first_two >= 84:  # 1984-1999
                year = 1900 + first_two
            else:  # 2000-2024
                year = 2000 + first_two

            return year
        except Exception as e:
            return None

    def compute_by_year_and_mode(self) -> Dict:
        """Compute performance metrics by year and evaluation mode (all models combined)."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            problem_id = result.get('problem_id', '')
            year = self.parse_year_from_problem_id(problem_id)
            mode = result.get('evaluation_mode', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and year is not None:
                stats[year][mode]['total'] += 1
                if is_correct:
                    stats[year][mode]['correct'] += 1

        # Calculate accuracy
        for year in stats:
            for mode in stats[year]:
                total = stats[year][mode]['total']
                correct = stats[year][mode]['correct']
                if total > 0:
                    stats[year][mode]['accuracy'] = correct / total
                else:
                    stats[year][mode]['accuracy'] = 0.0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_year_model_and_mode(self) -> Dict:
        """Compute performance metrics by year, model, and evaluation mode."""
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0})))

        for result in self.results:
            problem_id = result.get('problem_id', '')
            year = self.parse_year_from_problem_id(problem_id)
            model = result.get('model_name', 'unknown')
            mode = result.get('evaluation_mode', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and year is not None:
                stats[year][model][mode]['total'] += 1
                if is_correct:
                    stats[year][model][mode]['correct'] += 1

        # Calculate accuracy
        for year in stats:
            for model in stats[year]:
                for mode in stats[year][model]:
                    total = stats[year][model][mode]['total']
                    correct = stats[year][model][mode]['correct']
                    if total > 0:
                        stats[year][model][mode]['accuracy'] = correct / total
                    else:
                        stats[year][model][mode]['accuracy'] = 0.0

        return {k: {m: dict(modes) for m, modes in v.items()} for k, v in stats.items()}

    def generate_figures(self, output_dir: str = "figures"):
        """Generate publication-quality figures showing performance over time."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating time-series figures in {output_dir}/...")

        # Set larger font sizes (matching judge_analysis.py)
        plt.rcParams.update({
            'font.size': 13,
            'axes.labelsize': 14,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13
        })

        # Get short model names for display
        model_short_names = {
            'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
            'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
            'openai/gpt-5': 'GPT-5'
        }

        # Figure 1: All models combined - performance over time
        by_year_mode = self.compute_by_year_and_mode()

        fig, ax = plt.subplots(figsize=(10, 5))

        years = sorted(by_year_mode.keys())
        with_choices_acc = []
        without_choices_acc = []

        for year in years:
            with_acc = by_year_mode[year].get('with_choices', {}).get('accuracy', 0) * 100
            without_acc = by_year_mode[year].get('without_choices', {}).get('accuracy', 0) * 100
            with_choices_acc.append(with_acc)
            without_choices_acc.append(without_acc)

        ax.plot(years, with_choices_acc, marker='o', linewidth=2.5, markersize=7,
                label='With Choices', color=COLORS[0])
        ax.plot(years, without_choices_acc, marker='o', linewidth=2.5, markersize=7,
                label='Without Choices', color=COLORS[1])

        ax.set_xlabel('Year')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        ax.set_ylim(50, 105)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_performance_by_year_combined.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Per-model performance over time
        by_year_model_mode = self.compute_by_year_model_and_mode()

        fig, ax = plt.subplots(figsize=(10, 5))

        # Get all models
        all_models = set()
        for year_data in by_year_model_mode.values():
            all_models.update(year_data.keys())
        models = sorted(all_models)

        # Plot each model with two lines (with/without choices)
        for model in models:
            years_model = []
            with_choices_acc = []
            without_choices_acc = []

            for year in sorted(by_year_model_mode.keys()):
                if model in by_year_model_mode[year]:
                    with_acc = by_year_model_mode[year][model].get('with_choices', {}).get('accuracy', None)
                    without_acc = by_year_model_mode[year][model].get('without_choices', {}).get('accuracy', None)

                    if with_acc is not None and without_acc is not None:
                        years_model.append(year)
                        with_choices_acc.append(with_acc * 100)
                        without_choices_acc.append(without_acc * 100)

            model_color = MODEL_COLORS.get(model, COLORS[3])
            model_label = model_short_names.get(model, model)

            # Solid line for with_choices, dashed for without_choices
            ax.plot(years_model, with_choices_acc, marker='o', linewidth=2.5, markersize=6,
                    label=f'{model_label} (With)', color=model_color, linestyle='-')
            ax.plot(years_model, without_choices_acc, marker='s', linewidth=2.5, markersize=6,
                    label=f'{model_label} (Without)', color=model_color, linestyle='--')

        ax.set_xlabel('Year')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
        ax.set_ylim(50, 105)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_performance_by_year_per_model.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Generated 2 time-series figures in {output_dir}/")


def main():
    """Main execution function."""
    # Check if cache_judge directory exists
    if not Path("cache_judge").exists():
        print("Error: cache_judge directory not found")
        print("Please run llm_judge.py first to generate judge results")
        sys.exit(1)

    # Create analyzer
    analyzer = JudgeAnalyzerByTime()

    # Load results
    num_results = analyzer.load_all_results()

    if num_results == 0:
        print("No results found in cache_judge directory")
        sys.exit(1)

    # Generate figures
    analyzer.generate_figures()

    print("\nTime-series analysis complete!")


if __name__ == "__main__":
    main()
