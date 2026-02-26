#!/usr/bin/env python3
"""
Judge Analysis Script
Analyzes judge results and computes performance metrics across multiple dimensions.
Generates LaTeX tables and publication-quality figures.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a publication-quality backend
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-paper')

# Set colorblind-friendly palette
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494']
MODEL_COLORS = {'anthropic/claude-sonnet-4.5': COLORS[0],
                'google/gemini-2.5-pro': COLORS[1],
                'openai/gpt-5': COLORS[2]}


class JudgeAnalyzer:
    """Analyzes judge results from cache_judge directory."""

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

    def compute_by_model(self) -> Dict:
        """Compute performance metrics by model."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            model = result.get('model_name', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[model]['total'] += 1
                if is_correct:
                    stats[model]['correct'] += 1
                else:
                    stats[model]['incorrect'] += 1

        # Calculate accuracy
        for model in stats:
            if stats[model]['total'] > 0:
                stats[model]['accuracy'] = stats[model]['correct'] / stats[model]['total']
            else:
                stats[model]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_evaluation_mode(self) -> Dict:
        """Compute performance metrics by evaluation mode."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            mode = result.get('evaluation_mode', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[mode]['total'] += 1
                if is_correct:
                    stats[mode]['correct'] += 1
                else:
                    stats[mode]['incorrect'] += 1

        # Calculate accuracy
        for mode in stats:
            if stats[mode]['total'] > 0:
                stats[mode]['accuracy'] = stats[mode]['correct'] / stats[mode]['total']
            else:
                stats[mode]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_level(self) -> Dict:
        """Compute performance metrics by difficulty level."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            level = result.get('problem_data', {}).get('level')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and level is not None:
                stats[level]['total'] += 1
                if is_correct:
                    stats[level]['correct'] += 1
                else:
                    stats[level]['incorrect'] += 1

        # Calculate accuracy
        for level in stats:
            if stats[level]['total'] > 0:
                stats[level]['accuracy'] = stats[level]['correct'] / stats[level]['total']
            else:
                stats[level]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_answer_type(self) -> Dict:
        """Compute performance metrics by answer type (multiple_choice vs numeric)."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            answer_type = result.get('problem_data', {}).get('answer_type', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[answer_type]['total'] += 1
                if is_correct:
                    stats[answer_type]['correct'] += 1
                else:
                    stats[answer_type]['incorrect'] += 1

        # Calculate accuracy
        for atype in stats:
            if stats[atype]['total'] > 0:
                stats[atype]['accuracy'] = stats[atype]['correct'] / stats[atype]['total']
            else:
                stats[atype]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_problem_type(self) -> Dict:
        """Compute performance metrics by problem type."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            problem_types = result.get('problem_data', {}).get('problem_types', [])
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and problem_types:
                # A problem can have multiple types
                for ptype in problem_types:
                    stats[ptype]['total'] += 1
                    if is_correct:
                        stats[ptype]['correct'] += 1
                    else:
                        stats[ptype]['incorrect'] += 1

        # Calculate accuracy
        for ptype in stats:
            if stats[ptype]['total'] > 0:
                stats[ptype]['accuracy'] = stats[ptype]['correct'] / stats[ptype]['total']
            else:
                stats[ptype]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_model_and_mode(self) -> Dict:
        """Compute performance metrics by model and evaluation mode combination."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            model = result.get('model_name', 'unknown')
            mode = result.get('evaluation_mode', 'unknown')
            key = f"{model}/{mode}"
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[key]['total'] += 1
                if is_correct:
                    stats[key]['correct'] += 1
                else:
                    stats[key]['incorrect'] += 1

        # Calculate accuracy
        for key in stats:
            if stats[key]['total'] > 0:
                stats[key]['accuracy'] = stats[key]['correct'] / stats[key]['total']
            else:
                stats[key]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_model_and_level(self) -> Dict:
        """Compute performance metrics by model and level combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            model = result.get('model_name', 'unknown')
            level = result.get('problem_data', {}).get('level')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and level is not None:
                stats[model][level]['total'] += 1
                if is_correct:
                    stats[model][level]['correct'] += 1

        # Calculate accuracy and incorrect count
        for model in stats:
            for level in stats[model]:
                total = stats[model][level]['total']
                correct = stats[model][level]['correct']
                if total > 0:
                    stats[model][level]['accuracy'] = correct / total
                    stats[model][level]['incorrect'] = total - correct
                else:
                    stats[model][level]['accuracy'] = 0.0
                    stats[model][level]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_model_and_answer_type(self) -> Dict:
        """Compute performance metrics by model and answer type combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            model = result.get('model_name', 'unknown')
            answer_type = result.get('problem_data', {}).get('answer_type', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[model][answer_type]['total'] += 1
                if is_correct:
                    stats[model][answer_type]['correct'] += 1

        # Calculate accuracy and incorrect count
        for model in stats:
            for atype in stats[model]:
                total = stats[model][atype]['total']
                correct = stats[model][atype]['correct']
                if total > 0:
                    stats[model][atype]['accuracy'] = correct / total
                    stats[model][atype]['incorrect'] = total - correct
                else:
                    stats[model][atype]['accuracy'] = 0.0
                    stats[model][atype]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_model_and_problem_type(self) -> Dict:
        """Compute performance metrics by model and problem type combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            model = result.get('model_name', 'unknown')
            problem_types = result.get('problem_data', {}).get('problem_types', [])
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and problem_types:
                for ptype in problem_types:
                    stats[model][ptype]['total'] += 1
                    if is_correct:
                        stats[model][ptype]['correct'] += 1

        # Calculate accuracy and incorrect count
        for model in stats:
            for ptype in stats[model]:
                total = stats[model][ptype]['total']
                correct = stats[model][ptype]['correct']
                if total > 0:
                    stats[model][ptype]['accuracy'] = correct / total
                    stats[model][ptype]['incorrect'] = total - correct
                else:
                    stats[model][ptype]['accuracy'] = 0.0
                    stats[model][ptype]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_mode_and_level(self) -> Dict:
        """Compute performance metrics by evaluation mode and level combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            mode = result.get('evaluation_mode', 'unknown')
            level = result.get('problem_data', {}).get('level')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and level is not None:
                stats[mode][level]['total'] += 1
                if is_correct:
                    stats[mode][level]['correct'] += 1

        # Calculate accuracy and incorrect count
        for mode in stats:
            for level in stats[mode]:
                total = stats[mode][level]['total']
                correct = stats[mode][level]['correct']
                if total > 0:
                    stats[mode][level]['accuracy'] = correct / total
                    stats[mode][level]['incorrect'] = total - correct
                else:
                    stats[mode][level]['accuracy'] = 0.0
                    stats[mode][level]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_mode_and_answer_type(self) -> Dict:
        """Compute performance metrics by evaluation mode and answer type combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            mode = result.get('evaluation_mode', 'unknown')
            answer_type = result.get('problem_data', {}).get('answer_type', 'unknown')
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[mode][answer_type]['total'] += 1
                if is_correct:
                    stats[mode][answer_type]['correct'] += 1

        # Calculate accuracy and incorrect count
        for mode in stats:
            for atype in stats[mode]:
                total = stats[mode][atype]['total']
                correct = stats[mode][atype]['correct']
                if total > 0:
                    stats[mode][atype]['accuracy'] = correct / total
                    stats[mode][atype]['incorrect'] = total - correct
                else:
                    stats[mode][atype]['accuracy'] = 0.0
                    stats[mode][atype]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_mode_and_problem_type(self) -> Dict:
        """Compute performance metrics by evaluation mode and problem type combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            mode = result.get('evaluation_mode', 'unknown')
            problem_types = result.get('problem_data', {}).get('problem_types', [])
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None and problem_types:
                for ptype in problem_types:
                    stats[mode][ptype]['total'] += 1
                    if is_correct:
                        stats[mode][ptype]['correct'] += 1

        # Calculate accuracy and incorrect count
        for mode in stats:
            for ptype in stats[mode]:
                total = stats[mode][ptype]['total']
                correct = stats[mode][ptype]['correct']
                if total > 0:
                    stats[mode][ptype]['accuracy'] = correct / total
                    stats[mode][ptype]['incorrect'] = total - correct
                else:
                    stats[mode][ptype]['accuracy'] = 0.0
                    stats[mode][ptype]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_image_presence(self) -> Dict:
        """Compute performance metrics by whether problem has an image."""
        stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})

        for result in self.results:
            has_image = result.get('problem_data', {}).get('has_image', False)
            image_key = 'with_image' if has_image else 'without_image'
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[image_key]['total'] += 1
                if is_correct:
                    stats[image_key]['correct'] += 1
                else:
                    stats[image_key]['incorrect'] += 1

        # Calculate accuracy
        for key in stats:
            if stats[key]['total'] > 0:
                stats[key]['accuracy'] = stats[key]['correct'] / stats[key]['total']
            else:
                stats[key]['accuracy'] = 0.0

        return dict(stats)

    def compute_by_model_and_image(self) -> Dict:
        """Compute performance metrics by model and image presence combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            model = result.get('model_name', 'unknown')
            has_image = result.get('problem_data', {}).get('has_image', False)
            image_key = 'with_image' if has_image else 'without_image'
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[model][image_key]['total'] += 1
                if is_correct:
                    stats[model][image_key]['correct'] += 1

        # Calculate accuracy and incorrect count
        for model in stats:
            for key in stats[model]:
                total = stats[model][key]['total']
                correct = stats[model][key]['correct']
                if total > 0:
                    stats[model][key]['accuracy'] = correct / total
                    stats[model][key]['incorrect'] = total - correct
                else:
                    stats[model][key]['accuracy'] = 0.0
                    stats[model][key]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def compute_by_mode_and_image(self) -> Dict:
        """Compute performance metrics by evaluation mode and image presence combination."""
        stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

        for result in self.results:
            mode = result.get('evaluation_mode', 'unknown')
            has_image = result.get('problem_data', {}).get('has_image', False)
            image_key = 'with_image' if has_image else 'without_image'
            is_correct = result.get('judge_result', {}).get('is_correct')

            if is_correct is not None:
                stats[mode][image_key]['total'] += 1
                if is_correct:
                    stats[mode][image_key]['correct'] += 1

        # Calculate accuracy and incorrect count
        for mode in stats:
            for key in stats[mode]:
                total = stats[mode][key]['total']
                correct = stats[mode][key]['correct']
                if total > 0:
                    stats[mode][key]['accuracy'] = correct / total
                    stats[mode][key]['incorrect'] = total - correct
                else:
                    stats[mode][key]['accuracy'] = 0.0
                    stats[mode][key]['incorrect'] = 0

        return {k: dict(v) for k, v in stats.items()}

    def print_table(self, headers: List[str], rows: List[List], title: str = None):
        """Print a formatted table."""
        if title:
            print(f"\n{title}")
            print("=" * len(title))

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = "  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_line)

    def print_summary(self):
        """Print comprehensive summary of all analyses."""
        print("\n" + "=" * 80)
        print("JUDGE ANALYSIS REPORT")
        print("=" * 80)

        # By Model
        by_model = self.compute_by_model()
        rows = []
        for model in sorted(by_model.keys()):
            stats = by_model[model]
            rows.append([
                model,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Model", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY MODEL"
        )

        # By Evaluation Mode
        by_mode = self.compute_by_evaluation_mode()
        rows = []
        for mode in sorted(by_mode.keys()):
            stats = by_mode[mode]
            rows.append([
                mode,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Mode", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY EVALUATION MODE"
        )

        # By Model and Mode
        by_model_mode = self.compute_by_model_and_mode()
        rows = []
        for key in sorted(by_model_mode.keys()):
            stats = by_model_mode[key]
            rows.append([
                key,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Model/Mode", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY MODEL AND MODE"
        )

        # By Level
        by_level = self.compute_by_level()
        rows = []
        for level in sorted(by_level.keys()):
            stats = by_level[level]
            rows.append([
                f"Level {level}",
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Level", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY DIFFICULTY LEVEL"
        )

        # By Answer Type
        by_answer_type = self.compute_by_answer_type()
        rows = []
        for atype in sorted(by_answer_type.keys()):
            stats = by_answer_type[atype]
            rows.append([
                atype,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Answer Type", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY ANSWER TYPE"
        )

        # By Problem Type
        by_problem_type = self.compute_by_problem_type()
        rows = []
        for ptype in sorted(by_problem_type.keys(), key=lambda x: by_problem_type[x]['accuracy'], reverse=True):
            stats = by_problem_type[ptype]
            rows.append([
                ptype,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Problem Type", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY PROBLEM TYPE (sorted by accuracy)"
        )

        # By Image Presence
        by_image = self.compute_by_image_presence()
        rows = []
        for key in sorted(by_image.keys()):
            stats = by_image[key]
            rows.append([
                key,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']*100:.2f}%"
            ])
        self.print_table(
            ["Image Presence", "Total", "Correct", "Incorrect", "Accuracy"],
            rows,
            "PERFORMANCE BY IMAGE PRESENCE"
        )

        # Detailed breakdowns by model
        print("\n" + "=" * 80)
        print("DETAILED BREAKDOWN BY MODEL")
        print("=" * 80)

        by_model_level = self.compute_by_model_and_level()
        by_model_answer = self.compute_by_model_and_answer_type()
        by_model_problem = self.compute_by_model_and_problem_type()
        by_model_image = self.compute_by_model_and_image()

        for model in sorted(by_model_level.keys()):
            model_short = model.split('/')[-1]
            print(f"\n{model}")
            print("-" * len(model))

            # By Level
            if model in by_model_level:
                rows = []
                for level in sorted(by_model_level[model].keys()):
                    stats = by_model_level[model][level]
                    rows.append([
                        f"Level {level}",
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Level", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Difficulty Level"
                )

            # By Answer Type
            if model in by_model_answer:
                rows = []
                for atype in sorted(by_model_answer[model].keys()):
                    stats = by_model_answer[model][atype]
                    rows.append([
                        atype,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Answer Type", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Answer Type"
                )

            # By Problem Type
            if model in by_model_problem:
                rows = []
                for ptype in sorted(by_model_problem[model].keys(),
                                   key=lambda x: by_model_problem[model][x]['accuracy'],
                                   reverse=True):
                    stats = by_model_problem[model][ptype]
                    rows.append([
                        ptype,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Problem Type", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Problem Type (sorted by accuracy)"
                )

            # By Image Presence
            if model in by_model_image:
                rows = []
                for key in sorted(by_model_image[model].keys()):
                    stats = by_model_image[model][key]
                    rows.append([
                        key,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Image Presence", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Image Presence"
                )

        # Detailed breakdowns by evaluation mode
        print("\n" + "=" * 80)
        print("DETAILED BREAKDOWN BY EVALUATION MODE")
        print("=" * 80)

        by_mode_level = self.compute_by_mode_and_level()
        by_mode_answer = self.compute_by_mode_and_answer_type()
        by_mode_problem = self.compute_by_mode_and_problem_type()
        by_mode_image = self.compute_by_mode_and_image()

        for mode in sorted(by_mode_level.keys()):
            print(f"\n{mode}")
            print("-" * len(mode))

            # By Level
            if mode in by_mode_level:
                rows = []
                for level in sorted(by_mode_level[mode].keys()):
                    stats = by_mode_level[mode][level]
                    rows.append([
                        f"Level {level}",
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Level", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Difficulty Level"
                )

            # By Answer Type
            if mode in by_mode_answer:
                rows = []
                for atype in sorted(by_mode_answer[mode].keys()):
                    stats = by_mode_answer[mode][atype]
                    rows.append([
                        atype,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Answer Type", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Answer Type"
                )

            # By Problem Type
            if mode in by_mode_problem:
                rows = []
                for ptype in sorted(by_mode_problem[mode].keys(),
                                   key=lambda x: by_mode_problem[mode][x]['accuracy'],
                                   reverse=True):
                    stats = by_mode_problem[mode][ptype]
                    rows.append([
                        ptype,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Problem Type", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Problem Type (sorted by accuracy)"
                )

            # By Image Presence
            if mode in by_mode_image:
                rows = []
                for key in sorted(by_mode_image[mode].keys()):
                    stats = by_mode_image[mode][key]
                    rows.append([
                        key,
                        stats['total'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']*100:.2f}%"
                    ])
                self.print_table(
                    ["Image Presence", "Total", "Correct", "Incorrect", "Accuracy"],
                    rows,
                    "By Image Presence"
                )

        print("\n" + "=" * 80)

    def generate_latex_tables(self, output_dir: str = "tables"):
        """Generate LaTeX tables for all key results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating LaTeX tables in {output_dir}/...")

        # Model short names for tables
        model_short_names = {
            'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
            'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
            'openai/gpt-5': 'GPT-5'
        }

        # Table 1: Performance by evaluation mode (transposed - models as columns)
        by_model_mode = self.compute_by_model_and_mode()
        models = sorted(set(['/'.join(k.split('/')[:2]) for k in by_model_mode.keys()]))

        with open(output_path / "table_mode.tex", 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{l" + "c" * len(models) + "}\n")
            f.write("\\toprule\n")
            f.write("Evaluation Mode")
            for model in models:
                f.write(f" & {model_short_names.get(model, model)}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            for mode in ['with_choices', 'without_choices']:
                mode_label = "With Choices" if mode == 'with_choices' else "Without Choices"
                f.write(mode_label)
                for model in models:
                    key = f"{model}/{mode}"
                    acc = by_model_mode.get(key, {}).get('accuracy', 0) * 100
                    f.write(f" & {acc:.2f}\\%")
                f.write(" \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Accuracy (\\%) by evaluation mode for each model.}\n")
            f.write("\\label{tab:mode}\n")
            f.write("\\end{table}\n")

        # Table 2: Performance by difficulty level (transposed - levels as columns)
        by_level = self.compute_by_level()
        by_model_level = self.compute_by_model_and_level()
        levels = sorted(by_level.keys())

        with open(output_path / "table_difficulty.tex", 'w') as f:
            f.write("\\begin{table*}[t]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write("\\begin{tabular}{l" + "c" * len(levels) + "}\n")
            f.write("\\toprule\n")
            f.write("Model")
            for level in levels:
                f.write(f" & L{level}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            # Total row (divide by 6: 3 models × 2 evaluation modes)
            f.write("Total")
            for level in levels:
                total = by_level[level]['total'] // 6
                f.write(f" & {total}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            # Model rows
            for model in sorted(by_model_level.keys()):
                f.write(model_short_names.get(model, model))
                for level in levels:
                    acc = by_model_level[model].get(level, {}).get('accuracy', 0) * 100
                    if acc > 0:
                        f.write(f" & {acc:.1f}\\%")
                    else:
                        f.write(" & --")
                f.write(" \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Accuracy (\\%) by difficulty level for each model. Top row shows total problem count per level.}\n")
            f.write("\\label{tab:difficulty}\n")
            f.write("\\end{table*}\n")

        # Table 3: Performance by problem type (transposed - types as columns)
        by_problem_type = self.compute_by_problem_type()
        by_model_problem = self.compute_by_model_and_problem_type()
        problem_types = ['algebra', 'talnafræði', 'rúmfræði', 'fléttufræði']
        problem_type_labels = ['Algebra', 'Number Theory', 'Geometry', 'Combinatorics']

        with open(output_path / "table_problemtype.tex", 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{l" + "c" * len(problem_types) + "}\n")
            f.write("\\toprule\n")
            f.write("Model")
            for label in problem_type_labels:
                f.write(f" & {label}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            # Total row (divide by 6: 3 models × 2 evaluation modes)
            f.write("Total")
            for ptype in problem_types:
                total = by_problem_type[ptype]['total'] // 6
                f.write(f" & {total}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            # Model rows
            for model in sorted(by_model_problem.keys()):
                f.write(model_short_names.get(model, model))
                for ptype in problem_types:
                    acc = by_model_problem[model].get(ptype, {}).get('accuracy', 0) * 100
                    f.write(f" & {acc:.1f}\\%")
                f.write(" \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Accuracy (\\%) by problem type for each model. Top row shows total problem count per type.}\n")
            f.write("\\label{tab:problemtype}\n")
            f.write("\\end{table}\n")

        # Table 4: Performance by image presence (transposed - image presence as columns)
        by_image = self.compute_by_image_presence()
        by_model_image = self.compute_by_model_and_image()

        with open(output_path / "table_images.tex", 'w') as f:
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Model & With Images & Without Images \\\\\n")
            f.write("\\midrule\n")

            # Total row (divide by 6: 3 models × 2 evaluation modes)
            with_total = by_image['with_image']['total'] // 6
            without_total = by_image['without_image']['total'] // 6
            f.write(f"Total & {with_total} & {without_total} \\\\\n")
            f.write("\\midrule\n")

            # Model rows
            for model in sorted(by_model_image.keys()):
                with_acc = by_model_image[model].get('with_image', {}).get('accuracy', 0) * 100
                without_acc = by_model_image[model].get('without_image', {}).get('accuracy', 0) * 100
                f.write(f"{model_short_names.get(model, model)} & {with_acc:.1f}\\% & {without_acc:.1f}\\% \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Accuracy (\\%) by image presence for each model. Top row shows total problem count.}\n")
            f.write("\\label{tab:images}\n")
            f.write("\\end{table}\n")

        print(f"✓ Generated 4 LaTeX tables in {output_dir}/")

    def generate_figures(self, output_dir: str = "figures"):
        """Generate publication-quality figures."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nGenerating figures in {output_dir}/...")

        # Set larger font sizes
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

        # Translation for problem types
        problem_type_translation = {
            'algebra': 'Algebra',
            'talnafræði': 'Number Theory',
            'rúmfræði': 'Geometry',
            'fléttufræði': 'Combinatorics'
        }

        # Figure 1: Evaluation mode impact per model
        by_model_mode = self.compute_by_model_and_mode()

        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted(set([k.split('/')[0] + '/' + k.split('/')[1] for k in by_model_mode.keys()]))

        x = np.arange(len(models))
        width = 0.35

        with_choices_acc = []
        without_choices_acc = []

        for model in models:
            with_key = f"{model}/with_choices"
            without_key = f"{model}/without_choices"
            with_choices_acc.append(by_model_mode.get(with_key, {}).get('accuracy', 0) * 100)
            without_choices_acc.append(by_model_mode.get(without_key, {}).get('accuracy', 0) * 100)

        bars1 = ax.bar(x - width/2, with_choices_acc, width, label='With Choices', color=COLORS[0])
        bars2 = ax.bar(x + width/2, without_choices_acc, width, label='Without Choices', color=COLORS[1])

        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels([model_short_names.get(m, m) for m in models], ha='center')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        ax.set_ylim(0, 105)  # Increased to accommodate labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_evaluation_mode.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Performance by difficulty level
        by_model_level = self.compute_by_model_and_level()

        fig, ax = plt.subplots(figsize=(8, 5))

        for model in sorted(by_model_level.keys()):
            levels = sorted(by_model_level[model].keys())
            accuracies = [by_model_level[model][level]['accuracy'] * 100 for level in levels]
            ax.plot(levels, accuracies, marker='o', linewidth=2.5, markersize=7,
                   label=model_short_names.get(model, model), color=MODEL_COLORS.get(model, COLORS[3]))

        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticks(range(1, 11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_difficulty.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: Performance by problem type
        by_model_problem = self.compute_by_model_and_problem_type()

        problem_types = ['algebra', 'talnafræði', 'rúmfræði', 'fléttufræði']
        problem_types_english = [problem_type_translation[pt] for pt in problem_types]

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(problem_types))
        width = 0.25

        models_list = sorted(by_model_problem.keys())
        bars_list = []
        for i, model in enumerate(models_list):
            accuracies = []
            for ptype in problem_types:
                acc = by_model_problem[model].get(ptype, {}).get('accuracy', 0) * 100
                accuracies.append(acc)

            offset = (i - 1) * width
            bars = ax.bar(x + offset, accuracies, width, label=model_short_names.get(model, model),
                         color=MODEL_COLORS.get(model, COLORS[i]))
            bars_list.append(bars)

        # Add value labels on top of bars
        for bars in bars_list:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Problem Type')
        ax.set_xticks(x)
        ax.set_xticklabels(problem_types_english, ha='center')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        ax.set_ylim(0, 105)  # Increased to accommodate labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_problemtype.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 4: Image presence impact
        by_model_image = self.compute_by_model_and_image()

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(models_list))
        width = 0.35

        with_image_acc = []
        without_image_acc = []

        for model in models_list:
            with_acc = by_model_image[model].get('with_image', {}).get('accuracy', 0) * 100
            without_acc = by_model_image[model].get('without_image', {}).get('accuracy', 0) * 100
            with_image_acc.append(with_acc)
            without_image_acc.append(without_acc)

        bars1 = ax.bar(x - width/2, with_image_acc, width, label='With Images', color=COLORS[0])
        bars2 = ax.bar(x + width/2, without_image_acc, width, label='Without Images', color=COLORS[1])

        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels([model_short_names.get(m, m) for m in models_list], ha='center')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
        ax.set_ylim(0, 105)  # Increased to accommodate labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "fig_multimodal.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Generated 4 figures in {output_dir}/")

    def save_report(self, output_file: str = "judge_analysis_report.json"):
        """Save detailed analysis report to JSON file."""
        report = {
            "total_results": len(self.results),
            "by_model": self.compute_by_model(),
            "by_evaluation_mode": self.compute_by_evaluation_mode(),
            "by_level": self.compute_by_level(),
            "by_answer_type": self.compute_by_answer_type(),
            "by_problem_type": self.compute_by_problem_type(),
            "by_image_presence": self.compute_by_image_presence(),
            "by_model_and_mode": self.compute_by_model_and_mode(),
            "by_model_and_level": self.compute_by_model_and_level(),
            "by_model_and_answer_type": self.compute_by_model_and_answer_type(),
            "by_model_and_problem_type": self.compute_by_model_and_problem_type(),
            "by_model_and_image": self.compute_by_model_and_image(),
            "by_mode_and_level": self.compute_by_mode_and_level(),
            "by_mode_and_answer_type": self.compute_by_mode_and_answer_type(),
            "by_mode_and_problem_type": self.compute_by_mode_and_problem_type(),
            "by_mode_and_image": self.compute_by_mode_and_image()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Detailed report saved to: {output_file}")


class InterJudgeAnalyzer:
    """Analyzes agreement between multiple judge models."""

    JUDGE_DIRS = {
        "GPT-5": "cache_judge",
        "Gemini 3 Flash": "cache_judge_gemini-3-flash-preview",
        "Claude Sonnet 4.6": "cache_judge_claude-sonnet-4.6",
    }

    def __init__(self, judge_dirs: Optional[Dict[str, str]] = None):
        """
        Initialize inter-judge analyzer.

        Args:
            judge_dirs: Mapping of judge name -> cache directory path.
                        Defaults to the three standard judge directories.
        """
        self.judge_dirs = judge_dirs or self.JUDGE_DIRS
        # judgments[judge_name][(model, eval_mode, problem_id)] = bool
        self.judgments: Dict[str, Dict[tuple, bool]] = {}

    def load_all(self) -> Dict[str, int]:
        """Load judgments from all judge directories.

        Returns:
            Mapping of judge name -> number of results loaded
        """
        counts = {}
        for judge_name, cache_dir in self.judge_dirs.items():
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                print(f"  Warning: {cache_dir} not found, skipping {judge_name}")
                counts[judge_name] = 0
                continue

            judge_data = {}
            for f in cache_path.rglob("*.json"):
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    key = (
                        data.get('model_name'),
                        data.get('evaluation_mode'),
                        data.get('problem_id')
                    )
                    judge_result = data.get('judge_result', {})
                    judge_data[key] = bool(judge_result.get('is_correct', False))
                except Exception:
                    continue

            self.judgments[judge_name] = judge_data
            counts[judge_name] = len(judge_data)
            print(f"  Loaded {len(judge_data)} judgments from {judge_name} ({cache_dir})")

        return counts

    def _get_common_keys(self, judge_a: str, judge_b: str) -> set:
        """Get keys present in both judges."""
        return set(self.judgments.get(judge_a, {}).keys()) & set(self.judgments.get(judge_b, {}).keys())

    def pairwise_agreement(self, judge_a: str, judge_b: str) -> Dict:
        """Compute pairwise agreement and Cohen's Kappa between two judges.

        Returns:
            Dict with 'n', 'agreement', 'cohens_kappa'
        """
        common = self._get_common_keys(judge_a, judge_b)
        if not common:
            return {'n': 0, 'agreement': 0.0, 'cohens_kappa': 0.0}

        a_vals = [self.judgments[judge_a][k] for k in common]
        b_vals = [self.judgments[judge_b][k] for k in common]

        n = len(common)
        agree = sum(1 for a, b in zip(a_vals, b_vals) if a == b)
        p_o = agree / n  # observed agreement

        # Marginal proportions for Cohen's Kappa
        a_pos = sum(a_vals) / n
        b_pos = sum(b_vals) / n
        p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)  # expected agreement

        if p_e == 1.0:
            kappa = 1.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        return {
            'n': n,
            'agreement': p_o,
            'cohens_kappa': kappa
        }

    def fleiss_kappa(self) -> Dict:
        """Compute Fleiss' Kappa for all judges on common items.

        Returns:
            Dict with 'n', 'n_raters', 'fleiss_kappa'
        """
        judge_names = [j for j in self.judgments if self.judgments[j]]
        if len(judge_names) < 2:
            return {'n': 0, 'n_raters': len(judge_names), 'fleiss_kappa': 0.0}

        # Find items rated by all judges
        common = set(self.judgments[judge_names[0]].keys())
        for j in judge_names[1:]:
            common &= set(self.judgments[j].keys())

        if not common:
            return {'n': 0, 'n_raters': len(judge_names), 'fleiss_kappa': 0.0}

        n = len(common)
        k = len(judge_names)  # number of raters
        categories = 2  # True/False

        # Build rating matrix: n items x 2 categories
        # Each cell = number of raters who assigned that category
        common_list = sorted(common)
        matrix = []
        for key in common_list:
            true_count = sum(1 for j in judge_names if self.judgments[j][key])
            matrix.append([true_count, k - true_count])

        matrix = np.array(matrix, dtype=float)

        # Fleiss' Kappa computation
        # P_i for each item
        P_i = (np.sum(matrix ** 2, axis=1) - k) / (k * (k - 1))
        P_bar = np.mean(P_i)

        # p_j for each category
        p_j = np.sum(matrix, axis=0) / (n * k)
        P_e = np.sum(p_j ** 2)

        if P_e == 1.0:
            fleiss = 1.0
        else:
            fleiss = (P_bar - P_e) / (1 - P_e)

        return {
            'n': n,
            'n_raters': k,
            'fleiss_kappa': fleiss
        }

    def print_summary(self):
        """Print inter-judge agreement summary to console."""
        judge_names = [j for j in self.judgments if self.judgments[j]]

        if len(judge_names) < 2:
            print("Need at least 2 judges with results for agreement analysis.")
            return

        print("\n" + "=" * 70)
        print("INTER-JUDGE AGREEMENT ANALYSIS")
        print("=" * 70)

        # Pairwise
        print("\nPairwise Agreement:")
        print(f"{'Judge Pair':<40} {'N':>6} {'Agree%':>8} {'Cohen κ':>10}")
        print("-" * 66)
        for i, a in enumerate(judge_names):
            for b in judge_names[i+1:]:
                stats = self.pairwise_agreement(a, b)
                print(f"{a} vs {b:<20} {stats['n']:>6} "
                      f"{stats['agreement']*100:>7.2f}% {stats['cohens_kappa']:>9.4f}")

        # Fleiss
        fleiss = self.fleiss_kappa()
        print(f"\nFleiss' Kappa (all {fleiss['n_raters']} judges, {fleiss['n']} common items): "
              f"{fleiss['fleiss_kappa']:.4f}")
        print("=" * 70)

    def generate_latex_table(self, output_dir: str = "tables"):
        """Generate LaTeX table for inter-judge agreement.

        Args:
            output_dir: Directory to save the table
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        judge_names = [j for j in self.judgments if self.judgments[j]]
        if len(judge_names) < 2:
            return

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Judge Pair & $N$ & Agreement (\%) & Cohen's $\kappa$ \\",
            r"\midrule",
        ]

        for i, a in enumerate(judge_names):
            for b in judge_names[i+1:]:
                stats = self.pairwise_agreement(a, b)
                lines.append(
                    f"{a} vs. {b} & {stats['n']} & "
                    f"{stats['agreement']*100:.1f} & {stats['cohens_kappa']:.3f} \\\\"
                )

        fleiss = self.fleiss_kappa()
        lines.extend([
            r"\midrule",
            f"\\multicolumn{{4}}{{l}}{{Fleiss' $\\kappa$ (all {fleiss['n_raters']} judges, "
            f"$N={fleiss['n']}$): {fleiss['fleiss_kappa']:.3f}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Inter-judge agreement between three LLM judges (GPT-5, Gemini 3 Flash, "
            r"and Claude Sonnet 4.6) on correctness judgments. High agreement rates and substantial "
            r"Cohen's $\kappa$ values indicate consistency across judge models.}",
            r"\label{tab:judge_agreement}",
            r"\end{table}",
        ])

        table_file = output_path / "table_judge_agreement.tex"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

        print(f"  Generated {table_file}")

    def save_report(self, output_file: str = "inter_judge_report.json"):
        """Save inter-judge agreement report to JSON."""
        judge_names = [j for j in self.judgments if self.judgments[j]]
        report = {"judges": judge_names, "pairwise": {}, "fleiss": self.fleiss_kappa()}

        for i, a in enumerate(judge_names):
            for b in judge_names[i+1:]:
                key = f"{a} vs {b}"
                report["pairwise"][key] = self.pairwise_agreement(a, b)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"  Inter-judge report saved to {output_file}")


def main():
    """Main execution function."""
    # Check if cache_judge directory exists
    if not Path("cache_judge").exists():
        print("Error: cache_judge directory not found")
        print("Please run llm_judge.py first to generate judge results")
        sys.exit(1)

    # Create analyzer
    analyzer = JudgeAnalyzer()

    # Load results
    num_results = analyzer.load_all_results()

    if num_results == 0:
        print("No results found in cache_judge directory")
        sys.exit(1)

    # Print summary
    analyzer.print_summary()

    # Save detailed report
    analyzer.save_report()

    # Generate LaTeX tables
    analyzer.generate_latex_tables()

    # Generate figures
    analyzer.generate_figures()

    # Inter-judge agreement analysis (if multiple judge dirs exist)
    multi_judge_dirs = {
        name: path for name, path in InterJudgeAnalyzer.JUDGE_DIRS.items()
        if Path(path).exists()
    }
    if len(multi_judge_dirs) >= 2:
        print("\n" + "=" * 70)
        print("Running inter-judge agreement analysis...")
        inter = InterJudgeAnalyzer(multi_judge_dirs)
        inter.load_all()
        inter.print_summary()
        inter.generate_latex_table()
        inter.save_report()
    else:
        available = [name for name, path in InterJudgeAnalyzer.JUDGE_DIRS.items() if Path(path).exists()]
        print(f"\nInter-judge analysis skipped (found {len(available)} judge dir(s): {available})")
        print("Run llm_judge.py with --judge-model for additional judges.")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
