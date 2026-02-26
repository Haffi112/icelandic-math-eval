"""
Data Loader for Icelandic Math Problems
Loads and parses JSON files containing math competition problems.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional


class Problem:
    """Represents a single math problem."""

    def __init__(self, data: Dict, source_file: str, problem_index: int):
        self.source_file = source_file
        self.problem_index = problem_index
        self.problem_text = data.get("problem_text", "")
        self.problem_types = data.get("problem_types", [])
        self.level = data.get("level")
        self.source_info = data.get("source_info", "")

        # Determine problem type
        self.answer_type = data.get("answer_type", "multiple_choice")

        # Multiple choice fields
        self.choice_a = data.get("choice_a")
        self.choice_b = data.get("choice_b")
        self.choice_c = data.get("choice_c")
        self.choice_d = data.get("choice_d")
        self.correct_answer = data.get("correct_answer")  # Letter for MC

        # Numeric answer fields
        self.correct_answer_numeric = data.get("correct_answer_numeric")
        self.numeric_tolerance = data.get("numeric_tolerance", 0)

        # Image path (needs to be remapped)
        self.image = data.get("image")

    @property
    def problem_id(self) -> str:
        """Generate unique problem ID."""
        basename = Path(self.source_file).stem
        return f"{basename}_p{self.problem_index:03d}"

    @property
    def has_image(self) -> bool:
        """Check if problem has an associated image."""
        return self.image is not None and self.image != ""

    @property
    def is_multiple_choice(self) -> bool:
        """Check if problem is multiple choice."""
        return self.answer_type != "numeric"

    def get_remapped_image_path(self, image_dir: str) -> Optional[str]:
        """
        Remap image path from JSON to actual filesystem path.

        JSON uses: game/static/images/problem_images/e2425d10.png
        Actual path: ./STAK-daemi/Myndir/e2425d10.png
        """
        if not self.has_image:
            return None

        # Extract just the filename
        filename = Path(self.image).name
        full_path = os.path.join(image_dir, filename)

        if os.path.exists(full_path):
            return full_path

        return None

    def to_dict(self) -> Dict:
        """Convert problem to dictionary for serialization."""
        return {
            "problem_id": self.problem_id,
            "source_file": self.source_file,
            "problem_index": self.problem_index,
            "problem_text": self.problem_text,
            "problem_types": self.problem_types,
            "level": self.level,
            "source_info": self.source_info,
            "answer_type": self.answer_type,
            "choice_a": self.choice_a,
            "choice_b": self.choice_b,
            "choice_c": self.choice_c,
            "choice_d": self.choice_d,
            "correct_answer": self.correct_answer,
            "correct_answer_numeric": self.correct_answer_numeric,
            "numeric_tolerance": self.numeric_tolerance,
            "image": self.image,
            "has_image": self.has_image,
        }


class DataLoader:
    """Loads math problems from JSON files."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all_problems(self) -> List[Problem]:
        """Load all problems from all JSON files in the data directory."""
        problems = []

        json_files = sorted(self.data_dir.glob("*.json"))

        for json_file in json_files:
            problems.extend(self.load_problems_from_file(json_file))

        return problems

    def load_problems_from_file(self, file_path: Path) -> List[Problem]:
        """Load problems from a single JSON file."""
        problems = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list")
                return problems

            for idx, problem_data in enumerate(data):
                try:
                    problem = Problem(problem_data, str(file_path), idx)
                    problems.append(problem)
                except Exception as e:
                    print(f"Error parsing problem {idx} in {file_path}: {e}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

        return problems

    def get_statistics(self, problems: List[Problem]) -> Dict:
        """Get statistics about loaded problems."""
        total = len(problems)
        multiple_choice = sum(1 for p in problems if p.is_multiple_choice)
        numeric = total - multiple_choice
        with_images = sum(1 for p in problems if p.has_image)

        problem_types = {}
        for p in problems:
            for pt in p.problem_types:
                problem_types[pt] = problem_types.get(pt, 0) + 1

        levels = {}
        for p in problems:
            if p.level:
                levels[p.level] = levels.get(p.level, 0) + 1

        return {
            "total_problems": total,
            "multiple_choice": multiple_choice,
            "numeric": numeric,
            "with_images": with_images,
            "problem_types": problem_types,
            "levels": levels,
        }
