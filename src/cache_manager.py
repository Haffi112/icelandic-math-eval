"""
Cache Manager for Evaluation Results
Manages caching of LLM responses to avoid duplicate evaluations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class CacheManager:
    """Manages caching of evaluation results."""

    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for storing cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str
    ) -> Path:
        """
        Get path to cache file for a specific evaluation.

        Args:
            model_name: Name of the model
            evaluation_mode: Evaluation mode ("with_choices" or "without_choices")
            problem_id: Unique problem identifier

        Returns:
            Path to cache file
        """
        # Sanitize model name for filesystem
        safe_model_name = model_name.replace("/", "_").replace(":", "_")

        # Create directory structure: cache/model_name/evaluation_mode/
        cache_subdir = self.cache_dir / safe_model_name / evaluation_mode
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # Cache file: problem_id.json
        return cache_subdir / f"{problem_id}.json"

    def has_cached_result(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str
    ) -> bool:
        """
        Check if a cached result exists.

        Args:
            model_name: Name of the model
            evaluation_mode: Evaluation mode
            problem_id: Unique problem identifier

        Returns:
            True if cached result exists
        """
        cache_path = self._get_cache_path(model_name, evaluation_mode, problem_id)
        return cache_path.exists()

    def get_cached_result(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str
    ) -> Optional[Dict]:
        """
        Retrieve cached result.

        Args:
            model_name: Name of the model
            evaluation_mode: Evaluation mode
            problem_id: Unique problem identifier

        Returns:
            Cached result dictionary, or None if not found
        """
        cache_path = self._get_cache_path(model_name, evaluation_mode, problem_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache file {cache_path}: {e}")
            return None

    def save_result(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str,
        problem_data: Dict,
        prompt_summary: str,
        llm_response: Dict,
        extracted_answer: Optional[str],
        is_correct: Optional[bool],
        correct_answer: Optional[str]
    ) -> bool:
        """
        Save evaluation result to cache.

        Args:
            model_name: Name of the model
            evaluation_mode: Evaluation mode
            problem_id: Unique problem identifier
            problem_data: Dictionary with problem metadata
            prompt_summary: Text summary of the prompt used
            llm_response: Dictionary with LLM response data
            extracted_answer: Extracted answer from LLM
            is_correct: Whether answer is correct (None if can't determine)
            correct_answer: The correct answer

        Returns:
            True if saved successfully
        """
        cache_path = self._get_cache_path(model_name, evaluation_mode, problem_id)

        result = {
            "model_name": model_name,
            "evaluation_mode": evaluation_mode,
            "problem_id": problem_id,
            "timestamp": datetime.now().isoformat(),
            "problem_data": problem_data,
            "prompt_summary": prompt_summary,
            "llm_response": llm_response,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving cache file {cache_path}: {e}")
            return False

    def get_cache_statistics(self) -> Dict:
        """
        Get statistics about cached results.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_cached": 0,
            "by_model": {},
            "by_evaluation_mode": {},
        }

        for cache_file in self.cache_dir.rglob("*.json"):
            stats["total_cached"] += 1

            # Try to parse to get model and mode
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    model = data.get("model_name", "unknown")
                    mode = data.get("evaluation_mode", "unknown")

                    stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
                    stats["by_evaluation_mode"][mode] = stats["by_evaluation_mode"].get(mode, 0) + 1
            except:
                pass

        return stats

    def clear_cache(
        self,
        model_name: Optional[str] = None,
        evaluation_mode: Optional[str] = None
    ) -> int:
        """
        Clear cache files.

        Args:
            model_name: If specified, only clear cache for this model
            evaluation_mode: If specified, only clear cache for this mode

        Returns:
            Number of files deleted
        """
        count = 0

        if model_name and evaluation_mode:
            # Clear specific model + mode combination
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            cache_subdir = self.cache_dir / safe_model_name / evaluation_mode

            if cache_subdir.exists():
                for cache_file in cache_subdir.glob("*.json"):
                    cache_file.unlink()
                    count += 1

        elif model_name:
            # Clear all for specific model
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            model_dir = self.cache_dir / safe_model_name

            if model_dir.exists():
                for cache_file in model_dir.rglob("*.json"):
                    cache_file.unlink()
                    count += 1

        else:
            # Clear all cache
            for cache_file in self.cache_dir.rglob("*.json"):
                cache_file.unlink()
                count += 1

        return count
