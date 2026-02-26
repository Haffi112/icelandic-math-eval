"""
Evaluator for Math Problems
Main orchestration for evaluating LLMs on math problems.
"""

import asyncio
from typing import List, Dict, Optional
from tqdm import tqdm
import aiohttp
from .data_loader import Problem, DataLoader
from .prompt_builder import PromptBuilder
from .model_client import ModelClient
from .cache_manager import CacheManager
from .image_handler import ImageHandler


class Evaluator:
    """Orchestrates the evaluation of LLMs on math problems."""

    def __init__(
        self,
        config: Dict,
        model_client: ModelClient,
        cache_manager: CacheManager
    ):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            model_client: Model client instance
            cache_manager: Cache manager instance
        """
        self.config = config
        self.model_client = model_client
        self.cache_manager = cache_manager
        self.prompt_builder = PromptBuilder(config)

        self.evaluation_mode = config.get("evaluation_mode", "with_choices")
        self.image_dir = config.get("image_dir", "./STAK-daemi/Myndir")
        self.max_parallel_requests = config.get("max_parallel_requests", 10)

    def _is_failed_result(self, result: Dict) -> bool:
        """
        Check if a cached result represents a failed generation.

        Args:
            result: Cached result dictionary

        Returns:
            True if the result is failed/empty and should be retried
        """
        if not result:
            return True

        # Check if llm_response is None or missing
        llm_response = result.get("llm_response")
        if llm_response is None:
            return True

        # Check if response field is empty or missing
        response_text = llm_response.get("response", "")
        if not response_text or response_text.strip() == "":
            return True

        # Check if extracted_answer is None (extraction failed)
        # Note: extracted_answer can legitimately be None if the answer couldn't be parsed,
        # but we should still retry in that case
        if result.get("extracted_answer") is None:
            return True

        return False

    def evaluate_problem(
        self,
        problem: Problem,
        model: str
    ) -> Dict:
        """
        Evaluate a single problem with a single model.

        Args:
            problem: Problem to evaluate
            model: Model identifier

        Returns:
            Dictionary with evaluation results
        """
        problem_id = problem.problem_id

        # Check cache first
        cached_result = self.cache_manager.get_cached_result(
            model, self.evaluation_mode, problem_id
        )

        # If cached result exists and is not failed, return it
        if cached_result and not self._is_failed_result(cached_result):
            return cached_result

        # If we reach here, either no cache exists or cached result is failed
        # In either case, we'll query the model and overwrite the cache

        # Get image path if problem has image
        image_path = None
        if problem.has_image:
            image_path = problem.get_remapped_image_path(self.image_dir)
            if image_path and not ImageHandler.validate_image(image_path):
                print(f"Warning: Invalid image for problem {problem_id}")
                image_path = None

        # Build messages
        messages = self.prompt_builder.build_messages(
            problem, self.evaluation_mode, image_path
        )

        # Get prompt summary for caching
        prompt_summary = self.prompt_builder.get_prompt_summary(
            problem, self.evaluation_mode, image_path
        )

        # Query model
        llm_response = self.model_client.query_model(model, messages)

        if not llm_response:
            # Failed to get response
            result = {
                "model_name": model,
                "evaluation_mode": self.evaluation_mode,
                "problem_id": problem_id,
                "problem_data": problem.to_dict(),
                "prompt_summary": prompt_summary,
                "llm_response": None,
                "extracted_answer": None,
                "correct_answer": self._get_correct_answer(problem),
                "is_correct": None,
                "error": "Failed to get response from model"
            }
            return result

        # Extract answer from response
        extracted_answer = self.model_client.extract_answer(
            llm_response.get("response", ""),
            self.evaluation_mode,
            problem.answer_type
        )

        # Check if correct
        correct_answer = self._get_correct_answer(problem)
        is_correct = self._check_answer(
            extracted_answer,
            correct_answer,
            problem.answer_type,
            problem.numeric_tolerance
        )

        # Save to cache
        self.cache_manager.save_result(
            model_name=model,
            evaluation_mode=self.evaluation_mode,
            problem_id=problem_id,
            problem_data=problem.to_dict(),
            prompt_summary=prompt_summary,
            llm_response=llm_response,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            correct_answer=correct_answer
        )

        # Return result
        result = {
            "model_name": model,
            "evaluation_mode": self.evaluation_mode,
            "problem_id": problem_id,
            "problem_data": problem.to_dict(),
            "prompt_summary": prompt_summary,
            "llm_response": llm_response,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }

        return result

    async def evaluate_problem_async(
        self,
        problem: Problem,
        model: str,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore
    ) -> Dict:
        """
        Evaluate a single problem with a single model asynchronously.

        Args:
            problem: Problem to evaluate
            model: Model identifier
            session: aiohttp session for connection pooling
            semaphore: Semaphore to limit concurrent requests

        Returns:
            Dictionary with evaluation results
        """
        problem_id = problem.problem_id

        # Check cache first
        cached_result = self.cache_manager.get_cached_result(
            model, self.evaluation_mode, problem_id
        )

        # If cached result exists and is not failed, return it
        if cached_result and not self._is_failed_result(cached_result):
            return cached_result

        # If we reach here, either no cache exists or cached result is failed
        # In either case, we'll query the model and overwrite the cache

        # Get image path if problem has image
        image_path = None
        if problem.has_image:
            image_path = problem.get_remapped_image_path(self.image_dir)
            if image_path and not ImageHandler.validate_image(image_path):
                print(f"Warning: Invalid image for problem {problem_id}")
                image_path = None

        # Build messages
        messages = self.prompt_builder.build_messages(
            problem, self.evaluation_mode, image_path
        )

        # Get prompt summary for caching
        prompt_summary = self.prompt_builder.get_prompt_summary(
            problem, self.evaluation_mode, image_path
        )

        # Use semaphore to limit concurrent requests
        async with semaphore:
            # Query model asynchronously
            llm_response = await self.model_client.query_model_async(
                model, messages, session=session
            )

        if not llm_response:
            # Failed to get response
            result = {
                "model_name": model,
                "evaluation_mode": self.evaluation_mode,
                "problem_id": problem_id,
                "problem_data": problem.to_dict(),
                "prompt_summary": prompt_summary,
                "llm_response": None,
                "extracted_answer": None,
                "correct_answer": self._get_correct_answer(problem),
                "is_correct": None,
                "error": "Failed to get response from model"
            }
            return result

        # Extract answer from response
        extracted_answer = self.model_client.extract_answer(
            llm_response.get("response", ""),
            self.evaluation_mode,
            problem.answer_type
        )

        # Check if correct
        correct_answer = self._get_correct_answer(problem)
        is_correct = self._check_answer(
            extracted_answer,
            correct_answer,
            problem.answer_type,
            problem.numeric_tolerance
        )

        # Save to cache
        self.cache_manager.save_result(
            model_name=model,
            evaluation_mode=self.evaluation_mode,
            problem_id=problem_id,
            problem_data=problem.to_dict(),
            prompt_summary=prompt_summary,
            llm_response=llm_response,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            correct_answer=correct_answer
        )

        # Return result
        result = {
            "model_name": model,
            "evaluation_mode": self.evaluation_mode,
            "problem_id": problem_id,
            "problem_data": problem.to_dict(),
            "prompt_summary": prompt_summary,
            "llm_response": llm_response,
            "extracted_answer": extracted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }

        return result

    async def evaluate_model_async(
        self,
        problems: List[Problem],
        model: str
    ) -> List[Dict]:
        """
        Evaluate all problems for a single model with parallel requests.

        Args:
            problems: List of problems to evaluate
            model: Model identifier

        Returns:
            List of evaluation results
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_parallel_requests)

        # Create shared aiohttp session for connection pooling
        async with aiohttp.ClientSession() as session:
            # Create tasks for all problems
            tasks = [
                self.evaluate_problem_async(problem, model, session, semaphore)
                for problem in problems
            ]

            # Run tasks with progress bar
            results = []
            with tqdm(total=len(problems), desc=f"Evaluating {model}", leave=True) as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)

            # Sort results to match original problem order
            problem_id_to_result = {r["problem_id"]: r for r in results}
            sorted_results = [
                problem_id_to_result[problem.problem_id]
                for problem in problems
            ]

            return sorted_results

    def evaluate_all(
        self,
        problems: List[Problem],
        models: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Evaluate all problems with all models.
        Each model is evaluated sequentially, but problems within each model
        are evaluated in parallel (up to max_parallel_requests).

        Args:
            problems: List of problems to evaluate
            models: List of model identifiers

        Returns:
            Dictionary mapping model names to lists of results
        """
        all_results = {}

        for model in models:
            print(f"\n{'='*60}")
            print(f"Starting evaluation for: {model}")
            print(f"{'='*60}")

            # Run async evaluation for this model
            results = asyncio.run(self.evaluate_model_async(problems, model))
            all_results[model] = results

        return all_results

    def _get_correct_answer(self, problem: Problem) -> Optional[str]:
        """Get the correct answer for a problem."""
        if problem.is_multiple_choice:
            return problem.correct_answer
        else:
            return str(problem.correct_answer_numeric)

    def _check_answer(
        self,
        extracted_answer: Optional[str],
        correct_answer: Optional[str],
        answer_type: str,
        tolerance: float = 0
    ) -> Optional[bool]:
        """
        Check if extracted answer is correct.

        Args:
            extracted_answer: Answer extracted from LLM
            correct_answer: Correct answer
            answer_type: Type of answer ("multiple_choice" or "numeric")
            tolerance: Tolerance for numeric answers

        Returns:
            True if correct, False if incorrect, None if can't determine
        """
        if extracted_answer is None or correct_answer is None:
            return None

        if answer_type == "multiple_choice":
            # Case-insensitive comparison for letters
            return extracted_answer.upper() == correct_answer.upper()
        else:
            # Numeric comparison with tolerance
            try:
                extracted_num = float(extracted_answer)
                correct_num = float(correct_answer)

                if tolerance > 0:
                    return abs(extracted_num - correct_num) <= tolerance
                else:
                    return extracted_num == correct_num

            except (ValueError, TypeError):
                # If conversion fails, do string comparison
                return extracted_answer == correct_answer

    def compute_statistics(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Compute statistics for evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with statistics
        """
        total = len(results)
        if total == 0:
            return {}

        correct = sum(1 for r in results if r.get("is_correct") is True)
        incorrect = sum(1 for r in results if r.get("is_correct") is False)
        failed = sum(1 for r in results if r.get("is_correct") is None)

        accuracy = correct / total if total > 0 else 0

        # Statistics by problem type
        by_type = {}
        for result in results:
            problem_types = result["problem_data"].get("problem_types", [])
            for ptype in problem_types:
                if ptype not in by_type:
                    by_type[ptype] = {"total": 0, "correct": 0}
                by_type[ptype]["total"] += 1
                if result.get("is_correct") is True:
                    by_type[ptype]["correct"] += 1

        # Add accuracy for each type
        for ptype in by_type:
            total_type = by_type[ptype]["total"]
            by_type[ptype]["accuracy"] = (
                by_type[ptype]["correct"] / total_type if total_type > 0 else 0
            )

        # Statistics by level
        by_level = {}
        for result in results:
            level = result["problem_data"].get("level")
            if level is not None:
                if level not in by_level:
                    by_level[level] = {"total": 0, "correct": 0}
                by_level[level]["total"] += 1
                if result.get("is_correct") is True:
                    by_level[level]["correct"] += 1

        # Add accuracy for each level
        for level in by_level:
            total_level = by_level[level]["total"]
            by_level[level]["accuracy"] = (
                by_level[level]["correct"] / total_level if total_level > 0 else 0
            )

        return {
            "total_problems": total,
            "correct": correct,
            "incorrect": incorrect,
            "failed": failed,
            "accuracy": accuracy,
            "by_problem_type": by_type,
            "by_level": by_level,
        }
