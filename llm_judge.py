#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script
Uses an LLM judge to evaluate whether LLM responses to math problems are correct.
Supports multiple judge models: GPT-5 (default), Gemini, Claude via OpenRouter.
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
import aiohttp
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

from src.cache_manager import CacheManager
from src.image_handler import ImageHandler


class JudgmentResult(BaseModel):
    """Structured output for judge evaluation."""
    is_correct: bool
    explanation: str


class JudgePromptBuilder:
    """Builds prompts for the LLM judge."""

    @staticmethod
    def build_system_prompt(json_mode: bool = False) -> str:
        """Build system prompt for the judge (in English).

        Args:
            json_mode: If True, append JSON output instructions for OpenRouter models.
        """
        base = """You are an expert mathematics evaluator. Your task is to determine whether a given answer to a mathematical problem is correct.

You will be provided with:
1. A problem statement (in Icelandic)
2. The correct answer
3. The answer provided by an LLM

Your job is to evaluate whether the LLM's answer matches the correct answer. Consider the following:
- For multiple choice questions, the LLM's answer is correct if it either:
  * Matches the correct letter (A, B, C, or D), OR
  * Matches the actual value/content of the correct choice
  * Be flexible with formatting (e.g., '2000' matches '2000 kr.', '$2000$', or similar variations)
- For numeric answers, extract the final numerical answer from the LLM's response and compare it to the correct answer
- The LLM may provide reasoning or explanations - focus on the final answer
- Minor formatting differences are acceptable if the mathematical content is correct
- Be objective and fair in your evaluation

Note: The problem statements and answers are in Icelandic, but you should evaluate them objectively based on mathematical correctness."""

        if json_mode:
            base += """

You MUST respond with a JSON object containing exactly two fields:
- "is_correct": a boolean (true or false) indicating whether the answer is correct
- "explanation": a string explaining your reasoning

Example: {"is_correct": true, "explanation": "The LLM correctly identified the answer as 42."}"""

        return base

    @staticmethod
    def build_user_prompt(
        problem_text: str,
        correct_answer: str,
        llm_response: str,
        answer_type: str,
        extracted_answer: Optional[str] = None,
        image_path: Optional[str] = None,
        choice_values: Optional[Dict[str, str]] = None
    ) -> List[Dict] | str:
        """
        Build user prompt for the judge.

        Args:
            problem_text: The original problem statement (in Icelandic)
            correct_answer: The correct answer
            llm_response: The LLM's response
            answer_type: Type of answer ("multiple_choice" or "numeric")
            extracted_answer: The answer extracted from the LLM response
            image_path: Path to image if problem has one
            choice_values: Dictionary mapping choice letters to their values (for multiple choice)

        Returns:
            String or multimodal content list
        """
        # Build text content based on answer type
        if answer_type == "multiple_choice" and choice_values:
            # Format choices nicely
            choices_text = "\n".join([f"{k}) {v}" for k, v in sorted(choice_values.items()) if v])

            text_content = f"""**Problem Statement:**
{problem_text}

**Multiple Choice Options:**
{choices_text}

**Correct Answer Letter:** {correct_answer}

**LLM's Response:**
{llm_response}

**Extracted Answer:** {extracted_answer if extracted_answer else "None"}

**Answer Type:** {answer_type}

Please evaluate whether the LLM's answer is correct. The answer is correct if it matches either:
1. The correct letter ({correct_answer}), OR
2. The actual value of the correct choice (accept formatting variations)"""
        else:
            # For numeric or when choices not available
            text_content = f"""**Problem Statement:**
{problem_text}

**Correct Answer:** {correct_answer}

**LLM's Response:**
{llm_response}

**Extracted Answer:** {extracted_answer if extracted_answer else "None"}

**Answer Type:** {answer_type}

Please evaluate whether the LLM's answer is correct. Consider both the extracted answer and the full response context."""

        # If no image, return text only
        if not image_path:
            return text_content

        # If image exists, create multimodal content
        image_content = ImageHandler.create_image_content(image_path)
        if not image_content:
            return text_content

        # Return multimodal content
        return [
            {
                "type": "text",
                "text": text_content
            },
            image_content
        ]


class LLMJudge:
    """Main class for LLM-as-a-Judge evaluation."""

    def __init__(self, config: Dict, judge_model: str = "gpt-5", judge_cache_dir: str = "cache_judge"):
        """
        Initialize LLM Judge.

        Args:
            config: Configuration dictionary
            judge_model: Judge model identifier. "gpt-5" uses OpenAI directly;
                         anything else (e.g. "google/gemini-3-flash-preview") uses OpenRouter.
            judge_cache_dir: Directory to store judge results
        """
        self.config = config
        self.judge_model = judge_model
        self.use_openrouter = (judge_model != "gpt-5")

        if self.use_openrouter:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            # Use OpenAI client pointed at OpenRouter
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
            )
            self.async_client = None  # Use aiohttp for async OpenRouter
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=self.openai_api_key)
            self.async_client = AsyncOpenAI(api_key=self.openai_api_key)

        self.cache_dir = Path(config['cache_dir'])
        self.judge_cache_dir = Path(judge_cache_dir)
        self.image_dir = Path(config['image_dir'])

        self.temperature = 1.0
        self.max_tokens = config.get('max_tokens', 25000)
        self.max_retries = config.get('max_retries', 3)
        self.max_parallel_requests = config.get('max_parallel_requests', 10)

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse a JSON response from an OpenRouter judge model.

        Args:
            content: Raw response content string

        Returns:
            Dictionary with is_correct and explanation, or None if parsing fails
        """
        try:
            # Try direct JSON parse
            data = json.loads(content)
            return {
                "is_correct": bool(data["is_correct"]),
                "explanation": str(data.get("explanation", ""))
            }
        except (json.JSONDecodeError, KeyError):
            pass

        # Try extracting JSON from markdown code block
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return {
                    "is_correct": bool(data["is_correct"]),
                    "explanation": str(data.get("explanation", ""))
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # Try finding any JSON object in the response
        match = re.search(r'\{[^{}]*"is_correct"[^{}]*\}', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {
                    "is_correct": bool(data["is_correct"]),
                    "explanation": str(data.get("explanation", ""))
                }
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def query_judge(
        self,
        problem_text: str,
        correct_answer: str,
        llm_response: str,
        answer_type: str,
        extracted_answer: Optional[str] = None,
        image_path: Optional[str] = None,
        choice_values: Optional[Dict[str, str]] = None
    ) -> Optional[Dict]:
        """
        Query judge model. Uses structured outputs for GPT-5, JSON mode for OpenRouter.

        Args:
            problem_text: The problem statement
            correct_answer: The correct answer
            llm_response: The LLM's response
            answer_type: Type of answer
            extracted_answer: Extracted answer from LLM
            image_path: Path to image if present
            choice_values: Dictionary mapping choice letters to values (for multiple choice)

        Returns:
            Dictionary with judgment result or None if failed
        """
        # Build messages
        system_prompt = JudgePromptBuilder.build_system_prompt(json_mode=self.use_openrouter)
        user_content = JudgePromptBuilder.build_user_prompt(
            problem_text, correct_answer, llm_response,
            answer_type, extracted_answer, image_path, choice_values
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        for attempt in range(self.max_retries):
            try:
                if self.use_openrouter:
                    # OpenRouter path: JSON mode
                    completion = self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_headers={
                            "HTTP-Referer": "https://github.com/icelandic-math-eval",
                            "X-Title": "Icelandic Math Evaluation"
                        }
                    )
                    content = completion.choices[0].message.content
                    parsed = self._parse_json_response(content)
                    if parsed:
                        return {
                            "is_correct": parsed["is_correct"],
                            "explanation": parsed["explanation"],
                            "model": completion.model,
                            "judge_model": self.judge_model,
                            "usage": {
                                "prompt_tokens": completion.usage.prompt_tokens,
                                "completion_tokens": completion.usage.completion_tokens,
                                "total_tokens": completion.usage.total_tokens
                            },
                            "finish_reason": completion.choices[0].finish_reason
                        }
                    else:
                        print(f"Failed to parse JSON from judge response (attempt {attempt + 1})")
                else:
                    # GPT-5 path: structured outputs via Pydantic
                    completion = self.client.beta.chat.completions.parse(
                        model="gpt-5",
                        messages=messages,
                        response_format=JudgmentResult,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens
                    )
                    message = completion.choices[0].message
                    if message.parsed:
                        return {
                            "is_correct": message.parsed.is_correct,
                            "explanation": message.parsed.explanation,
                            "model": completion.model,
                            "judge_model": "gpt-5",
                            "usage": {
                                "prompt_tokens": completion.usage.prompt_tokens,
                                "completion_tokens": completion.usage.completion_tokens,
                                "total_tokens": completion.usage.total_tokens
                            },
                            "finish_reason": completion.choices[0].finish_reason
                        }
                    elif message.refusal:
                        print(f"Judge refused to evaluate (attempt {attempt + 1}): {message.refusal}")
                        return None

            except Exception as e:
                print(f"Error querying judge (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None

        return None

    async def _query_openrouter_async(
        self,
        messages: List[Dict],
        session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Query OpenRouter judge model asynchronously via aiohttp.

        Args:
            messages: Chat messages
            session: aiohttp session for connection pooling

        Returns:
            Dictionary with judgment result or None if failed
        """
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/icelandic-math-eval",
            "X-Title": "Icelandic Math Evaluation",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.judge_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(0.5 * (2 ** attempt))

                async with session.post(
                    url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=360)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = self._parse_json_response(content)
                        if parsed:
                            return {
                                "is_correct": parsed["is_correct"],
                                "explanation": parsed["explanation"],
                                "model": data.get("model", self.judge_model),
                                "judge_model": self.judge_model,
                                "usage": {
                                    "prompt_tokens": data["usage"]["prompt_tokens"],
                                    "completion_tokens": data["usage"]["completion_tokens"],
                                    "total_tokens": data["usage"]["total_tokens"]
                                },
                                "finish_reason": data["choices"][0].get("finish_reason")
                            }
                    else:
                        error_text = await response.text()
                        if attempt == self.max_retries - 1:
                            print(f"OpenRouter HTTP {response.status}: {error_text[:200]}")

            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return None

        return None

    async def query_judge_async(
        self,
        problem_text: str,
        correct_answer: str,
        llm_response: str,
        answer_type: str,
        extracted_answer: Optional[str] = None,
        image_path: Optional[str] = None,
        choice_values: Optional[Dict[str, str]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[Dict]:
        """
        Query judge model asynchronously. Uses structured outputs for GPT-5,
        aiohttp+JSON mode for OpenRouter models.

        Args:
            problem_text: The problem statement
            correct_answer: The correct answer
            llm_response: The LLM's response
            answer_type: Type of answer
            extracted_answer: Extracted answer from LLM
            image_path: Path to image if present
            choice_values: Dictionary mapping choice letters to values (for multiple choice)
            session: Optional aiohttp session (used for OpenRouter)

        Returns:
            Dictionary with judgment result or None if failed
        """
        # Build messages
        system_prompt = JudgePromptBuilder.build_system_prompt(json_mode=self.use_openrouter)
        user_content = JudgePromptBuilder.build_user_prompt(
            problem_text, correct_answer, llm_response,
            answer_type, extracted_answer, image_path, choice_values
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        if self.use_openrouter:
            # OpenRouter async path via aiohttp
            return await self._query_openrouter_async(messages, session)
        else:
            # GPT-5 path: structured outputs via AsyncOpenAI
            for attempt in range(self.max_retries):
                try:
                    completion = await self.async_client.beta.chat.completions.parse(
                        model="gpt-5",
                        messages=messages,
                        response_format=JudgmentResult,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens
                    )

                    message = completion.choices[0].message
                    if message.parsed:
                        return {
                            "is_correct": message.parsed.is_correct,
                            "explanation": message.parsed.explanation,
                            "model": completion.model,
                            "judge_model": "gpt-5",
                            "usage": {
                                "prompt_tokens": completion.usage.prompt_tokens,
                                "completion_tokens": completion.usage.completion_tokens,
                                "total_tokens": completion.usage.total_tokens
                            },
                            "finish_reason": completion.choices[0].finish_reason
                        }
                    elif message.refusal:
                        return None

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return None
                    await asyncio.sleep(0.5 * (2 ** attempt))

            return None

    def get_judge_cache_path(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str
    ) -> Path:
        """
        Get path to judge cache file.

        Args:
            model_name: Name of the model
            evaluation_mode: Evaluation mode
            problem_id: Problem identifier

        Returns:
            Path to judge cache file
        """
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        cache_subdir = self.judge_cache_dir / safe_model_name / evaluation_mode
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{problem_id}.json"

    def is_already_judged(
        self,
        model_name: str,
        evaluation_mode: str,
        problem_id: str
    ) -> bool:
        """Check if a problem has already been judged."""
        judge_cache_path = self.get_judge_cache_path(model_name, evaluation_mode, problem_id)
        return judge_cache_path.exists()

    def save_judged_result(
        self,
        original_cache: Dict,
        judge_result: Dict,
        judge_prompt_summary: str
    ) -> bool:
        """
        Save judged result to cache_judge directory.

        Args:
            original_cache: Original cache data
            judge_result: Judge evaluation result
            judge_prompt_summary: Summary of judge prompt

        Returns:
            True if saved successfully
        """
        model_name = original_cache['model_name']
        evaluation_mode = original_cache['evaluation_mode']
        problem_id = original_cache['problem_id']

        judge_cache_path = self.get_judge_cache_path(model_name, evaluation_mode, problem_id)

        # Augment original cache with judge data
        augmented_data = original_cache.copy()
        augmented_data['judge_prompt'] = judge_prompt_summary
        augmented_data['judge_result'] = judge_result

        try:
            with open(judge_cache_path, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving judge cache: {e}")
            return False

    async def judge_single_cached_result_async(
        self,
        cache_data: Dict,
        semaphore: asyncio.Semaphore,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict:
        """
        Judge a single cached result asynchronously.

        Args:
            cache_data: The cached evaluation data
            semaphore: Semaphore to limit concurrent requests

        Returns:
            Dictionary with judgment status and results
        """
        model_name = cache_data.get('model_name')
        evaluation_mode = cache_data.get('evaluation_mode')
        problem_id = cache_data.get('problem_id')

        # Check if already judged
        if self.is_already_judged(model_name, evaluation_mode, problem_id):
            return {
                'status': 'already_judged',
                'disagreement': False,
                'model_name': model_name,
                'evaluation_mode': evaluation_mode,
                'problem_id': problem_id
            }

        # Extract data
        problem_data = cache_data.get('problem_data', {})
        llm_response_data = cache_data.get('llm_response', {})

        problem_text = problem_data.get('problem_text', '')
        correct_answer = cache_data.get('correct_answer', '')
        llm_response = llm_response_data.get('response', '')
        extracted_answer = cache_data.get('extracted_answer', '')
        answer_type = problem_data.get('answer_type', 'multiple_choice')
        original_is_correct = cache_data.get('is_correct')

        # Handle image if present
        image_path = None
        if problem_data.get('has_image') and problem_data.get('image'):
            image_rel_path = problem_data.get('image')
            potential_image_path = self.image_dir / Path(image_rel_path).name
            if potential_image_path.exists():
                image_path = str(potential_image_path)

        # Extract choice values if multiple choice
        choice_values = None
        if answer_type == "multiple_choice":
            choice_values = {
                "A": problem_data.get("choice_a"),
                "B": problem_data.get("choice_b"),
                "C": problem_data.get("choice_c"),
                "D": problem_data.get("choice_d")
            }
            choice_values = {k: v for k, v in choice_values.items() if v is not None}

        # Query judge with semaphore control
        async with semaphore:
            judge_result = await self.query_judge_async(
                problem_text,
                correct_answer,
                llm_response,
                answer_type,
                extracted_answer,
                image_path,
                choice_values,
                session=session
            )

        if judge_result:
            # Create prompt summary
            judge_prompt_summary = f"""SYSTEM: {JudgePromptBuilder.build_system_prompt()}

USER: Problem: {problem_text[:100]}...
Correct Answer: {correct_answer}
LLM Response: {llm_response[:100]}..."""

            # Save result
            if self.save_judged_result(cache_data, judge_result, judge_prompt_summary):
                judge_says_correct = judge_result['is_correct']
                disagreement = (original_is_correct is not None and
                              original_is_correct != judge_says_correct)

                return {
                    'status': 'judged',
                    'disagreement': disagreement,
                    'original_is_correct': original_is_correct,
                    'judge_is_correct': judge_says_correct,
                    'model_name': model_name,
                    'evaluation_mode': evaluation_mode,
                    'problem_id': problem_id
                }
            else:
                return {
                    'status': 'failed',
                    'disagreement': False,
                    'reason': 'save_failed',
                    'model_name': model_name,
                    'evaluation_mode': evaluation_mode,
                    'problem_id': problem_id
                }
        else:
            return {
                'status': 'failed',
                'disagreement': False,
                'reason': 'query_failed',
                'model_name': model_name,
                'evaluation_mode': evaluation_mode,
                'problem_id': problem_id
            }

    def load_cache_files(self) -> List[Tuple[Path, Dict]]:
        """
        Load all cache files from cache directory.

        Returns:
            List of tuples (cache_path, cache_data)
        """
        cache_files = []

        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cache_files.append((cache_file, data))
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")

        return cache_files

    async def judge_model_mode_async(
        self,
        cache_files: List[Tuple[Path, Dict]],
        model_name: str,
        evaluation_mode: str,
        semaphore: asyncio.Semaphore,
        position: int,
        session: Optional[aiohttp.ClientSession] = None
    ) -> Dict:
        """
        Judge all files for a specific model and evaluation mode with progress bar.

        Args:
            cache_files: List of (path, data) tuples for this model/mode
            model_name: Name of the model
            evaluation_mode: Evaluation mode
            semaphore: Shared semaphore for rate limiting
            position: Position for tqdm progress bar
            session: Optional aiohttp session (for OpenRouter)

        Returns:
            Dictionary with aggregated statistics
        """
        tasks = [
            self.judge_single_cached_result_async(data, semaphore, session=session)
            for path, data in cache_files
        ]

        results = []
        desc = f"{model_name.split('/')[-1]}/{evaluation_mode}"

        with tqdm(
            total=len(tasks),
            desc=desc,
            position=position,
            leave=True,
            ncols=100
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        # Aggregate statistics
        stats = {
            'model_name': model_name,
            'evaluation_mode': evaluation_mode,
            'total': len(results),
            'already_judged': sum(1 for r in results if r['status'] == 'already_judged'),
            'judged': sum(1 for r in results if r['status'] == 'judged'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'disagreements': sum(1 for r in results if r.get('disagreement', False))
        }

        return stats

    def judge_all_cached_results(self):
        """Judge all cached results with parallel processing and progress bars."""
        print(f"Judge model: {self.judge_model}")
        print(f"Judge cache dir: {self.judge_cache_dir}")
        print(f"Using {'OpenRouter' if self.use_openrouter else 'OpenAI direct'} API")
        print()
        print("Loading cache files...")
        cache_files = self.load_cache_files()

        # Filter to only complete models (not grok-4)
        cache_files = [
            (path, data) for path, data in cache_files
            if 'grok-4' not in data.get('model_name', '')
        ]

        # Group files by model and evaluation mode
        groups = defaultdict(list)
        for path, data in cache_files:
            key = (data.get('model_name'), data.get('evaluation_mode'))
            groups[key].append((path, data))

        total = len(cache_files)
        print(f"Found {total} cache files to process")
        print(f"Organized into {len(groups)} model/mode combinations")
        print(f"Max parallel requests: {self.max_parallel_requests}")
        print()

        use_openrouter = self.use_openrouter

        # Run async judgment with parallel processing
        async def run_all_groups():
            # Create shared semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.max_parallel_requests)

            # Create aiohttp session for OpenRouter (shared across all tasks)
            session = None
            if use_openrouter:
                session = aiohttp.ClientSession()

            try:
                # Create tasks for each model/mode combination
                tasks = []
                for position, ((model_name, eval_mode), files) in enumerate(sorted(groups.items())):
                    task = self.judge_model_mode_async(
                        files, model_name, eval_mode, semaphore, position, session=session
                    )
                    tasks.append(task)

                # Run all groups concurrently
                all_stats = await asyncio.gather(*tasks)
                return all_stats
            finally:
                if session:
                    await session.close()

        # Execute async tasks
        all_stats = asyncio.run(run_all_groups())

        # Print final statistics
        print("\n" + "="*60)
        print("JUDGING COMPLETE")
        print("="*60)

        total_already_judged = sum(s['already_judged'] for s in all_stats)
        total_judged = sum(s['judged'] for s in all_stats)
        total_failed = sum(s['failed'] for s in all_stats)
        total_disagreements = sum(s['disagreements'] for s in all_stats)

        print(f"Total cache files: {total}")
        print(f"Already judged: {total_already_judged}")
        print(f"Newly judged: {total_judged}")
        print(f"Failed: {total_failed}")
        print(f"Disagreements with original evaluation: {total_disagreements}")

        print("\nPer-model/mode breakdown:")
        for stat in sorted(all_stats, key=lambda x: (x['model_name'], x['evaluation_mode'])):
            model_short = stat['model_name'].split('/')[-1]
            print(f"  {model_short}/{stat['evaluation_mode']}:")
            print(f"    Judged: {stat['judged']}, Already: {stat['already_judged']}, "
                  f"Failed: {stat['failed']}, Disagreements: {stat['disagreements']}")

        print("="*60)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Judge model to use. 'gpt-5' uses OpenAI direct API; "
             "anything else (e.g. 'google/gemini-3-flash-preview', "
             "'anthropic/claude-sonnet-4.6') uses OpenRouter. Default: gpt-5"
    )
    parser.add_argument(
        "--judge-cache-dir",
        default=None,
        help="Directory to store judge results. Default: derived from judge model name"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Derive judge cache dir from model name if not explicitly provided
    if args.judge_cache_dir is None:
        if args.judge_model == "gpt-5":
            args.judge_cache_dir = "cache_judge"
        else:
            args.judge_cache_dir = f"cache_judge_{args.judge_model.split('/')[-1]}"

    print("LLM-as-a-Judge Evaluation")
    print("="*60)

    # Load environment variables
    load_dotenv()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        sys.exit(1)

    # Initialize judge
    try:
        judge = LLMJudge(
            config,
            judge_model=args.judge_model,
            judge_cache_dir=args.judge_cache_dir
        )
    except ValueError as e:
        print(f"Error: {e}")
        if args.judge_model == "gpt-5":
            print("\nPlease ensure OPENAI_API_KEY is set in your .env file")
        else:
            print("\nPlease ensure OPENROUTER_API_KEY is set in your .env file")
        sys.exit(1)

    # Run judgment on all cached results
    judge.judge_all_cached_results()

    print(f"\nJudgment complete! Results saved to {args.judge_cache_dir}/")


if __name__ == "__main__":
    main()
