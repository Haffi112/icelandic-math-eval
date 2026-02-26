# LLM-as-a-Judge Evaluation

This document describes the LLM-as-a-Judge evaluation system for the Icelandic Math Evaluation project.

## Overview

The `llm_judge.py` script uses GPT-5 to evaluate whether LLM responses to mathematical problems are correct. It provides an independent assessment of the accuracy of model responses.

## Features

- **Structured Outputs**: Uses OpenAI's structured outputs API with Pydantic models for reliable binary judgments
- **Multimodal Support**: Handles problems with associated images
- **Smart Caching**: Only evaluates problems that haven't been judged yet
- **Progress Tracking**: Shows real-time progress and statistics
- **Disagreement Detection**: Flags cases where the judge disagrees with the original evaluation

## Usage

### Basic Usage

```bash
python3 llm_judge.py
```

This will:
1. Load all cached evaluation results from `cache/`
2. For each result not yet judged:
   - Query GPT-5 with the problem, correct answer, and LLM response
   - Get a binary judgment (correct/incorrect) with explanation
   - Save the augmented result to `cache_judge/`
3. Display statistics and disagreements

### Requirements

- OpenAI API key set in `.env` file:
  ```
  OPENAI_API_KEY=your_key_here
  ```
- Python dependencies:
  - `openai` (with Pydantic support)
  - `pyyaml`
  - `python-dotenv`
  - `pillow`

## Output Structure

### Cache Directory

The judge creates a `cache_judge/` directory that mirrors the structure of `cache/`:

```
cache_judge/
├── openai_gpt-5/
│   └── without_choices/
│       └── keppni_1314_p007.json
├── anthropic_claude-sonnet-4.5/
│   └── without_choices/
│       └── keppni_1314_p007.json
└── google_gemini-2.5-pro/
    └── without_choices/
        └── keppni_1314_p007.json
```

### Augmented Cache Format

Each judged cache file contains:
- **All original fields** from the cache
- **`judge_prompt`**: The prompt sent to the judge (system + user messages)
- **`judge_result`**: The structured judgment result with:
  - `is_correct`: Boolean indicating if the answer is correct
  - `explanation`: Detailed explanation of the judgment
  - `model`: Model used for judgment (e.g., "gpt-5-2025-08-07")
  - `usage`: Token usage statistics
  - `finish_reason`: Completion finish reason

### Example Output

```json
{
  "model_name": "google/gemini-2.5-pro",
  "evaluation_mode": "without_choices",
  "problem_id": "keppni_1314_p007",
  "timestamp": "2025-10-01T14:02:00.540485",
  "problem_data": { ... },
  "llm_response": { ... },
  "extracted_answer": "1.",
  "correct_answer": "B",
  "is_correct": false,
  "judge_prompt": "SYSTEM: You are an expert mathematics evaluator...",
  "judge_result": {
    "is_correct": false,
    "explanation": "The problem is multiple-choice with correct answer B. The LLM provided a numeric answer (2000)...",
    "model": "gpt-5-2025-08-07",
    "usage": {
      "prompt_tokens": 913,
      "completion_tokens": 525,
      "total_tokens": 1438
    },
    "finish_reason": "stop"
  }
}
```

## Judge Prompt Design

### System Prompt (English)

The judge receives instructions in English:
- Evaluate mathematical correctness
- Consider answer type (multiple choice vs. numeric)
- **For multiple choice**: Accept answers that match either the correct letter OR the actual value of the correct choice
- Focus on final answer, not reasoning process
- Be flexible with formatting variations (e.g., "2000" matches "2000 kr.", "$2000$")
- Be objective and fair

### User Prompt

The judge receives:
- Problem statement (in Icelandic)
- **For multiple choice**: All choice options (A, B, C, D) with their values
- Correct answer (letter for multiple choice)
- LLM's full response
- Extracted answer
- Answer type
- Image (if applicable)

The judge explicitly knows content is in Icelandic but evaluates objectively.

### Multiple Choice Evaluation Logic

For multiple choice questions, the judge now uses a **dual-matching approach**:

1. **Letter Match**: Accept if LLM provides the correct letter (e.g., "B")
2. **Value Match**: Accept if LLM provides the value corresponding to the correct choice (e.g., "2000" when B = "2000")

This allows the judge to correctly evaluate cases where:
- LLM computed the correct answer but didn't format it as a letter choice
- LLM provided reasoning and final numerical answer instead of selecting a letter
- Minor formatting variations exist (e.g., "2000" vs "2000 kr." vs "$2000$")

## Configuration

The judge uses settings from `config.yaml`:
- `max_tokens`: Maximum tokens for judge response (default: 25000)
- `max_retries`: Number of retry attempts (default: 3)
- `image_dir`: Directory for problem images

Temperature is fixed at 1.0 for consistency with other evaluations.

## Statistics

The judge tracks:
- **Total cache files**: Number of problems to evaluate
- **Already judged**: Problems previously judged (skipped)
- **Newly judged**: Problems judged in this run
- **Failed**: Problems that failed to be judged
- **Disagreements**: Cases where judge disagrees with original `is_correct` field

## Notes

- The judge skips incomplete evaluations (e.g., grok-4 results)
- Judgments are cached to avoid re-evaluation
- The judge can handle both multiple choice and numeric answers
- Multimodal evaluation is supported for problems with images
