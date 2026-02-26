# Icelandic Math LLM Evaluation

System for evaluating large language models on Icelandic math competition problems.

## Features

- **Multi-provider support**: OpenAI Direct API + OpenRouter API
- **Automatic prompt caching**: 75% cost reduction for OpenAI models
- Two evaluation modes:
  - **With choices**: LLM receives problem + multiple choice options
  - **Without choices**: LLM receives only problem statement
- Supports problems with images (vision models)
- Caching system to avoid duplicate evaluations
- Progress tracking with tqdm
- Comprehensive statistics by problem type and difficulty level

## Setup

1. **Create virtual environment and install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure API keys:**
Create a `.env` file in the project root:
```bash
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional, for direct OpenAI access
```

You need at least one API key. Use both for maximum flexibility:
- `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys
- `OPENROUTER_API_KEY`: Get from https://openrouter.ai/keys

3. **Configure evaluation settings:**
Edit `config.yaml` to:
- Choose evaluation mode (`with_choices` or `without_choices`)
- Select which models to evaluate
- Adjust API settings (rate limits, retries, parallel requests, etc.)
- Set `max_parallel_requests` to control concurrency (default: 10)

## Usage

Run the evaluation:
```bash
source venv/bin/activate
python main.py
```

The script will:
1. Load all problems from `./STAK-daemi/Keppnir`
2. Check cache for existing results
3. Evaluate each model sequentially with a dedicated progress bar
4. Within each model, evaluate problems in parallel (configurable concurrency)
5. Display statistics per model
6. Save detailed results to `./results/`

**Performance:** With `max_parallel_requests: 10`, evaluating 100 problems can be ~10x faster than sequential evaluation!

## Model IDs

The system automatically routes to the correct API based on the model prefix:

### OpenAI Direct API (recommended for OpenAI models)
Use `openai/` prefix to access OpenAI's API directly with automatic prompt caching:
- `openai/gpt-5`
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

**Benefits**: 75% cost reduction via automatic caching, better reliability, lower latency

### OpenRouter API
All other model IDs use OpenRouter. Find models at: https://openrouter.ai/models
- `anthropic/claude-sonnet-4.5`
- `anthropic/claude-opus-4.1`
- `google/gemini-2.5-pro`
- `x-ai/grok-4`

**See [OPENAI_INTEGRATION.md](OPENAI_INTEGRATION.md) for detailed documentation on the dual-API system.**

## Project Structure

```
icelandic_math_eval/
├── main.py                 # Main script
├── config.yaml             # Configuration
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
├── src/
│   ├── data_loader.py      # Load problems from JSON
│   ├── image_handler.py    # Handle images for vision models
│   ├── prompt_builder.py   # Build Icelandic prompts
│   ├── model_client.py     # Unified API client (OpenAI + OpenRouter)
│   ├── llm_client.py       # Legacy OpenRouter-only client (deprecated)
│   ├── cache_manager.py    # Cache evaluation results
│   └── evaluator.py        # Main evaluation logic
├── cache/                  # Cached results (auto-generated)
├── results/                # Final results (auto-generated)
└── STAK-daemi/
    ├── Keppnir/            # Math problems (JSON)
    └── Myndir/             # Problem images
```

## Cache Structure

Cached results are stored in:
```
cache/{model_name}/{evaluation_mode}/{problem_id}.json
```

Each cache file contains:
- Problem data and metadata
- Prompt used
- LLM response (raw)
- Extracted answer
- Correctness determination

## Output

Results are saved to `results/{model_name}_results.json` containing:
- Individual problem results
- Response statistics
- Token usage

Console output shows:
- Overall accuracy
- Accuracy by problem type
- Accuracy by difficulty level

## Evaluation Modes

### With Choices
- Problem text + options A, B, C, D presented
- LLM selects answer letter
- Suitable for multiple choice problems
- Prompt in Icelandic

### Without Choices
- Only problem text presented
- LLM generates numeric or algebraic answer
- More challenging mode
- Tests deeper understanding

## Performance & Parallelization

The evaluation system supports parallel API requests for faster execution:

- **Parallel requests**: Set `max_parallel_requests` in `config.yaml` (default: 10)
- **Per-model progress bars**: Each model shows its own progress: "Evaluating openai/gpt-5: 45/100"
- **Sequential models**: Models are evaluated one at a time to avoid overwhelming the API
- **Async implementation**: Uses `asyncio` and `aiohttp` for efficient concurrent requests

Example timing (100 problems):
- Sequential (1 request at a time): ~200 seconds
- Parallel (10 requests): ~20-30 seconds

**Tip:** Adjust `max_parallel_requests` based on your OpenRouter rate limits. Start with 5-10 for safety.

## Notes

- First run will be slow as it queries all models for all problems
- Subsequent runs use cache and only evaluate new problems
- Images are automatically encoded and sent to vision-capable models
- Rate limiting with exponential backoff is built in to avoid API throttling
- Cached results are reused even across different evaluation modes
