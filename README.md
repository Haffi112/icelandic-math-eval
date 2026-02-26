# Icelandic Math LLM Evaluation

A benchmark for evaluating large language models on Icelandic mathematics competition problems from STAK (_Stærðfræðikeppni framhaldsskólanema_, 1984–2025). The dataset contains 41 competition files with 275 accompanying images.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{einarsson2026icelandic,
    title = {{Icelandic Math Eval}: A Competitive Mathematics Benchmark for Large Language Models},
    author = {Einarsson, Hafsteinn and Haraldsson, J{\"o}kull Ari and Derayat, {\'I}var Armin and Lund, Sigr{\'u}n Helga and Magn{\'u}sson, Benedikt Steinar},
    booktitle = {Proceedings of the 2026 Language Resources and Evaluation Conference (LREC 2026)},
    month = {May},
    year = {2026},
    address = {Palma, Mallorca, Spain},
    publisher = {European Language Resources Association (ELRA)}
}
```

## Features

- **Multi-provider support** — OpenAI Direct API and OpenRouter API with automatic routing
- **Two evaluation modes** — with choices (multiple choice) and without choices (open-ended)
- **Vision model support** — problems with images are automatically encoded and sent to vision-capable models
- **LLM judge system** — GPT-5-based judge with structured outputs for grading open-ended responses
- **Analysis tools** — breakdowns by model, evaluation mode, difficulty level, problem type, and time period
- **Caching** — avoids duplicate evaluations across runs
- **Parallel evaluation** — configurable concurrent API requests with progress tracking

## Setup

1. **Create virtual environment and install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure API keys:**
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
You need at least one of:
- `OPENAI_API_KEY` — from https://platform.openai.com/api-keys
- `OPENROUTER_API_KEY` — from https://openrouter.ai/keys

3. **Configure evaluation settings** in `config.yaml`:
- Evaluation mode (`with_choices` or `without_choices`)
- Which models to evaluate
- API settings (rate limits, retries, parallel requests)

## Usage

### 1. Run model evaluations

```bash
python main.py
```

Loads problems from `STAK-daemi/Keppnir`, checks the cache, evaluates each model in parallel, and saves results to `results/`.

### 2. Judge cached results with LLM judge

```bash
python llm_judge.py
```

Uses GPT-5 as a judge to grade cached model responses with structured outputs. Particularly useful for the `without_choices` mode where answers are open-ended.

### 3. Analyze and report

```bash
python judge_analysis.py
python judge_analysis_by_time.py
```

Generates accuracy breakdowns by model, evaluation mode, difficulty level, problem type, and competition year.

## Model IDs

The system routes to the correct API based on the model prefix:

**OpenAI Direct API** — use `openai/` prefix:
- `openai/gpt-5`

**OpenRouter API** — all other model IDs (see https://openrouter.ai/models):
- `anthropic/claude-sonnet-4.5`
- `google/gemini-2.5-pro`

## Project Structure

```
icelandic-math-eval/
├── main.py                    # Run model evaluations
├── llm_judge.py               # LLM judge (GPT-5, structured outputs)
├── judge_analysis.py          # Analysis by model, mode, difficulty, type
├── judge_analysis_by_time.py  # Analysis by competition year
├── config.yaml                # Evaluation configuration
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── JUDGE_UPDATE_SUMMARY.md    # Judge system documentation
├── src/
│   ├── data_loader.py         # Load problems from JSON
│   ├── image_handler.py       # Encode images for vision models
│   ├── prompt_builder.py      # Build Icelandic prompts
│   ├── model_client.py        # Unified API client (OpenAI + OpenRouter)
│   ├── cache_manager.py       # Cache evaluation results
│   └── evaluator.py           # Main evaluation logic
├── cache/                     # Cached results (auto-generated)
├── results/                   # Output results (auto-generated)
└── STAK-daemi/
    ├── Keppnir/               # 41 competition JSON files (1984–2025)
    └── Myndir/                # 275 problem images
```

## Dataset

The benchmark is built from **STAK** (_Stærðfræðikeppni framhaldsskólanema_), Iceland's national secondary school mathematics competition. The dataset spans 41 competitions from 1984 to 2025.

Each problem includes:
- `problem_text` — the question in Icelandic
- `choice_a` through `choice_d` — multiple choice options
- `correct_answer` — the correct option (A/B/C/D)
- `problem_types` — categories (e.g. algebra, geometry, combinatorics)
- `level` — difficulty level (1–5)
- `source_info` — competition year and tier
