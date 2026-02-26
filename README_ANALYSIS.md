# Judge Analysis

This document describes the judge analysis tool for evaluating LLM performance across multiple dimensions.

## Overview

`judge_analysis.py` analyzes all judged results in the `cache_judge/` directory and computes comprehensive performance metrics.

## Usage

```bash
python3 judge_analysis.py
```

This will:
1. Load all judge results from `cache_judge/`
2. Compute performance metrics across multiple dimensions
3. Display formatted tables in the console
4. Save a detailed JSON report to `judge_analysis_report.json`

## Analysis Dimensions

### 1. By Model
Performance comparison across all evaluated models:
- openai/gpt-5
- anthropic/claude-sonnet-4.5
- google/gemini-2.5-pro

**Metrics**: Total problems, correct answers, incorrect answers, accuracy percentage

### 2. By Evaluation Mode
Performance in different evaluation settings:
- **with_choices**: Problems presented with multiple choice options
- **without_choices**: Problems presented without options (free-form answer)

### 3. By Model and Mode
Cross-analysis showing how each model performs in each evaluation mode.

### 4. By Difficulty Level
Performance across difficulty levels 1-10, where:
- **Level 1-3**: Easier problems
- **Level 4-7**: Medium difficulty
- **Level 8-10**: Hardest problems

### 5. By Answer Type
Performance on different answer formats:
- **multiple_choice**: Problems with A/B/C/D options
- **numeric**: Problems requiring numerical answers

### 6. By Problem Type
Performance on different mathematical domains:
- **algebra** (algebra)
- **talnafræði** (number theory)
- **rúmfræði** (geometry)
- **fléttufræði** (combinatorics)

Note: Problems can have multiple types, so totals may exceed the total number of problems.

### 7. By Model and Level
Detailed breakdown showing how each model performs at each difficulty level (available in JSON report).

## Output Files

### Console Output
Formatted tables showing key metrics for each dimension.

### JSON Report (`judge_analysis_report.json`)
Comprehensive JSON file containing:
- All metrics from console output
- Additional cross-dimensional analysis
- Detailed model-by-level breakdown
- Machine-readable format for further analysis

## Example Output

```
================================================================================
JUDGE ANALYSIS REPORT
================================================================================

PERFORMANCE BY MODEL
====================
Model                        Total  Correct  Incorrect  Accuracy
----------------------------------------------------------------
anthropic/claude-sonnet-4.5  2054   1665     389        81.06%
google/gemini-2.5-pro        2054   1870     184        91.04%
openai/gpt-5                 2054   1902     152        92.60%

PERFORMANCE BY EVALUATION MODE
==============================
Mode             Total  Correct  Incorrect  Accuracy
----------------------------------------------------
with_choices     3081   2821     260        91.56%
without_choices  3081   2616     465        84.91%
```

## Key Findings (from sample data)

### Model Performance
1. **Best overall**: openai/gpt-5 (92.60%)
2. **Second**: google/gemini-2.5-pro (91.04%)
3. **Third**: anthropic/claude-sonnet-4.5 (81.06%)

### Evaluation Mode Impact
- **with_choices** performs better (91.56%) than **without_choices** (84.91%)
- This holds true across all models
- GPT-5 shows smallest gap: 96.01% vs 89.19%
- Claude shows largest gap: 85.20% vs 76.92%

### Difficulty Levels
- Levels 1-7 have >80% accuracy
- Level 8 drops to 60.42%
- Level 9 is hardest at 43.33%
- Very few Level 10 problems (only 6)

### Problem Types
1. **Algebra**: Highest accuracy (93.42%)
2. **Number theory** (talnafræði): 92.11%
3. **Geometry** (rúmfræði): 82.14%
4. **Combinatorics** (fléttufræði): Lowest at 80.06%

### Answer Type
- Multiple choice: 88.33%
- Numeric: 87.78%
- Very similar performance, slight advantage to multiple choice

## Integration with Other Tools

The JSON report can be used for:
- Further statistical analysis (Python, R, etc.)
- Visualization (matplotlib, plotly, etc.)
- Spreadsheet analysis (import JSON to Excel/Google Sheets)
- Automated reporting pipelines

## Requirements

- Python 3.8+
- Standard library only (no external dependencies)
- Requires `cache_judge/` directory with judge results
