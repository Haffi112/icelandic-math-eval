# Judge Update: Multiple Choice Value Matching

## Summary

Updated the LLM-as-a-judge evaluation system to accept **value-based matching** for multiple choice questions, in addition to letter matching.

## Problem Solved

**Before**: Multiple choice questions were only marked correct if the LLM provided the exact letter (A, B, C, or D).

**Issue**: Many LLMs (especially in "without_choices" mode) compute the correct numerical/text answer but don't format it as a letter choice. For example:
- Correct answer: "B" (which equals "2000")
- LLM response: "2000" (with correct reasoning)
- Original evaluation: ❌ INCORRECT (because it's not "B")

**After**: The judge now accepts both:
1. ✅ Letter match: "B"
2. ✅ Value match: "2000" (the value of choice B)

## Technical Changes

### 1. Updated System Prompt
- Now instructs judge to accept either letter OR value matches
- Allows formatting variations (e.g., "2000" ≈ "2000 kr." ≈ "$2000$")

### 2. Enhanced User Prompt
- For multiple choice: Now includes all choice options with their values
- Example format:
  ```
  Multiple Choice Options:
  A) 1500
  B) 2000
  C) 2500
  D) 3000
  
  Correct Answer Letter: B
  ```

### 3. Updated Function Signatures
- Added `choice_values` parameter to:
  - `build_user_prompt()`
  - `query_judge()`
- Extracts choice values from problem data in `judge_all_cached_results()`

### 4. Data Flow
```python
# Extract choices from problem data
choice_values = {
    "A": "1500",
    "B": "2000",  # <- Correct answer
    "C": "2500",
    "D": "3000"
}

# Pass to judge along with correct letter
judge.query_judge(
    ...,
    correct_answer="B",
    choice_values=choice_values
)
```

## Test Results

✅ **Verified working** on test case `keppni_1314_p007`:
- Correct answer: B (= "2000")
- LLM response: "2000"
- Old evaluation: INCORRECT
- **New evaluation: CORRECT** ✨

Judge explanation:
> "The LLM's final answer is 2000, which matches the value of the correct choice B. The reasoning provided is also consistent with the equal split calculation."

## Impact

This update will significantly improve evaluation accuracy for models that:
- Provide correct mathematical answers without letter formatting
- Are evaluated in "without_choices" mode (no multiple choice options shown)
- Include reasoning with final numerical answers

## Files Modified

1. `llm_judge.py`:
   - `JudgePromptBuilder.build_system_prompt()` - Updated instructions
   - `JudgePromptBuilder.build_user_prompt()` - Added choice display
   - `LLMJudge.query_judge()` - Added choice_values parameter
   - `LLMJudge.judge_all_cached_results()` - Extract and pass choices

2. `README_JUDGE.md`:
   - Added documentation for dual-matching approach
   - Updated user prompt description
   - Added multiple choice evaluation logic section

3. `test_judge_update.py` (new):
   - Test script to verify value-based matching works
