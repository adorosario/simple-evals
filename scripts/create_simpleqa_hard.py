#!/usr/bin/env python3
"""
SimpleQA-Hard Dataset Creator

Creates a curated subset of SimpleQA containing questions that challenged
SOTA RAG systems (OpenAI RAG or CustomGPT RAG).

A question is "hard" if either OpenAI_RAG OR CustomGPT_RAG failed on it
at least once across all benchmark runs.

Output:
- simpleqa_hard.csv: Clean subset in same format as original SimpleQA
- simpleqa_hard_forensics.json: Full failure details for debugging
- simpleqa_hard_methodology.md: Documentation
- simpleqa_hard_summary.json: Statistics
"""

import json
import csv
import os
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Set


# Configuration - Docker container paths (mounted at /app)
RESULTS_DIR = Path("/app/results")
SOURCE_CSV = Path("/app/build-rag/simple_qa_test_set.csv")
OUTPUT_DIR = Path("/app/build-rag/simpleqa_hard")

# RAG providers to track (exclude OpenAI_Vanilla)
RAG_PROVIDERS = {"OpenAI_RAG", "CustomGPT_RAG", "Google_Gemini_RAG"}


def find_judge_evaluation_files() -> List[Path]:
    """Find all judge_evaluations.jsonl files in results directory."""
    files = list(RESULTS_DIR.glob("**/judge_evaluations.jsonl"))
    print(f"Found {len(files)} judge_evaluations.jsonl files")
    return sorted(files)


def parse_judge_evaluations(filepath: Path) -> List[Dict[str, Any]]:
    """Parse a judge_evaluations.jsonl file."""
    evaluations = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    evaluations.append(data)
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error in {filepath}:{line_num}: {e}")
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
    return evaluations


def extract_failure_info(evaluation: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Extract failure information from a judge evaluation.
    Returns None if not a failure or not from a RAG provider.
    """
    # Get provider name
    metadata = evaluation.get("metadata", {})
    provider_name = metadata.get("real_provider_name", "")

    # Skip if not a RAG provider
    if provider_name not in RAG_PROVIDERS:
        return None

    # Get grade - check both formats
    grades = evaluation.get("grades", {})

    # Find the grade for this evaluation
    grade = None
    for key, value in grades.items():
        grade = value
        break

    # Only interested in failures
    if grade != "INCORRECT":
        return None

    # Get provider response (wrong answer)
    provider_responses = evaluation.get("provider_responses", {})
    wrong_answer = ""
    for key, value in provider_responses.items():
        wrong_answer = value
        break

    # Get judge reasoning
    judge_info = evaluation.get("judge", {})
    if isinstance(judge_info, dict):
        reasoning = evaluation.get("reasoning", judge_info.get("reasoning", ""))
    else:
        reasoning = evaluation.get("reasoning", "")

    return {
        "provider": provider_name,
        "question_id": evaluation.get("question_id", ""),
        "question": evaluation.get("question", ""),
        "target_answer": evaluation.get("target_answer", ""),
        "wrong_answer": wrong_answer,
        "judge_reasoning": reasoning,
        "run_id": evaluation.get("run_id", ""),
        "timestamp": evaluation.get("timestamp", "")
    }


def load_source_dataset() -> Dict[str, Dict[str, Any]]:
    """Load the source SimpleQA dataset indexed by question text."""
    dataset = {}

    with open(SOURCE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Parse metadata
            metadata_str = row.get('metadata', '{}')
            try:
                metadata = ast.literal_eval(metadata_str)
            except:
                metadata = {}

            question = row.get('problem', '')
            answer = row.get('answer', '')

            # Index by question text (normalized)
            key = question.strip()
            dataset[key] = {
                "index": i,
                "question_id": f"simpleqa_{i:04d}",
                "metadata": metadata,
                "problem": question,
                "answer": answer,
                "raw_metadata_str": metadata_str
            }

    print(f"Loaded {len(dataset)} questions from source dataset")
    return dataset


def analyze_all_runs() -> Dict[str, Dict[str, Any]]:
    """
    Analyze all benchmark runs and build failure registry.

    Returns:
        Dictionary mapping question_id -> failure info
    """
    # Track failures per question
    failure_registry: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "question": "",
        "target_answer": "",
        "failures": {
            "OpenAI_RAG": [],
            "CustomGPT_RAG": []
        },
        "seen_by": set()
    })

    # Track overall stats
    stats = {
        "total_evaluations": 0,
        "rag_evaluations": 0,
        "failures": 0,
        "runs_processed": 0
    }

    judge_files = find_judge_evaluation_files()

    for filepath in judge_files:
        run_id = filepath.parent.name
        evaluations = parse_judge_evaluations(filepath)
        stats["runs_processed"] += 1

        for eval_data in evaluations:
            stats["total_evaluations"] += 1

            # Get provider
            metadata = eval_data.get("metadata", {})
            provider = metadata.get("real_provider_name", "")

            if provider in RAG_PROVIDERS:
                stats["rag_evaluations"] += 1

                question_id = eval_data.get("question_id", "")
                question = eval_data.get("question", "")
                target_answer = eval_data.get("target_answer", "")

                # Update registry
                if question_id:
                    failure_registry[question_id]["question"] = question
                    failure_registry[question_id]["target_answer"] = target_answer
                    failure_registry[question_id]["seen_by"].add(provider)

                # Check for failure
                failure_info = extract_failure_info(eval_data)
                if failure_info:
                    stats["failures"] += 1
                    failure_registry[question_id]["failures"][provider].append({
                        "run_id": failure_info["run_id"],
                        "wrong_answer": failure_info["wrong_answer"],
                        "judge_reasoning": failure_info["judge_reasoning"],
                        "timestamp": failure_info["timestamp"]
                    })

    print(f"\nAnalysis complete:")
    print(f"  Runs processed: {stats['runs_processed']}")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  RAG evaluations: {stats['rag_evaluations']}")
    print(f"  Failures found: {stats['failures']}")
    print(f"  Unique questions seen: {len(failure_registry)}")

    return dict(failure_registry), stats


def identify_hard_questions(failure_registry: Dict[str, Dict[str, Any]]) -> Set[str]:
    """
    Identify hard questions based on failure criteria.

    A question is hard if either OpenAI_RAG OR CustomGPT_RAG failed at least once.
    """
    hard_questions = set()

    for question_id, data in failure_registry.items():
        failures = data["failures"]
        openai_failures = len(failures.get("OpenAI_RAG", []))
        customgpt_failures = len(failures.get("CustomGPT_RAG", []))

        if openai_failures > 0 or customgpt_failures > 0:
            hard_questions.add(question_id)

    return hard_questions


def create_outputs(
    failure_registry: Dict[str, Dict[str, Any]],
    hard_questions: Set[str],
    source_dataset: Dict[str, Dict[str, Any]],
    stats: Dict[str, int]
):
    """Create all output files."""

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build question_id to source data mapping
    question_id_to_source = {}
    for question_text, data in source_dataset.items():
        qid = data["question_id"]
        question_id_to_source[qid] = data

    # Also try to match by question text for questions that might have different IDs
    question_text_to_source = source_dataset

    # 1. Create simpleqa_hard.csv (clean format)
    create_clean_csv(failure_registry, hard_questions, question_id_to_source, question_text_to_source)

    # 2. Create simpleqa_hard_forensics.json
    create_forensics_json(failure_registry, hard_questions, question_id_to_source)

    # 3. Create simpleqa_hard_summary.json
    create_summary_json(failure_registry, hard_questions, stats)

    # 4. Create simpleqa_hard_methodology.md
    create_methodology_doc(failure_registry, hard_questions, stats)


def create_clean_csv(
    failure_registry: Dict[str, Dict[str, Any]],
    hard_questions: Set[str],
    question_id_to_source: Dict[str, Dict[str, Any]],
    question_text_to_source: Dict[str, Dict[str, Any]]
):
    """Create clean CSV in same format as original SimpleQA."""

    output_path = OUTPUT_DIR / "simpleqa_hard.csv"
    rows_written = 0

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metadata', 'problem', 'answer'])

        for question_id in sorted(hard_questions):
            data = failure_registry.get(question_id, {})
            question_text = data.get("question", "")
            target_answer = data.get("target_answer", "")

            # Try to get source metadata
            source_data = question_id_to_source.get(question_id)
            if not source_data and question_text:
                source_data = question_text_to_source.get(question_text.strip())

            if source_data:
                # Use original metadata string
                metadata_str = source_data.get("raw_metadata_str", "{}")
                problem = source_data.get("problem", question_text)
                answer = source_data.get("answer", target_answer)
            else:
                # Fallback: create minimal metadata
                metadata_str = "{}"
                problem = question_text
                answer = target_answer

            writer.writerow([metadata_str, problem, answer])
            rows_written += 1

    print(f"\nCreated {output_path}")
    print(f"  Rows written: {rows_written}")


def create_forensics_json(
    failure_registry: Dict[str, Dict[str, Any]],
    hard_questions: Set[str],
    question_id_to_source: Dict[str, Dict[str, Any]]
):
    """Create forensics JSON with full failure details."""

    forensics = {}

    for question_id in sorted(hard_questions):
        data = failure_registry.get(question_id, {})

        # Get source metadata
        source_data = question_id_to_source.get(question_id, {})
        metadata = source_data.get("metadata", {})

        openai_failures = data["failures"].get("OpenAI_RAG", [])
        customgpt_failures = data["failures"].get("CustomGPT_RAG", [])

        failed_by = []
        if openai_failures:
            failed_by.append("OpenAI_RAG")
        if customgpt_failures:
            failed_by.append("CustomGPT_RAG")

        forensics[question_id] = {
            "question": data.get("question", ""),
            "target_answer": data.get("target_answer", ""),
            "metadata": metadata,
            "failures": {
                "OpenAI_RAG": openai_failures,
                "CustomGPT_RAG": customgpt_failures
            },
            "failure_summary": {
                "openai_rag_failures": len(openai_failures),
                "customgpt_rag_failures": len(customgpt_failures),
                "total_failures": len(openai_failures) + len(customgpt_failures),
                "failed_by": failed_by
            }
        }

    output_path = OUTPUT_DIR / "simpleqa_hard_forensics.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(forensics, f, indent=2, ensure_ascii=False)

    print(f"Created {output_path}")


def create_summary_json(
    failure_registry: Dict[str, Dict[str, Any]],
    hard_questions: Set[str],
    stats: Dict[str, int]
):
    """Create summary statistics JSON."""

    # Compute detailed stats
    only_openai_failed = 0
    only_customgpt_failed = 0
    both_failed = 0

    topic_breakdown = defaultdict(int)
    answer_type_breakdown = defaultdict(int)

    failure_frequency = defaultdict(int)  # number of failures -> count

    for question_id in hard_questions:
        data = failure_registry.get(question_id, {})
        openai_count = len(data["failures"].get("OpenAI_RAG", []))
        customgpt_count = len(data["failures"].get("CustomGPT_RAG", []))

        if openai_count > 0 and customgpt_count > 0:
            both_failed += 1
        elif openai_count > 0:
            only_openai_failed += 1
        elif customgpt_count > 0:
            only_customgpt_failed += 1

        total_failures = openai_count + customgpt_count
        failure_frequency[total_failures] += 1

    # Questions seen by providers
    seen_by_openai = sum(1 for d in failure_registry.values() if "OpenAI_RAG" in d.get("seen_by", set()))
    seen_by_customgpt = sum(1 for d in failure_registry.values() if "CustomGPT_RAG" in d.get("seen_by", set()))
    seen_by_both = sum(1 for d in failure_registry.values()
                       if "OpenAI_RAG" in d.get("seen_by", set()) and "CustomGPT_RAG" in d.get("seen_by", set()))

    summary = {
        "created_at": datetime.now().isoformat(),
        "source_dataset": str(SOURCE_CSV),
        "results_analyzed": str(RESULTS_DIR),
        "overview": {
            "total_simpleqa_questions": 4332,
            "questions_evaluated": len(failure_registry),
            "hard_questions": len(hard_questions),
            "hard_percentage": round(len(hard_questions) / len(failure_registry) * 100, 2) if failure_registry else 0
        },
        "coverage": {
            "questions_seen_by_openai_rag": seen_by_openai,
            "questions_seen_by_customgpt_rag": seen_by_customgpt,
            "questions_seen_by_both": seen_by_both,
            "runs_analyzed": stats.get("runs_processed", 0)
        },
        "failure_analysis": {
            "only_openai_rag_failed": only_openai_failed,
            "only_customgpt_rag_failed": only_customgpt_failed,
            "both_providers_failed": both_failed,
            "total_hard_questions": len(hard_questions)
        },
        "failure_frequency_distribution": dict(sorted(failure_frequency.items())),
        "methodology": {
            "definition": "Question is hard if OpenAI_RAG OR CustomGPT_RAG failed at least once",
            "excluded_providers": ["OpenAI_Vanilla"],
            "included_providers": list(RAG_PROVIDERS)
        }
    }

    output_path = OUTPUT_DIR / "simpleqa_hard_summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Created {output_path}")


def create_methodology_doc(
    failure_registry: Dict[str, Dict[str, Any]],
    hard_questions: Set[str],
    stats: Dict[str, int]
):
    """Create methodology documentation."""

    # Calculate some stats for the doc
    only_openai = sum(1 for qid in hard_questions
                      if len(failure_registry[qid]["failures"]["OpenAI_RAG"]) > 0
                      and len(failure_registry[qid]["failures"]["CustomGPT_RAG"]) == 0)
    only_customgpt = sum(1 for qid in hard_questions
                         if len(failure_registry[qid]["failures"]["CustomGPT_RAG"]) > 0
                         and len(failure_registry[qid]["failures"]["OpenAI_RAG"]) == 0)
    both = len(hard_questions) - only_openai - only_customgpt

    doc = f"""# SimpleQA-Hard: Methodology and Documentation

## Overview

**SimpleQA-Hard** is a curated subset of the SimpleQA benchmark containing questions that proved challenging for state-of-the-art RAG (Retrieval-Augmented Generation) systems.

- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Hard Questions**: {len(hard_questions)}
- **Source Dataset**: SimpleQA ({4332} questions)

## Selection Criteria

A question is included in SimpleQA-Hard if it meets the following criterion:

> **Definition**: A question is "hard" if **OpenAI_RAG** OR **CustomGPT_RAG** failed on it at least once across all benchmark runs.

### Rationale

1. **OpenAI_RAG** and **CustomGPT_RAG** represent state-of-the-art RAG implementations:
   - OpenAI RAG uses OpenAI's vector store file search
   - CustomGPT RAG uses CustomGPT's knowledge base

2. Both systems were provided with curated knowledge bases built from the source URLs in SimpleQA

3. If either of these sophisticated RAG systems fails on a question, it indicates genuine difficulty that goes beyond simple retrieval

### Excluded Providers

- **OpenAI_Vanilla** (baseline LLM without RAG) is excluded from the hardness calculation
- Rationale: OpenAI Vanilla failures reflect knowledge cutoff or training data gaps, not RAG-specific difficulty

## Data Sources

### Benchmark Runs Analyzed
- **Total Runs**: {stats.get('runs_processed', 0)}
- **Location**: `{RESULTS_DIR}`
- **File Pattern**: `run_*/judge_evaluations.jsonl`

### Source Dataset
- **File**: `{SOURCE_CSV}`
- **Total Questions**: 4,332

## Statistics

### Coverage
| Metric | Value |
|--------|-------|
| Questions evaluated by RAG providers | {len(failure_registry)} |
| Questions identified as hard | {len(hard_questions)} |
| Hard percentage | {round(len(hard_questions) / len(failure_registry) * 100, 2) if failure_registry else 0}% |

### Failure Breakdown
| Category | Count |
|----------|-------|
| Only OpenAI_RAG failed | {only_openai} |
| Only CustomGPT_RAG failed | {only_customgpt} |
| Both providers failed | {both} |
| **Total Hard Questions** | **{len(hard_questions)}** |

## Output Files

| File | Description |
|------|-------------|
| `simpleqa_hard.csv` | Clean subset in same format as original SimpleQA (drop-in replacement) |
| `simpleqa_hard_forensics.json` | Full failure details with wrong answers and judge reasoning |
| `simpleqa_hard_summary.json` | Statistical summary |
| `simpleqa_hard_methodology.md` | This documentation |

## Usage

### As SimpleQA Replacement
```python
import pandas as pd
df = pd.read_csv("simpleqa_hard.csv")
# Use exactly like original SimpleQA
```

### For Failure Analysis
```python
import json
with open("simpleqa_hard_forensics.json") as f:
    forensics = json.load(f)

# Example: Get all questions where both RAGs failed
both_failed = [qid for qid, data in forensics.items()
               if data["failure_summary"]["failed_by"] == ["OpenAI_RAG", "CustomGPT_RAG"]]
```

## Reproducibility

This dataset was created by:
1. Parsing {stats.get('runs_processed', 0)} benchmark runs from `{RESULTS_DIR}`
2. Extracting evaluations for OpenAI_RAG and CustomGPT_RAG providers
3. Identifying questions with at least one INCORRECT grade
4. Cross-referencing with source SimpleQA dataset for metadata

The creation script is located at: `/home/adorosario/simple-evals/scripts/create_simpleqa_hard.py`

## License

This dataset is derived from SimpleQA by OpenAI and inherits its license terms.
"""

    output_path = OUTPUT_DIR / "simpleqa_hard_methodology.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(f"Created {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SimpleQA-Hard Dataset Creator")
    print("=" * 60)

    # Load source dataset
    print("\n[1/4] Loading source SimpleQA dataset...")
    source_dataset = load_source_dataset()

    # Analyze all runs
    print("\n[2/4] Analyzing benchmark runs...")
    failure_registry, stats = analyze_all_runs()

    # Identify hard questions
    print("\n[3/4] Identifying hard questions...")
    hard_questions = identify_hard_questions(failure_registry)
    print(f"  Hard questions identified: {len(hard_questions)}")

    # Create outputs
    print("\n[4/4] Creating output files...")
    create_outputs(failure_registry, hard_questions, source_dataset, stats)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
