#!/usr/bin/env python3
"""
Extract Hard Questions from Past Benchmark Failures

Scans all benchmark runs to extract questions that any RAG provider failed on.
Creates a curated "hard questions" dataset for targeted explainability testing.

Usage:
    docker compose run --rm simple-evals python scripts/extract_hard_questions.py
    docker compose run --rm simple-evals python scripts/extract_hard_questions.py --output hard_questions.csv
    docker compose run --rm simple-evals python scripts/extract_hard_questions.py --provider CustomGPT  # Only CustomGPT failures
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# RAG providers we care about (exclude vanilla baselines)
RAG_PROVIDERS = ["CustomGPT", "OpenAI_RAG", "Google_Gemini_RAG"]

# Map directory names to provider names
DIR_TO_PROVIDER = {
    "customgpt_penalty_analysis": "CustomGPT",
    "openai_rag_penalty_analysis": "OpenAI_RAG",
    "gemini_rag_penalty_analysis": "Google_Gemini_RAG",
    "google_gemini_rag_penalty_analysis": "Google_Gemini_RAG",
}


def load_verified_questions(verified_path: Path) -> Dict[str, dict]:
    """Load simpleqa_verified.csv into a lookup dict by question_id"""
    questions = {}
    with open(verified_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id") or row.get("id")
            if qid:
                questions[qid] = {
                    "question": row.get("problem", row.get("question", "")),
                    "target_answer": row.get("answer", row.get("target_answer", "")),
                    "metadata": row.get("metadata", ""),
                }
    return questions


def find_penalty_files(results_dir: Path) -> List[Tuple[Path, str]]:
    """Find all penalty analysis JSON files and their provider names"""
    files = []

    for run_dir in results_dir.glob("run_*"):
        if not run_dir.is_dir():
            continue

        for penalty_dir in run_dir.iterdir():
            if not penalty_dir.is_dir():
                continue

            dir_name = penalty_dir.name
            provider = DIR_TO_PROVIDER.get(dir_name)

            if not provider:
                continue

            # Find JSON files in this directory
            for json_file in penalty_dir.glob("*.json"):
                if "penalty_analysis" in json_file.name:
                    files.append((json_file, provider))

    return files


def extract_failures_from_file(json_path: Path, provider: str) -> List[dict]:
    """Extract failure records from a penalty analysis JSON file"""
    failures = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return failures

    # Handle different JSON structures
    penalty_cases = data.get("penalty_cases", data.get("failures", []))

    if not isinstance(penalty_cases, list):
        return failures

    run_id = data.get("metadata", {}).get("run_id", json_path.parent.parent.name)

    for case in penalty_cases:
        if not isinstance(case, dict):
            continue

        qid = case.get("question_id")
        if not qid:
            continue

        failures.append({
            "question_id": qid,
            "question": case.get("question", ""),
            "target_answer": case.get("target_answer", ""),
            "provider_answer": case.get(f"{provider.lower()}_answer", case.get("answer", "")),
            "grade": case.get(f"{provider.lower()}_grade", case.get("grade", "B")),
            "judge_reasoning": case.get("judge_reasoning", ""),
            "competitor_results": case.get("competitor_results", {}),
            "run_id": run_id,
            "provider": provider,
        })

    return failures


def analyze_failures(
    results_dir: Path,
    filter_provider: str = None
) -> Tuple[Dict[str, dict], Dict[str, Set[str]]]:
    """
    Analyze all penalty files and extract unique failures.

    Returns:
        - failures_by_question: Dict[question_id, {question details, failed_providers, runs}]
        - provider_failures: Dict[provider, Set[question_ids]]
    """
    penalty_files = find_penalty_files(results_dir)
    print(f"Found {len(penalty_files)} penalty analysis files")

    # Track failures by question
    failures_by_question: Dict[str, dict] = {}
    provider_failures: Dict[str, Set[str]] = defaultdict(set)

    for json_path, provider in penalty_files:
        if filter_provider and provider != filter_provider:
            continue

        failures = extract_failures_from_file(json_path, provider)

        for failure in failures:
            qid = failure["question_id"]
            provider_failures[provider].add(qid)

            if qid not in failures_by_question:
                failures_by_question[qid] = {
                    "question_id": qid,
                    "question": failure["question"],
                    "target_answer": failure["target_answer"],
                    "failed_providers": set(),
                    "runs": set(),
                    "sample_judge_reasoning": failure["judge_reasoning"],
                    "competitor_results": {},
                }

            failures_by_question[qid]["failed_providers"].add(provider)
            failures_by_question[qid]["runs"].add(failure["run_id"])

            # Merge competitor results
            for comp, result in failure.get("competitor_results", {}).items():
                if comp not in failures_by_question[qid]["competitor_results"]:
                    failures_by_question[qid]["competitor_results"][comp] = result

    return failures_by_question, dict(provider_failures)


def categorize_failures(
    failures_by_question: Dict[str, dict],
    provider_failures: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """
    Categorize failures into groups:
    - customgpt_only: Only CustomGPT failed
    - shared_with_openai: CustomGPT and OpenAI RAG both failed
    - all_rag_failed: All 3 RAG providers failed
    - openai_rag_only: Only OpenAI RAG failed
    - gemini_only: Only Gemini RAG failed
    """
    categories = {
        "customgpt_only": [],
        "shared_with_openai": [],
        "shared_with_gemini": [],
        "all_rag_failed": [],
        "openai_rag_only": [],
        "gemini_only": [],
        "other": [],
    }

    customgpt_fails = provider_failures.get("CustomGPT", set())
    openai_fails = provider_failures.get("OpenAI_RAG", set())
    gemini_fails = provider_failures.get("Google_Gemini_RAG", set())

    for qid, details in failures_by_question.items():
        providers = details["failed_providers"]

        in_customgpt = "CustomGPT" in providers
        in_openai = "OpenAI_RAG" in providers
        in_gemini = "Google_Gemini_RAG" in providers

        if in_customgpt and in_openai and in_gemini:
            categories["all_rag_failed"].append(qid)
        elif in_customgpt and in_openai:
            categories["shared_with_openai"].append(qid)
        elif in_customgpt and in_gemini:
            categories["shared_with_gemini"].append(qid)
        elif in_customgpt:
            categories["customgpt_only"].append(qid)
        elif in_openai:
            categories["openai_rag_only"].append(qid)
        elif in_gemini:
            categories["gemini_only"].append(qid)
        else:
            categories["other"].append(qid)

    return categories


def write_hard_questions_csv(
    failures_by_question: Dict[str, dict],
    output_path: Path,
    verified_questions: Dict[str, dict] = None
):
    """Write hard questions to CSV file"""

    # Sort by question_id
    sorted_qids = sorted(failures_by_question.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "question",
            "target_answer",
            "failed_providers",
            "failure_count",
            "runs_failed",
            "category",
        ])

        for qid in sorted_qids:
            details = failures_by_question[qid]

            # Determine category
            providers = details["failed_providers"]
            if len(providers) == 3:
                category = "all_rag_failed"
            elif "CustomGPT" in providers and "OpenAI_RAG" in providers:
                category = "shared_customgpt_openai"
            elif "CustomGPT" in providers:
                category = "customgpt_failure"
            elif "OpenAI_RAG" in providers:
                category = "openai_rag_failure"
            else:
                category = "other"

            # Use verified questions if available
            question = details["question"]
            target = details["target_answer"]
            if verified_questions and qid in verified_questions:
                question = verified_questions[qid]["question"] or question
                target = verified_questions[qid]["target_answer"] or target

            writer.writerow([
                qid,
                question,
                target,
                "|".join(sorted(providers)),
                len(providers),
                len(details["runs"]),
                category,
            ])

    print(f"Wrote {len(sorted_qids)} hard questions to {output_path}")


def write_question_ids_file(
    question_ids: List[str],
    output_path: Path
):
    """Write just question IDs (for --question-ids-file parameter)"""
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in sorted(question_ids):
            f.write(f"{qid}\n")

    print(f"Wrote {len(question_ids)} question IDs to {output_path}")


def print_summary(
    failures_by_question: Dict[str, dict],
    provider_failures: Dict[str, Set[str]],
    categories: Dict[str, List[str]]
):
    """Print analysis summary"""
    print("\n" + "=" * 60)
    print("HARD QUESTIONS ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal unique hard questions: {len(failures_by_question)}")

    print("\nFailures by Provider:")
    for provider, qids in sorted(provider_failures.items()):
        print(f"  {provider}: {len(qids)} failures")

    print("\nFailure Categories:")
    for category, qids in sorted(categories.items()):
        if qids:
            print(f"  {category}: {len(qids)} questions")
            if len(qids) <= 5:
                for qid in sorted(qids):
                    print(f"    - {qid}")

    # Show CustomGPT-specific insights
    customgpt_only = set(categories.get("customgpt_only", []))
    if customgpt_only:
        print(f"\nCustomGPT-Only Failures ({len(customgpt_only)} questions):")
        print("  These are opportunities where CustomGPT alone failed:")
        for qid in sorted(customgpt_only)[:10]:
            details = failures_by_question[qid]
            print(f"    {qid}: {details['question'][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hard questions from past benchmark failures"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark results (default: results)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simpleqa-verified/hard_questions.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--ids-output",
        type=Path,
        default=Path("simpleqa-verified/hard_question_ids.txt"),
        help="Output file for just question IDs"
    )
    parser.add_argument(
        "--verified-csv",
        type=Path,
        default=Path("simpleqa-verified/simpleqa_verified.csv"),
        help="Path to simpleqa_verified.csv for question lookup"
    )
    parser.add_argument(
        "--provider",
        choices=["CustomGPT", "OpenAI_RAG", "Google_Gemini_RAG"],
        help="Filter to only one provider's failures"
    )
    parser.add_argument(
        "--customgpt-only",
        action="store_true",
        help="Only include questions where CustomGPT failed"
    )

    args = parser.parse_args()

    # Load verified questions for reference
    verified_questions = {}
    if args.verified_csv.exists():
        verified_questions = load_verified_questions(args.verified_csv)
        print(f"Loaded {len(verified_questions)} verified questions")

    # Analyze all failures
    failures_by_question, provider_failures = analyze_failures(
        args.results_dir,
        filter_provider=args.provider
    )

    if not failures_by_question:
        print("No failures found in any benchmark runs")
        return

    # Filter to CustomGPT only if requested
    if args.customgpt_only:
        customgpt_qids = provider_failures.get("CustomGPT", set())
        failures_by_question = {
            qid: details
            for qid, details in failures_by_question.items()
            if qid in customgpt_qids
        }
        print(f"Filtered to {len(failures_by_question)} CustomGPT failures")

    # Categorize failures
    categories = categorize_failures(failures_by_question, provider_failures)

    # Print summary
    print_summary(failures_by_question, provider_failures, categories)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ids_output.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    write_hard_questions_csv(failures_by_question, args.output, verified_questions)
    write_question_ids_file(list(failures_by_question.keys()), args.ids_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
