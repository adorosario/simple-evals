#!/usr/bin/env python3
"""
Create simpleqa_verified_hard.csv - a subset of questions where at least one
RAG provider failed (got INCORRECT) in runs after Dec 10th 2025.
"""
import json
import os
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration - use relative paths for Docker compatibility
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ORIGINAL_CSV = PROJECT_ROOT / "simpleqa-verified" / "simpleqa_verified.csv"
OUTPUT_CSV = PROJECT_ROOT / "simpleqa-verified" / "simpleqa_verified_hard.csv"
KB_MANIFEST = PROJECT_ROOT / "knowledge_base_verified" / "build_manifest.json"
KB_DIR = PROJECT_ROOT / "knowledge_base_verified"

# RAG providers to consider (exclude vanilla baseline)
RAG_PROVIDERS = {"CustomGPT_RAG", "Google_Gemini_RAG", "OpenAI_RAG"}

# Cutoff date: Dec 10th 2025
CUTOFF_DATE = datetime(2025, 12, 10)


def parse_run_date(run_name: str):
    """Extract date from run folder name like run_20251210_131902_858."""
    try:
        parts = run_name.split("_")
        if len(parts) >= 2:
            date_str = parts[1]  # 20251210
            return datetime.strptime(date_str, "%Y%m%d")
    except (ValueError, IndexError):
        pass
    return None


def get_qualifying_runs():
    """Find all runs from Dec 10th onwards."""
    qualifying = []
    for item in RESULTS_DIR.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            run_date = parse_run_date(item.name)
            if run_date and run_date >= CUTOFF_DATE:
                judge_file = item / "judge_evaluations.jsonl"
                if judge_file.exists():
                    qualifying.append(item)
    return sorted(qualifying, key=lambda x: x.name)


def extract_rag_failures(run_dir: Path):
    """
    Extract question_ids where RAG providers failed.
    Returns dict: question_id -> set of failed provider names
    """
    failures = defaultdict(set)
    judge_file = run_dir / "judge_evaluations.jsonl"

    with open(judge_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                question_id = data.get("question_id", "")
                provider_name = data.get("metadata", {}).get("real_provider_name", "")

                # Skip consistency tests and non-RAG providers
                if provider_name.startswith("ConsistencyTest_"):
                    continue
                if provider_name not in RAG_PROVIDERS:
                    continue

                # Check if this provider got it wrong
                grades = data.get("grades", {})
                for blind_id, grade in grades.items():
                    if grade == "INCORRECT":
                        failures[question_id].add(provider_name)

            except json.JSONDecodeError:
                continue

    return failures


def load_original_csv():
    """Load original CSV and create lookup by original_index."""
    with open(ORIGINAL_CSV) as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows_by_index = {}
        for row in reader:
            # original_index is first column
            orig_idx = row[0]
            rows_by_index[orig_idx] = row
    return headers, rows_by_index


def load_kb_manifest():
    """
    Load KB manifest to map original_index -> KB filename.
    Returns dict: original_index (int) -> filename (str)
    """
    if not KB_MANIFEST.exists():
        print(f"Warning: KB manifest not found at {KB_MANIFEST}")
        return {}

    with open(KB_MANIFEST) as f:
        manifest = json.load(f)

    mapping = {}
    for doc in manifest.get("documents", []):
        mapping[doc["original_index"]] = doc["filename"]

    return mapping


def main():
    print("=" * 60)
    print("Creating simpleqa_verified_hard.csv")
    print("=" * 60)

    # Find qualifying runs
    runs = get_qualifying_runs()
    print(f"\nFound {len(runs)} runs from Dec 10th onwards:")
    for r in runs:
        print(f"  - {r.name}")

    # Aggregate failures across all runs
    all_failures = defaultdict(lambda: {"providers": set(), "runs": []})

    for run_dir in runs:
        run_failures = extract_rag_failures(run_dir)
        for qid, providers in run_failures.items():
            all_failures[qid]["providers"].update(providers)
            all_failures[qid]["runs"].append(run_dir.name)

    print(f"\nTotal unique questions with RAG failures: {len(all_failures)}")

    # Show breakdown by provider
    provider_fail_counts = defaultdict(int)
    for qid, data in all_failures.items():
        for prov in data["providers"]:
            provider_fail_counts[prov] += 1

    print("\nFailures by provider:")
    for prov, count in sorted(provider_fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {prov}: {count} questions")

    # Load original CSV
    headers, rows_by_index = load_original_csv()
    print(f"\nOriginal CSV has {len(rows_by_index)} questions")

    # Load KB manifest for filename mapping
    kb_mapping = load_kb_manifest()
    print(f"Loaded KB manifest with {len(kb_mapping)} entries")

    # Match question_ids to CSV rows
    # question_id format: simpleqa_{original_index:04d} (uses original_index, not row position)
    hard_questions = []
    with open(ORIGINAL_CSV) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            orig_idx = int(row[0])  # original_index is first column
            qid = f"simpleqa_{orig_idx:04d}"
            if qid in all_failures:
                kb_file = kb_mapping.get(orig_idx, "UNKNOWN")
                hard_questions.append({
                    "row": row,
                    "qid": qid,
                    "kb_file": kb_file,
                    "failed_providers": all_failures[qid]["providers"],
                    "failed_in_runs": len(all_failures[qid]["runs"])
                })

    print(f"\nMatched {len(hard_questions)} hard questions")

    # Write the hard subset (with kb_file column)
    output_headers = headers + ["kb_file"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)
        for item in hard_questions:
            writer.writerow(item["row"] + [item["kb_file"]])

    print(f"\nWrote {len(hard_questions)} questions to:")
    print(f"  {OUTPUT_CSV}")

    # Show some statistics
    print("\n" + "=" * 60)
    print("HARD QUESTION STATISTICS")
    print("=" * 60)

    # Count by number of providers that failed
    by_fail_count = defaultdict(list)
    for item in hard_questions:
        num_failed = len(item["failed_providers"])
        by_fail_count[num_failed].append(item["qid"])

    print("\nBy number of RAG providers that failed:")
    for count in sorted(by_fail_count.keys(), reverse=True):
        print(f"  {count} providers failed: {len(by_fail_count[count])} questions")

    # Show some example hard questions
    print("\nExample hard questions (failed by all 3 RAG providers):")
    multi_fail = [q for q in hard_questions if len(q["failed_providers"]) >= 3]
    for item in multi_fail[:5]:
        row = item["row"]
        print(f"  - Q: {row[1][:80]}...")
        print(f"    A: {row[2]}")
        print()

    # SPOT CHECK: Verify mapping is correct
    print("\n" + "=" * 60)
    print("MAPPING VERIFICATION (Spot Checks)")
    print("=" * 60)

    spot_check_count = min(5, len(hard_questions))
    print(f"\nVerifying {spot_check_count} random questions have correct KB file mapping:")

    import random
    random.seed(42)
    spot_checks = random.sample(hard_questions, spot_check_count)

    for item in spot_checks:
        orig_idx = int(item["row"][0])
        kb_file = item["kb_file"]
        qid = item["qid"]
        question_preview = item["row"][1][:60]
        answer = item["row"][2]

        # Verify the KB file exists
        kb_path = KB_DIR / kb_file
        kb_exists = kb_path.exists() if kb_file != "UNKNOWN" else False

        # Read first line of KB file to show content preview
        kb_preview = ""
        if kb_exists:
            with open(kb_path) as f:
                first_lines = f.read(200).replace('\n', ' ')[:100]
                kb_preview = first_lines

        status = "OK" if kb_exists else "MISSING"
        print(f"\n  [{status}] {qid} (original_index={orig_idx})")
        print(f"      KB file: {kb_file}")
        print(f"      Q: {question_preview}...")
        print(f"      A: {answer[:60]}...")
        if kb_preview:
            print(f"      KB preview: {kb_preview}...")

    return len(hard_questions)


if __name__ == "__main__":
    main()
