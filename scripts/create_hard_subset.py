#!/usr/bin/env python3
"""
Create simpleqa_verified_hard.csv - a subset of questions where at least one
RAG provider failed (got INCORRECT) in runs after Dec 10th 2025.

Handles BOTH question_id formats:
- OLD (before df22734): simpleqa_{row_position:04d} (0-indexed row in CSV)
- NEW (after df22734): simpleqa_{original_index:04d} (original_index column value)
"""
import json
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


def load_csv_mappings():
    """
    Load CSV and build bidirectional mappings.

    Returns:
        - row_to_orig: dict mapping row_position (0-indexed) -> original_index
        - orig_to_row: dict mapping original_index -> row_position (0-indexed)
        - valid_original_indices: set of all valid original_index values
        - rows_by_orig: dict mapping original_index -> full CSV row
        - headers: CSV headers
    """
    row_to_orig = {}
    orig_to_row = {}
    rows_by_orig = {}

    with open(ORIGINAL_CSV) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row_idx, row in enumerate(reader):
            orig_idx = int(row[0])
            row_to_orig[row_idx] = orig_idx
            orig_to_row[orig_idx] = row_idx
            rows_by_orig[orig_idx] = row

    valid_original_indices = set(orig_to_row.keys())
    return row_to_orig, orig_to_row, valid_original_indices, rows_by_orig, headers


def detect_run_format(judge_file: Path, valid_original_indices: set) -> str:
    """
    Detect whether a run uses OLD (row position) or NEW (original_index) format.

    Returns: 'NEW' or 'OLD'
    """
    new_format_count = 0
    old_format_count = 0

    with open(judge_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data.get("question_id", "")
                if qid.startswith("simpleqa_") and "_consistency_" not in qid:
                    idx = int(qid.replace("simpleqa_", ""))
                    if idx in valid_original_indices:
                        new_format_count += 1
                    elif idx < 1000:  # Likely row position (0-999)
                        old_format_count += 1
            except (json.JSONDecodeError, ValueError):
                continue

    return "NEW" if new_format_count > old_format_count else "OLD"


def extract_rag_failures(run_dir: Path, valid_original_indices: set, row_to_orig: dict):
    """
    Extract failures and normalize question_ids to original_index.

    Returns:
        - failures: dict mapping original_index -> set of failed provider names
        - run_format: 'NEW' or 'OLD'
    """
    judge_file = run_dir / "judge_evaluations.jsonl"
    run_format = detect_run_format(judge_file, valid_original_indices)

    failures = defaultdict(set)

    with open(judge_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                qid = data.get("question_id", "")
                provider_name = data.get("metadata", {}).get("real_provider_name", "")

                # Skip consistency tests and non-RAG providers
                if "_consistency_" in qid:
                    continue
                if provider_name not in RAG_PROVIDERS:
                    continue
                if not qid.startswith("simpleqa_"):
                    continue

                # Check if this provider got it wrong
                grades = data.get("grades", {})
                has_failure = any(grade == "INCORRECT" for grade in grades.values())

                if has_failure:
                    # Normalize to original_index
                    idx = int(qid.replace("simpleqa_", ""))

                    if run_format == "NEW":
                        # idx IS original_index
                        if idx in valid_original_indices:
                            orig_idx = idx
                        else:
                            continue  # Invalid original_index, skip
                    else:
                        # idx is row position (0-indexed), convert to original_index
                        if idx in row_to_orig:
                            orig_idx = row_to_orig[idx]
                        else:
                            continue  # Invalid row position, skip

                    failures[orig_idx].add(provider_name)

            except (json.JSONDecodeError, ValueError):
                continue

    return failures, run_format


def load_kb_manifest():
    """
    Load KB manifest to map original_index -> KB filename.
    Returns dict: original_index (int) -> filename (str)
    """
    if not KB_MANIFEST.exists():
        raise FileNotFoundError(f"KB manifest not found at {KB_MANIFEST}")

    with open(KB_MANIFEST) as f:
        manifest = json.load(f)

    mapping = {}
    for doc in manifest.get("documents", []):
        mapping[doc["original_index"]] = doc["filename"]

    return mapping


def verify_kb_mapping(kb_mapping: dict, rows_by_orig: dict) -> bool:
    """
    Verify that the KB manifest mapping is consistent with the CSV.

    Checks:
    1. Every original_index in CSV has a KB file
    2. Every KB file exists on disk
    3. Sequence numbers are correct (row 1 -> verified_0001.txt, etc.)
    """
    print("\n" + "=" * 60)
    print("KB MANIFEST VERIFICATION")
    print("=" * 60)

    errors = []

    # Check 1: Every original_index has a KB file
    missing_kb = set(rows_by_orig.keys()) - set(kb_mapping.keys())
    if missing_kb:
        errors.append(f"Missing KB entries for {len(missing_kb)} original_indices: {sorted(missing_kb)[:5]}...")

    # Check 2: KB files exist on disk
    missing_files = []
    for orig_idx, filename in kb_mapping.items():
        if not (KB_DIR / filename).exists():
            missing_files.append((orig_idx, filename))
    if missing_files:
        errors.append(f"{len(missing_files)} KB files missing on disk")

    # Check 3: Verify sequence mapping by loading manifest and checking
    with open(KB_MANIFEST) as f:
        manifest = json.load(f)

    # Build orig_to_row for sequence check
    orig_to_row = {}
    with open(ORIGINAL_CSV) as f:
        reader = csv.reader(f)
        next(reader)
        for row_idx, row in enumerate(reader):
            orig_to_row[int(row[0])] = row_idx

    sequence_errors = []
    for doc in manifest.get("documents", []):
        orig_idx = doc["original_index"]
        expected_seq = orig_to_row.get(orig_idx, -1) + 1  # 1-indexed
        expected_filename = f"verified_{expected_seq:04d}.txt"
        if doc["filename"] != expected_filename:
            sequence_errors.append((orig_idx, doc["filename"], expected_filename))

    if sequence_errors:
        errors.append(f"{len(sequence_errors)} sequence mismatches found")
        for orig_idx, actual, expected in sequence_errors[:3]:
            print(f"  ERROR: original_index={orig_idx} has {actual}, expected {expected}")

    if errors:
        print("\n❌ VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("\n✓ All checks passed:")
        print(f"  - {len(kb_mapping)} KB entries match {len(rows_by_orig)} CSV rows")
        print(f"  - All KB files exist on disk")
        print(f"  - Sequence numbers are correct (row N -> verified_000N.txt)")
        return True


def main():
    print("=" * 60)
    print("Creating simpleqa_verified_hard.csv")
    print("=" * 60)

    # Load CSV mappings first
    row_to_orig, orig_to_row, valid_original_indices, rows_by_orig, headers = load_csv_mappings()
    print(f"\nLoaded simpleqa_verified.csv: {len(rows_by_orig)} questions")
    print(f"  original_index range: {min(valid_original_indices)} to {max(valid_original_indices)}")

    # Load and verify KB manifest
    kb_mapping = load_kb_manifest()
    print(f"Loaded KB manifest: {len(kb_mapping)} entries")

    if not verify_kb_mapping(kb_mapping, rows_by_orig):
        print("\n❌ ABORTING: KB mapping verification failed!")
        return 0

    # Find qualifying runs
    runs = get_qualifying_runs()
    print(f"\n" + "=" * 60)
    print(f"Found {len(runs)} runs from Dec 10th onwards")
    print("=" * 60)

    # Aggregate failures across all runs (normalized to original_index)
    all_failures = defaultdict(lambda: {"providers": set(), "runs": [], "formats": []})
    run_formats = {}

    for run_dir in runs:
        failures, run_format = extract_rag_failures(run_dir, valid_original_indices, row_to_orig)
        run_formats[run_dir.name] = run_format

        for orig_idx, providers in failures.items():
            all_failures[orig_idx]["providers"].update(providers)
            all_failures[orig_idx]["runs"].append(run_dir.name)
            all_failures[orig_idx]["formats"].append(run_format)

    # Show run format breakdown
    old_runs = [r for r, f in run_formats.items() if f == "OLD"]
    new_runs = [r for r, f in run_formats.items() if f == "NEW"]
    print(f"\nRun format breakdown:")
    print(f"  OLD format (row position): {len(old_runs)} runs")
    print(f"  NEW format (original_index): {len(new_runs)} runs")

    print(f"\nTotal unique questions with RAG failures: {len(all_failures)}")

    # Show breakdown by provider
    provider_fail_counts = defaultdict(int)
    for orig_idx, data in all_failures.items():
        for prov in data["providers"]:
            provider_fail_counts[prov] += 1

    print("\nFailures by provider:")
    for prov, count in sorted(provider_fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {prov}: {count} questions")

    # Build hard questions list with full verification
    hard_questions = []
    verification_errors = []

    for orig_idx in sorted(all_failures.keys()):
        # VERIFY: original_index exists in CSV
        if orig_idx not in rows_by_orig:
            verification_errors.append(f"original_index={orig_idx} not in CSV!")
            continue

        # VERIFY: original_index has KB file
        if orig_idx not in kb_mapping:
            verification_errors.append(f"original_index={orig_idx} has no KB file!")
            continue

        row = rows_by_orig[orig_idx]
        kb_file = kb_mapping[orig_idx]

        # VERIFY: KB file exists on disk
        kb_path = KB_DIR / kb_file
        if not kb_path.exists():
            verification_errors.append(f"KB file {kb_file} for original_index={orig_idx} doesn't exist!")
            continue

        # VERIFY: Row's original_index column matches our key
        if int(row[0]) != orig_idx:
            verification_errors.append(f"Row mismatch: row[0]={row[0]} but orig_idx={orig_idx}")
            continue

        hard_questions.append({
            "original_index": orig_idx,
            "row": row,
            "kb_file": kb_file,
            "failed_providers": all_failures[orig_idx]["providers"],
            "failed_in_runs": len(all_failures[orig_idx]["runs"])
        })

    if verification_errors:
        print(f"\n❌ VERIFICATION ERRORS ({len(verification_errors)}):")
        for err in verification_errors[:10]:
            print(f"  - {err}")
        if len(verification_errors) > 10:
            print(f"  ... and {len(verification_errors) - 10} more")
        print("\n❌ ABORTING: Verification failed!")
        return 0

    print(f"\n✓ All {len(hard_questions)} hard questions verified:")
    print(f"  - All original_indices exist in simpleqa_verified.csv")
    print(f"  - All have corresponding KB files in manifest")
    print(f"  - All KB files exist on disk")

    # Write the hard subset (with kb_file column)
    output_headers = headers + ["kb_file"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_headers)
        for item in hard_questions:
            writer.writerow(item["row"] + [item["kb_file"]])

    print(f"\nWrote {len(hard_questions)} questions to:")
    print(f"  {OUTPUT_CSV}")

    # Show statistics
    print("\n" + "=" * 60)
    print("HARD QUESTION STATISTICS")
    print("=" * 60)

    # Count by number of providers that failed
    by_fail_count = defaultdict(list)
    for item in hard_questions:
        num_failed = len(item["failed_providers"])
        by_fail_count[num_failed].append(item["original_index"])

    print("\nBy number of RAG providers that failed:")
    for count in sorted(by_fail_count.keys(), reverse=True):
        print(f"  {count} providers failed: {len(by_fail_count[count])} questions")

    # COMPREHENSIVE SPOT CHECKS
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SPOT CHECKS (5 random questions)")
    print("=" * 60)

    import random
    random.seed(42)
    spot_checks = random.sample(hard_questions, min(5, len(hard_questions)))

    all_checks_passed = True
    for item in spot_checks:
        orig_idx = item["original_index"]
        kb_file = item["kb_file"]
        row = item["row"]
        question = row[1]
        answer = row[2]

        # Compute expected KB file from row position
        row_pos = orig_to_row[orig_idx]  # 0-indexed
        expected_kb_file = f"verified_{row_pos + 1:04d}.txt"

        # Verify KB file matches expected
        kb_match = kb_file == expected_kb_file

        # Read KB file and check it exists
        kb_path = KB_DIR / kb_file
        kb_exists = kb_path.exists()

        # Read KB content preview
        kb_preview = ""
        if kb_exists:
            with open(kb_path) as f:
                kb_preview = f.read(300).replace('\n', ' ')[:150]

        # Check if answer appears in KB (basic check)
        answer_in_kb = False
        if kb_exists:
            with open(kb_path) as f:
                kb_content = f.read().lower()
                # Check for answer or key parts of answer
                answer_lower = answer.lower()
                # Strip parenthetical notes for matching
                answer_clean = answer_lower.split("(")[0].strip()
                answer_in_kb = answer_clean in kb_content or any(
                    part.strip() in kb_content
                    for part in answer_clean.split()
                    if len(part.strip()) > 3
                )

        status = "✓" if (kb_match and kb_exists) else "❌"
        if not (kb_match and kb_exists):
            all_checks_passed = False

        print(f"\n{status} original_index={orig_idx} (CSV row {row_pos})")
        print(f"   KB file: {kb_file} {'(CORRECT)' if kb_match else f'(EXPECTED: {expected_kb_file})'}")
        print(f"   KB exists: {kb_exists}")
        print(f"   Q: {question[:70]}...")
        print(f"   A: {answer[:50]}...")
        print(f"   Answer in KB: {'likely yes' if answer_in_kb else 'not found (may need deeper check)'}")
        if kb_preview:
            print(f"   KB preview: {kb_preview}...")

    if all_checks_passed:
        print(f"\n✓ ALL SPOT CHECKS PASSED")
    else:
        print(f"\n❌ SOME SPOT CHECKS FAILED - REVIEW OUTPUT ABOVE")

    return len(hard_questions)


if __name__ == "__main__":
    main()
