#!/usr/bin/env python3
"""
Add gold_in_kb column to simpleqa-verified.csv.

Strategy:
1. Exact string match (case-insensitive, normalized whitespace)
2. If no match, use LLM to check semantic equivalence

This helps identify:
- Which questions have KB files that actually contain the gold answer
- Provider faithfulness vs parametric memory usage

Usage:
    docker compose run --rm simple-evals python scripts/add_gold_in_kb_column.py
    docker compose run --rm simple-evals python scripts/add_gold_in_kb_column.py --sample 50  # Process first 50
"""
import csv
import json
import os
import re
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Paths (Docker context)
KB_DIR = Path("/app/knowledge_base_verified")
CSV_PATH = Path("/app/simpleqa-verified/simpleqa_verified.csv")
OUTPUT_PATH = Path("/app/simpleqa-verified/simpleqa_verified_with_kb_check.csv")
PROGRESS_PATH = Path("/app/simpleqa-verified/kb_check_progress.json")

client = OpenAI()


def normalize_text(text: str) -> str:
    """Normalize for comparison - lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def gold_in_file_exact(gold: str, content: str) -> bool:
    """Check if gold answer is in file content (case-insensitive, normalized)."""
    gold_norm = normalize_text(gold)
    content_norm = normalize_text(content)
    return gold_norm in content_norm


def gold_in_file_partial(gold: str, content: str) -> bool:
    """Check for partial matches - useful for numbers, IDs, etc.

    This handles cases like:
    - "120,000 euros" matching "120,000" in content
    - "C9LVQ0YUXG" matching the UNII code in content
    """
    # Extract meaningful tokens from gold (remove common words)
    gold_clean = normalize_text(gold)
    content_norm = normalize_text(content)

    # For numeric answers, extract the number
    numbers = re.findall(r'\d+[,.]?\d*', gold_clean)
    for num in numbers:
        num_clean = num.replace(',', '')
        if num_clean in content_norm.replace(',', ''):
            return True

    # For alphanumeric IDs (like UNII codes), check direct presence
    alphanum = re.findall(r'[A-Z0-9]{6,}', gold.upper())
    for code in alphanum:
        if code.lower() in content_norm:
            return True

    return False


def gold_in_file_llm(gold: str, content: str, question: str) -> tuple[bool, str]:
    """Use LLM to check if gold answer is semantically present.

    Returns (found: bool, explanation: str)
    """
    # Truncate content to fit in context (roughly 15k chars ~ 4k tokens)
    content_truncated = content[:15000]

    prompt = f"""You are verifying if a knowledge base document contains information that would allow answering a specific question.

**Question**: {question}

**Expected Gold Answer**: {gold}

**Document Content**:
```
{content_truncated}
```

**Task**: Determine if this document contains information sufficient to derive the answer "{gold}".

The information does NOT need to be word-for-word identical, but must be semantically equivalent or contain the key facts needed to answer correctly.

Examples of matches:
- Gold: "4 December 2001" matches document containing "December 4, 2001" or "4th of December, 2001"
- Gold: "160" (strike rate) matches document containing "strike rate of 160.00"
- Gold: "ROC" matches document containing "Russian Olympic Committee (ROC)"

Examples of non-matches:
- Document discusses the topic but doesn't contain the specific answer
- Document contains different values for the same question
- Document is about a different entity with similar name

**Response Format**:
Answer with ONLY a JSON object:
{{"found": true/false, "explanation": "brief reason"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle case where response might have markdown code blocks
        if "```" in result_text:
            result_text = re.search(r'\{.*\}', result_text, re.DOTALL).group()

        result = json.loads(result_text)
        return result.get("found", False), result.get("explanation", "")

    except Exception as e:
        return False, f"LLM error: {str(e)}"


def check_gold_in_kb(row: dict, row_index: int, use_llm: bool = True) -> dict:
    """Check if gold answer is in KB file.

    Args:
        row: CSV row dict
        row_index: 0-based index in the CSV (NOT original_index)

    Returns dict with:
    - gold_in_kb: "true" or "false"
    - kb_check_method: "exact_match", "partial_match", "llm_semantic", "not_found", or "file_missing"
    - kb_check_explanation: explanation (for LLM checks)

    Note: KB files are numbered 1-1000 based on position in verified CSV,
    NOT based on original_index (which references the full SimpleQA dataset).
    """
    # KB filename uses 1-indexed position in our verified subset (row 0 -> verified_0001.txt)
    kb_filename = f"verified_{row_index + 1:04d}.txt"
    kb_path = KB_DIR / kb_filename

    result = {
        "gold_in_kb": "false",
        "kb_check_method": "not_found",
        "kb_check_explanation": "",
        "kb_filename": kb_filename
    }

    if not kb_path.exists():
        result["kb_check_method"] = "file_missing"
        result["kb_check_explanation"] = f"KB file {kb_filename} not found"
        return result

    try:
        content = kb_path.read_text(encoding='utf-8')
    except Exception as e:
        result["kb_check_method"] = "file_error"
        result["kb_check_explanation"] = f"Error reading file: {str(e)}"
        return result

    gold = row.get("answer", "").strip()
    question = row.get("problem", "").strip()

    if not gold:
        result["kb_check_method"] = "no_gold_answer"
        result["kb_check_explanation"] = "No gold answer in CSV"
        return result

    # Strategy 1: Exact string match
    if gold_in_file_exact(gold, content):
        result["gold_in_kb"] = "true"
        result["kb_check_method"] = "exact_match"
        result["kb_check_explanation"] = f"Found '{gold[:50]}...' verbatim in KB"
        return result

    # Strategy 2: Partial/numeric match
    if gold_in_file_partial(gold, content):
        result["gold_in_kb"] = "true"
        result["kb_check_method"] = "partial_match"
        result["kb_check_explanation"] = "Found partial/numeric match in KB"
        return result

    # Strategy 3: LLM semantic check (if enabled)
    if use_llm:
        found, explanation = gold_in_file_llm(gold, content, question)
        if found:
            result["gold_in_kb"] = "true"
            result["kb_check_method"] = "llm_semantic"
            result["kb_check_explanation"] = explanation
        else:
            result["kb_check_method"] = "not_found"
            result["kb_check_explanation"] = explanation or "LLM did not find semantic match"

    return result


def load_progress() -> dict:
    """Load progress from checkpoint file."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"processed": [], "results": {}}


def save_progress(progress: dict):
    """Save progress to checkpoint file."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Add gold_in_kb column to simpleqa CSV")
    parser.add_argument("--sample", type=int, help="Only process first N rows")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM fallback (faster)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for LLM calls")
    args = parser.parse_args()

    print("=" * 80)
    print("GOLD-IN-KB VERIFICATION")
    print("=" * 80)
    print(f"CSV: {CSV_PATH}")
    print(f"KB Directory: {KB_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"LLM Fallback: {'disabled' if args.no_llm else 'enabled'}")
    print()

    # Check paths exist
    if not KB_DIR.exists():
        print(f"ERROR: KB directory not found: {KB_DIR}")
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV file not found: {CSV_PATH}")
        sys.exit(1)

    # Load CSV
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + ['gold_in_kb', 'kb_check_method', 'kb_check_explanation', 'kb_filename']

    if args.sample:
        rows = rows[:args.sample]
        print(f"Processing sample of {args.sample} rows\n")
    else:
        print(f"Processing all {len(rows)} rows\n")

    # Load progress if resuming
    progress = load_progress() if args.resume else {"processed": [], "results": {}}

    # Process rows
    use_llm = not args.no_llm
    processed_count = 0
    stats = {"exact_match": 0, "partial_match": 0, "llm_semantic": 0, "not_found": 0, "file_missing": 0, "file_error": 0}

    def process_row(idx_row):
        idx, row = idx_row
        original_index = row.get("original_index", "")

        # Skip if already processed (use row index as key)
        if str(idx) in progress["results"]:
            return idx, row, progress["results"][str(idx)]

        result = check_gold_in_kb(row, row_index=idx, use_llm=use_llm)
        return idx, row, result

    # Use thread pool for LLM calls
    results_list = []
    with ThreadPoolExecutor(max_workers=args.workers if use_llm else 1) as executor:
        futures = {executor.submit(process_row, (i, row)): i for i, row in enumerate(rows)}

        for future in as_completed(futures):
            idx, row, result = future.result()
            results_list.append((idx, row, result))

            # Update row with results
            row.update(result)

            # Update stats
            method = result["kb_check_method"]
            if method in stats:
                stats[method] += 1

            # Save progress periodically (use row index as key)
            progress["results"][str(idx)] = result
            processed_count += 1

            if processed_count % 50 == 0:
                save_progress(progress)
                found = stats["exact_match"] + stats["partial_match"] + stats["llm_semantic"]
                print(f"Progress: {processed_count}/{len(rows)} | Found: {found} | Not Found: {stats['not_found']}")

    # Sort results back to original order
    results_list.sort(key=lambda x: x[0])

    # Write output CSV
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row, result in results_list:
            row.update(result)
            writer.writerow(row)

    # Final stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found_count = stats["exact_match"] + stats["partial_match"] + stats["llm_semantic"]
    total = len(rows)

    print(f"\n📊 Overall Results:")
    print(f"   Total questions: {total}")
    print(f"   Gold IN KB: {found_count} ({100*found_count/total:.1f}%)")
    print(f"   Gold NOT in KB: {stats['not_found']} ({100*stats['not_found']/total:.1f}%)")

    print(f"\n📋 Detection Method Breakdown:")
    print(f"   Exact match:     {stats['exact_match']:4d}")
    print(f"   Partial match:   {stats['partial_match']:4d}")
    print(f"   LLM semantic:    {stats['llm_semantic']:4d}")
    print(f"   Not found:       {stats['not_found']:4d}")
    print(f"   File missing:    {stats['file_missing']:4d}")
    print(f"   File error:      {stats['file_error']:4d}")

    print(f"\n✅ Output saved to: {OUTPUT_PATH}")

    # Clean up progress file
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
