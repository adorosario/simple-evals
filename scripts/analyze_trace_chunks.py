#!/usr/bin/env python3
"""
Extract and compare CONTEXT chunks from MySQL trace files.
Determines if chunks are deterministic and if gold answers are present.

This script analyzes the openai_prompt field from MySQL traces to:
1. Extract the CONTEXT section (chunks sent to LLM)
2. Check if chunks are identical across runs (determinism)
3. Check if gold answer appears in any chunk
4. Output detailed analysis for each abstention question

Usage:
    docker compose run --rm simple-evals python scripts/analyze_trace_chunks.py
"""
import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Path to trace files - check local copy first, then original location
LOCAL_TRACES = Path("/app/data/traces")  # Inside Docker
ORIGINAL_TRACES = Path("/home/adorosario/quick-and-dirty/customgpt-response-debugger/issues/nondeterministic-retrieval/traces")

# Use local copy if available (Docker), otherwise original path (host)
TRACES_DIR = LOCAL_TRACES if LOCAL_TRACES.exists() else ORIGINAL_TRACES

# Gold answers for each question (from simpleqa dataset)
GOLD_ANSWERS = {
    "rimegepant": "51049968",
    "le_chatelier": "10",
    "axitinib": "C9LVQ0YUXG",
    "miss_world": "Lesley Langley",
    "tokyo_olympics": "ROC",
    "lady_annabel": "Miss Annabel Lee",
    "manish_pandey": "160",
    "selvaganapathy": "4 December 2001",
    "solar_eclipse": "1.0495",
    "glipodes": "1962"
}


def extract_context_section(openai_prompt: str) -> str:
    """Extract content between <CONTEXT> and </CONTEXT> tags.

    Note: There are multiple CONTEXT sections in the prompt:
    - Empty ones in the instructions
    - The actual chunk content (contains '* filename:')

    We want the one with the actual chunks.
    """
    # Find all CONTEXT sections
    matches = list(re.finditer(r'<CONTEXT>(.*?)</CONTEXT>', openai_prompt, re.DOTALL))

    # Return the one containing actual chunks (has '* filename:')
    for match in matches:
        content = match.group(1).strip()
        if '* filename:' in content:
            return content

    # Fallback: return the longest non-empty section
    non_empty = [m.group(1).strip() for m in matches if m.group(1).strip()]
    if non_empty:
        return max(non_empty, key=len)

    return ""


def extract_chunks(context: str) -> list[str]:
    """Parse individual chunks from CONTEXT section.

    Chunks are prefixed with '* filename:' in the CustomGPT prompt format.
    """
    # Split on chunk boundaries
    chunks = re.split(r'\* filename:', context)
    # Clean up chunks, excluding the time header
    cleaned = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and not chunk.startswith('*Current time'):
            cleaned.append(chunk)
    return cleaned


def compute_chunk_hash(chunks: list[str]) -> str:
    """Compute stable hash for chunk list to compare across runs."""
    content = "|||".join(chunks)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def check_gold_in_chunks(chunks: list[str], gold: str) -> bool:
    """Check if gold answer appears in any chunk (case-insensitive)."""
    gold_lower = gold.lower()
    for chunk in chunks:
        if gold_lower in chunk.lower():
            return True
    return False


def analyze_trace_file(filepath: Path, gold_answer: str) -> dict:
    """Analyze all runs in a trace file."""
    with open(filepath) as f:
        traces = json.load(f)

    if not traces:
        return {"error": "No traces in file"}

    results = {
        "question": traces[0].get("question", ""),
        "gold_answer": gold_answer,
        "total_runs": len(traces),
        "runs": [],
        "unique_chunk_hashes": set(),
        "gold_in_any_chunk": False
    }

    for trace in traces:
        prompt_id = trace.get("prompt_id", "unknown")
        response = trace.get("openai_response", "")[:100]
        openai_prompt = trace.get("openai_prompt", "")

        context = extract_context_section(openai_prompt)
        chunks = extract_chunks(context)
        chunk_hash = compute_chunk_hash(chunks)
        gold_found = check_gold_in_chunks(chunks, gold_answer)

        results["runs"].append({
            "prompt_id": prompt_id,
            "response_preview": response,
            "num_chunks": len(chunks),
            "chunk_hash": chunk_hash,
            "gold_in_chunks": gold_found,
            "is_abstention": "sorry" in response.lower() or "don't know" in response.lower()
        })

        results["unique_chunk_hashes"].add(chunk_hash)
        if gold_found:
            results["gold_in_any_chunk"] = True

    # Summary stats
    results["is_deterministic"] = len(results["unique_chunk_hashes"]) == 1
    results["num_unique_hashes"] = len(results["unique_chunk_hashes"])
    results["unique_chunk_hashes"] = list(results["unique_chunk_hashes"])  # Convert set to list for JSON

    return results


def main():
    print("=" * 80)
    print("CHUNK DETERMINISM & GOLD ANSWER ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing traces from: {TRACES_DIR}")
    print()

    if not TRACES_DIR.exists():
        print(f"ERROR: Traces directory not found: {TRACES_DIR}")
        return

    all_results = {}
    summary = {
        "deterministic": 0,
        "non_deterministic": 0,
        "gold_in_chunks": 0,
        "gold_not_in_chunks": 0
    }

    trace_files = sorted(TRACES_DIR.glob("*_traces.json"))
    if not trace_files:
        print(f"ERROR: No trace files found in {TRACES_DIR}")
        return

    print(f"Found {len(trace_files)} trace files\n")
    print("-" * 80)

    for trace_file in trace_files:
        name = trace_file.stem.replace("_traces", "")
        gold = GOLD_ANSWERS.get(name, "UNKNOWN")

        result = analyze_trace_file(trace_file, gold)
        all_results[name] = result

        # Determine status
        if result.get("is_deterministic"):
            det_status = "DETERMINISTIC"
            summary["deterministic"] += 1
        else:
            det_status = "NON-DETERMINISTIC"
            summary["non_deterministic"] += 1

        if result.get("gold_in_any_chunk"):
            gold_status = "IN CHUNKS"
            summary["gold_in_chunks"] += 1
        else:
            gold_status = "NOT IN CHUNKS"
            summary["gold_not_in_chunks"] += 1

        # Print per-question analysis
        print(f"\n{name.upper()}")
        print(f"  Question: {result.get('question', '')[:80]}...")
        print(f"  Gold answer: '{gold}'")
        print(f"  Chunks: {det_status} ({result.get('num_unique_hashes', 0)} unique version(s))")
        print(f"  Gold in chunks: {gold_status}")
        print(f"  Runs analyzed: {result.get('total_runs', 0)}")

        # Show per-run details
        for run in result.get("runs", []):
            abstention = "ABSTAINED" if run["is_abstention"] else "ANSWERED"
            print(f"    - {run['prompt_id']}: {abstention}, {run['num_chunks']} chunks, hash={run['chunk_hash']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nChunk Determinism:")
    print(f"  Deterministic: {summary['deterministic']}/{len(trace_files)}")
    print(f"  Non-deterministic: {summary['non_deterministic']}/{len(trace_files)}")

    print(f"\nGold Answer Presence:")
    print(f"  Gold in chunks: {summary['gold_in_chunks']}/{len(trace_files)}")
    print(f"  Gold NOT in chunks: {summary['gold_not_in_chunks']}/{len(trace_files)}")

    print(f"\nRoot Cause Analysis:")
    for name, result in all_results.items():
        gold = GOLD_ANSWERS.get(name, "?")
        if result.get("gold_in_any_chunk"):
            cause = "LLM_EXTRACTION_FAIL"
        else:
            cause = "CHUNKING_OR_KB_GAP"
        print(f"  {name}: {cause}")

    # Save full results to JSON
    output_path = TRACES_DIR / "chunk_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
