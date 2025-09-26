#!/usr/bin/env python3
"""
Find Aggregation Bug
The issue is now clear: 28 abstentions are correctly handled, but 84 A grades
are being miscounted as incorrect in the aggregation. Let's find exactly where.
"""

import json
from collections import Counter

def main():
    print("🔍 FINDING THE AGGREGATION BUG")
    print("=" * 60)
    print()

    # Load all judge grades and simulate the exact aggregation
    judge_evaluations = []

    with open('results/run_20250926_033923_770/judge_evaluations.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                eval_data = json.loads(line)
                if eval_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                    judge_response = json.loads(eval_data['judge']['response'])
                    judge_evaluations.append({
                        'question_id': eval_data['question_id'],
                        'grade': judge_response['grade'],
                        'confidence': judge_response.get('confidence', 'N/A')
                    })

    print(f"Judge evaluations loaded: {len(judge_evaluations)}")

    # Count grades
    grade_counts = Counter(eval['grade'] for eval in judge_evaluations)
    print(f"Judge grade counts: {dict(grade_counts)}")
    print()

    # Simulate the aggregation logic exactly as in the code
    n_correct = sum(1 for eval in judge_evaluations if eval['grade'] == "A")
    n_incorrect = sum(1 for eval in judge_evaluations if eval['grade'] == "B")
    n_not_attempted = sum(1 for eval in judge_evaluations if eval['grade'] not in ["A", "B"])

    print("SIMULATED AGGREGATION (from judge data only):")
    print(f"   n_correct: {n_correct}")
    print(f"   n_incorrect: {n_incorrect}")
    print(f"   n_not_attempted: {n_not_attempted}")
    print(f"   Total: {n_correct + n_incorrect + n_not_attempted}")
    print()

    # Compare with final results
    with open('results/run_20250926_033923_770/quality_benchmark_results.json', 'r') as f:
        data = json.load(f)

    customgpt_result = None
    for result in data['results']:
        if result['sampler_name'] == 'CustomGPT_RAG':
            customgpt_result = result
            break

    if customgpt_result:
        metrics = customgpt_result['metrics']
        print("ACTUAL FINAL RESULTS:")
        print(f"   n_correct: {metrics['n_correct']}")
        print(f"   n_incorrect: {metrics['n_incorrect']}")
        print(f"   n_not_attempted: {metrics['n_not_attempted']}")
        print(f"   Total: {metrics['n_correct'] + metrics['n_incorrect'] + metrics['n_not_attempted']}")
        print()

        print("📊 DETAILED DISCREPANCY ANALYSIS:")
        print(f"   Judge correct (A grades): {n_correct}")
        print(f"   Final correct: {metrics['n_correct']}")
        print(f"   Lost correct answers: {n_correct - metrics['n_correct']}")
        print()
        print(f"   Judge incorrect (B grades): {n_incorrect}")
        print(f"   Final incorrect: {metrics['n_incorrect']}")
        print(f"   Gained incorrect answers: {metrics['n_incorrect'] - n_incorrect}")
        print()
        print(f"   Judge not attempted: {n_not_attempted}")
        print(f"   Final not attempted: {metrics['n_not_attempted']}")
        print(f"   Abstention handling: {'✅ CORRECT' if n_not_attempted == 0 and metrics['n_not_attempted'] == 28 else '❌ ISSUE'}")
        print()

    print("🎯 CONCLUSION:")
    print("The issue is that 84 A grades from the judge are being:")
    print("1. Lost from the correct count (-84)")
    print("2. Added to the incorrect count (+84)")
    print()
    print("This suggests a bug in the evaluation result creation where:")
    print("- Some judge A grades are being processed as incorrect")
    print("- OR there's a mapping issue between questions and grades")
    print("- OR there's duplicate processing with different outcomes")
    print()
    print("🔍 NEXT: Examine the exact evaluation loop in confidence_threshold_simpleqa_eval.py")
    print("Look for where SingleEvalResult objects are created and how grades are assigned")

if __name__ == "__main__":
    main()