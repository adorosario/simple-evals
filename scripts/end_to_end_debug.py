#!/usr/bin/env python3
"""
End-to-End Debug: Complete Pipeline Trace
This will trace the exact path from judge evaluations to final JSON to find where 84 A grades disappear
"""

import json
from collections import defaultdict, Counter

def main():
    print("🔍 END-TO-END PIPELINE DEBUG")
    print("=" * 70)
    print()

    # Step 1: Count raw judge grades
    print("STEP 1: Raw Judge Grade Analysis")
    judge_grades = {}
    judge_grade_counts = Counter()

    with open('results/run_20250926_033923_770/judge_evaluations.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                eval_data = json.loads(line)
                if eval_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                    question_id = eval_data['question_id']
                    judge_response = json.loads(eval_data['judge']['response'])
                    grade = judge_response['grade']
                    judge_grades[question_id] = grade
                    judge_grade_counts[grade] += 1

    print(f"Judge evaluations: {len(judge_grades)} total")
    for grade, count in sorted(judge_grade_counts.items()):
        print(f"   {grade}: {count}")
    print()

    # Step 2: Check provider responses to see if there are mismatches
    print("STEP 2: Provider Response Analysis")
    provider_responses = {}

    with open('results/run_20250926_033923_770/provider_requests.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                request_data = json.loads(line)
                if request_data['provider'] == 'CustomGPT_RAG':
                    question_id = request_data['question_id']
                    response = request_data['response']['content']
                    provider_responses[question_id] = response[:100] + "..." if len(response) > 100 else response

    print(f"Provider responses: {len(provider_responses)} total")
    print()

    # Step 3: Cross-reference questions to find any evaluation gaps
    print("STEP 3: Cross-Reference Analysis")
    judge_questions = set(judge_grades.keys())
    provider_questions = set(provider_responses.keys())

    only_in_judge = judge_questions - provider_questions
    only_in_provider = provider_questions - judge_questions
    in_both = judge_questions & provider_questions

    print(f"Questions only in judge: {len(only_in_judge)}")
    print(f"Questions only in provider: {len(only_in_provider)}")
    print(f"Questions in both: {len(in_both)}")
    print()

    if only_in_provider:
        print("❗ CRITICAL FINDING: Questions in provider but not judge!")
        print("This suggests some responses didn't make it to the judge")
        print(f"Missing from judge: {sorted(list(only_in_provider))[:10]}")
        print()

    # Step 4: Check the actual aggregation result structure
    print("STEP 4: Final Results Analysis")
    with open('results/run_20250926_033923_770/quality_benchmark_results.json', 'r') as f:
        data = json.load(f)

    customgpt_result = None
    for result in data['results']:
        if result['sampler_name'] == 'CustomGPT_RAG':
            customgpt_result = result
            break

    if customgpt_result:
        metrics = customgpt_result['metrics']
        print(f"Final results show:")
        print(f"   n_correct: {metrics['n_correct']}")
        print(f"   n_incorrect: {metrics['n_incorrect']}")
        print(f"   n_not_attempted: {metrics['n_not_attempted']}")
        print(f"   conversations: {metrics['conversations']}")
        print()

        total_final = metrics['n_correct'] + metrics['n_incorrect'] + metrics['n_not_attempted']
        print(f"Total in final results: {total_final}")
        print(f"Total judge evaluations: {len(judge_grades)}")
        print(f"Total provider responses: {len(provider_responses)}")
        print()

        # Calculate the discrepancy
        discrepancy = judge_grade_counts['A'] - metrics['n_correct']
        print(f"📊 DISCREPANCY ANALYSIS:")
        print(f"   Judge A grades: {judge_grade_counts['A']}")
        print(f"   Final correct: {metrics['n_correct']}")
        print(f"   Missing A grades: {discrepancy}")
        print()

        print(f"   Judge B grades: {judge_grade_counts['B']}")
        print(f"   Final incorrect: {metrics['n_incorrect']}")
        print(f"   Extra incorrect: {metrics['n_incorrect'] - judge_grade_counts['B']}")
        print()

    # Step 5: Hypothesis testing
    print("STEP 5: HYPOTHESIS TESTING")
    print()
    print("❌ Eliminated causes:")
    print("   1. Abstention classification override: No A->abstention overrides found")
    print("   2. Judge validation changes: No A->B validation changes found")
    print()
    print("🔍 Remaining possibilities:")
    print("   1. Evaluation counting logic error (double-counting incorrect responses)")
    print("   2. Response-to-evaluation mapping issue")
    print("   3. Bug in aggregation where responses are processed multiple times")
    print("   4. Issue in the evaluation loop itself")
    print()

    if len(provider_responses) != len(judge_grades):
        print("🚨 SMOKING GUN: Mismatch between provider responses and judge evaluations!")
        print("This suggests not all responses are being judged correctly")
    else:
        print("💡 NEXT INVESTIGATION: Check if the same question is being processed multiple times")
        print("or if there's a bug in the evaluation result creation logic")

    print()
    print("🎯 RECOMMENDATION:")
    print("Investigate the evaluation loop in confidence_threshold_simpleqa_eval.py")
    print("Look for duplicate processing or incorrect result aggregation")

if __name__ == "__main__":
    main()