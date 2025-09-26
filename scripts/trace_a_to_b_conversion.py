#!/usr/bin/env python3
"""
Trace A->B Conversion Bug
This will cross-reference judge evaluations with the final result metrics to find exactly
which 84 A grades are being counted as incorrect.
"""

import json
from collections import defaultdict

def main():
    print("🔍 TRACING A->B CONVERSION BUG")
    print("=" * 60)
    print()

    # Load judge evaluations with their true grades
    judge_grades = {}
    abstention_classifications = {}

    print("Loading judge evaluations...")
    with open('results/run_20250926_033923_770/judge_evaluations.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                eval_data = json.loads(line)
                if eval_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                    question_id = eval_data['question_id']
                    judge_response = json.loads(eval_data['judge']['response'])
                    judge_grades[question_id] = {
                        'grade': judge_response['grade'],
                        'confidence': judge_response.get('confidence', 'N/A'),
                        'reasoning': judge_response.get('reasoning', 'N/A')[:100] + "..."
                    }

    print(f"Loaded {len(judge_grades)} judge evaluations")
    print()

    # Check if there's an abstention classifications file
    try:
        print("Loading abstention classifications...")
        with open('results/run_20250926_033923_770/abstention_classifications.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    abstention_data = json.loads(line)
                    if abstention_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                        question_id = abstention_data['question_id']
                        abstention_classifications[question_id] = {
                            'type': abstention_data['classifier']['classification'],
                            'confidence': abstention_data['classifier']['confidence'],
                            'reasoning': abstention_data['classifier']['reasoning'][:100] + "..."
                        }

        print(f"Loaded {len(abstention_classifications)} abstention classifications")
    except FileNotFoundError:
        print("No abstention classifications file found")
        abstention_classifications = {}

    print()

    # Analyze grade distribution
    judge_grade_counts = defaultdict(int)
    for grade_data in judge_grades.values():
        judge_grade_counts[grade_data['grade']] += 1

    print("Judge Grade Distribution:")
    for grade, count in sorted(judge_grade_counts.items()):
        print(f"   {grade}: {count}")
    print()

    # Look for conflicts between judge grades and abstention classifications
    conflicts = []
    abstention_overrides = []

    for question_id in judge_grades:
        judge_grade = judge_grades[question_id]['grade']

        if question_id in abstention_classifications:
            abstention_type = abstention_classifications[question_id]['type']

            # Check for A grades that get overridden by abstention
            if judge_grade == 'A' and abstention_type == 'abstention':
                abstention_overrides.append({
                    'question_id': question_id,
                    'judge_grade': judge_grade,
                    'judge_confidence': judge_grades[question_id]['confidence'],
                    'abstention_type': abstention_type,
                    'abstention_confidence': abstention_classifications[question_id]['confidence'],
                    'judge_reasoning': judge_grades[question_id]['reasoning'],
                    'abstention_reasoning': abstention_classifications[question_id]['reasoning']
                })

    print("🚨 A GRADES OVERRIDDEN BY ABSTENTION CLASSIFICATION:")
    print(f"Found {len(abstention_overrides)} cases where judge said A but abstention classifier said abstention")
    print()

    if len(abstention_overrides) == 84:
        print("✅ FOUND THE BUG! 84 A grades are being overridden by abstention classification")
        print()

        print("Sample cases (first 10):")
        for i, case in enumerate(abstention_overrides[:10], 1):
            print(f"{i:2d}. Q{case['question_id']}")
            print(f"     Judge: Grade {case['judge_grade']} (conf: {case['judge_confidence']})")
            print(f"     Abstention: {case['abstention_type']} (conf: {case['abstention_confidence']:.2f})")
            print(f"     Judge reasoning: {case['judge_reasoning']}")
            print(f"     Abstention reasoning: {case['abstention_reasoning']}")
            print()

    elif len(abstention_overrides) > 0:
        print(f"🔍 Found {len(abstention_overrides)} A->abstention overrides, need to investigate further")
        print("This might be part of the issue but not the complete picture")

    else:
        print("❓ No A->abstention overrides found. Bug must be elsewhere.")
        print()
        print("Other potential issues to check:")
        print("1. Judge consistency validation changing A->B")
        print("2. Post-processing logic modifying grades")
        print("3. Aggregation counting logic errors")

    print()
    print("🎯 CONCLUSION:")
    if len(abstention_overrides) == 84:
        print("The bug is that 84 judge A grades are being overridden by abstention classification")
        print("These responses went to the judge, got graded A, but then abstention classifier")
        print("decided they were actually abstentions and changed them to not_attempted")
    else:
        print("Need to investigate other parts of the pipeline")

if __name__ == "__main__":
    main()