#!/usr/bin/env python3
"""
Deep Debug: Track Individual Grade Changes
This will examine every single CustomGPT evaluation to find exactly which 84 are being miscounted.
"""

import json

def main():
    print("🔍 DEEP DEBUGGING: INDIVIDUAL GRADE TRACKING")
    print("=" * 70)
    print()

    # Load judge evaluations and track grades
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0, "other": 0}
    sample_evaluations = []

    with open('results/run_20250926_033923_770/judge_evaluations.jsonl', 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    eval_data = json.loads(line)
                    if eval_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                        judge_response = json.loads(eval_data['judge']['response'])
                        grade_letter = judge_response['grade']
                        grade_counts[grade_letter if grade_letter in grade_counts else "other"] += 1

                        # Collect sample evaluations for detailed analysis
                        if len(sample_evaluations) < 20:
                            sample_evaluations.append({
                                "question_id": eval_data['question_id'],
                                "grade": grade_letter,
                                "confidence": judge_response.get('confidence', 'N/A'),
                                "reasoning": judge_response.get('reasoning', 'N/A')[:200] + "..." if len(judge_response.get('reasoning', '')) > 200 else judge_response.get('reasoning', 'N/A'),
                                "line_number": line_num
                            })

                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")

    print("📊 RAW JUDGE GRADE DISTRIBUTION:")
    for grade, count in grade_counts.items():
        print(f"   {grade}: {count}")
    print()

    total_evaluated = sum(grade_counts.values())
    print(f"Total evaluations: {total_evaluated}")
    print(f"A grades (correct): {grade_counts['A']}")
    print(f"B grades (incorrect): {grade_counts['B']}")
    print(f"Non-A/B grades: {total_evaluated - grade_counts['A'] - grade_counts['B']}")
    print()

    print("🔍 SAMPLE EVALUATIONS (first 20):")
    for i, eval_sample in enumerate(sample_evaluations, 1):
        print(f"{i:2d}. Q{eval_sample['question_id']} | Grade: {eval_sample['grade']} | Conf: {eval_sample['confidence']}")
        print(f"     Reasoning: {eval_sample['reasoning']}")
        print()

    print("💡 ANALYSIS:")
    if grade_counts['A'] == 925 and grade_counts['B'] == 47:
        print("✅ Raw judge evaluations show correct distribution: 925 A grades, 47 B grades")
        print("❌ But final results show: 841 correct, 131 incorrect")
        print("🐛 Bug confirmed: 84 A grades are being converted to incorrect somewhere")
        print()
        print("HYPOTHESIS TESTING:")
        print("1. 84 A grades might be getting converted to B due to abstention logic")
        print("2. Or there's a post-processing step that modifies the grades")
        print("3. Or the aggregation is somehow double-counting incorrect responses")
    else:
        print("❓ Raw judge evaluations don't match expected 925/47 distribution")
        print("🔍 Need to investigate judge evaluation process")

    print()
    print("🎯 NEXT STEPS:")
    print("1. Check if abstention classification is changing A->B grades")
    print("2. Look for any post-processing that modifies grades")
    print("3. Verify the aggregation logic again with specific examples")

if __name__ == "__main__":
    main()