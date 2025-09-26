#!/usr/bin/env python3
"""
Trace Judge Validation Changes
Check if the judge consistency validation is changing A->B grades
"""

import json

def main():
    print("🔍 TRACING JUDGE VALIDATION CHANGES")
    print("=" * 60)
    print()

    # Look for validation changes in the judge logs
    validation_changes = []
    total_customgpt_evaluations = 0

    with open('results/run_20250926_033923_770/judge_evaluations.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                eval_data = json.loads(line)
                if eval_data['metadata']['real_provider_name'] == 'CustomGPT_RAG':
                    total_customgpt_evaluations += 1

                    # Check for validation changes in the judge data
                    judge = eval_data.get('judge', {})
                    validation = judge.get('validation', {})

                    if validation:
                        original_grade = validation.get('original_grade')
                        final_grade = validation.get('final_grade') or validation.get('suggested_grade')
                        validation_passed = validation.get('validation_passed', True)
                        inconsistency_type = validation.get('inconsistency_type')

                        if original_grade and final_grade and original_grade != final_grade:
                            validation_changes.append({
                                'question_id': eval_data['question_id'],
                                'original_grade': original_grade,
                                'final_grade': final_grade,
                                'validation_passed': validation_passed,
                                'inconsistency_type': inconsistency_type
                            })

    print(f"Total CustomGPT evaluations: {total_customgpt_evaluations}")
    print(f"Validation changes found: {len(validation_changes)}")
    print()

    if validation_changes:
        print("📝 VALIDATION CHANGES:")
        a_to_b_changes = [c for c in validation_changes if c['original_grade'] == 'A' and c['final_grade'] == 'B']

        print(f"Total changes: {len(validation_changes)}")
        print(f"A->B changes: {len(a_to_b_changes)}")

        if len(a_to_b_changes) == 84:
            print("✅ FOUND THE BUG! 84 A grades changed to B by validation")
        elif len(a_to_b_changes) > 0:
            print(f"🔍 Found {len(a_to_b_changes)} A->B changes, investigating...")

        print()
        print("Sample validation changes:")
        for i, change in enumerate(validation_changes[:10], 1):
            print(f"{i:2d}. Q{change['question_id']}: {change['original_grade']} -> {change['final_grade']}")
            print(f"     Reason: {change['inconsistency_type']}")
            print(f"     Validation passed: {change['validation_passed']}")
            print()
    else:
        print("❓ No validation changes found in judge data")
        print()
        print("Maybe the validation data is stored elsewhere or the bug is different...")

    print()
    print("🎯 ALTERNATIVE HYPOTHESIS:")
    print("The bug might be in the aggregation itself - let me check if there's a logic error")
    print("where some evaluations are being double-counted or miscategorized")

if __name__ == "__main__":
    main()