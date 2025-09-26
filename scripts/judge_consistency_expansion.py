#!/usr/bin/env python3
"""
Judge Consistency Expansion Analysis
Analyze the inconsistent case and validate controversial decisions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import random
from typing import Dict, List, Tuple, Any

def load_consistency_validation(consistency_file: str) -> List[Dict]:
    """Load judge consistency validation from JSONL file."""
    validations = []
    with open(consistency_file, 'r') as f:
        for line in f:
            if line.strip():
                validations.append(json.loads(line))
    return validations

def load_judge_evaluations(judge_file: str) -> List[Dict]:
    """Load judge evaluations from JSONL file."""
    evaluations = []
    with open(judge_file, 'r') as f:
        for line in f:
            if line.strip():
                evaluations.append(json.loads(line))
    return evaluations

def analyze_inconsistent_cases(validations: List[Dict]) -> Dict[str, Any]:
    """Analyze the inconsistent cases from judge consistency validation."""

    inconsistent_analysis = {
        'total_validations': len(validations),
        'total_responses_tested': 0,
        'consistent_responses': 0,
        'inconsistent_responses': 0,
        'inconsistent_cases': [],
        'consistency_rates_by_run': [],
        'overall_consistency_rate': 0.0
    }

    for validation in validations:
        if 'summary' in validation:
            summary = validation['summary']
            inconsistent_analysis['total_responses_tested'] += summary.get('total_responses_tested', 0)
            inconsistent_analysis['consistent_responses'] += summary.get('consistent_responses', 0)
            inconsistent_analysis['inconsistent_responses'] += summary.get('inconsistent_responses', 0)
            inconsistent_analysis['consistency_rates_by_run'].append(summary.get('consistency_rate', 0.0))

            # Extract detailed inconsistent cases
            if 'detailed_results' in summary:
                for result in summary['detailed_results']:
                    if not result.get('is_consistent', True):
                        inconsistent_case = {
                            'timestamp': validation.get('timestamp'),
                            'question_id': result['question_id'],
                            'question': result['question'],
                            'target': result['target'],
                            'predicted_answer': result['predicted_answer'],
                            'evaluations': result['evaluations'],
                            'unique_grades': result['unique_grades'],
                            'grade_distribution': result['grade_distribution']
                        }
                        inconsistent_analysis['inconsistent_cases'].append(inconsistent_case)

    # Calculate overall consistency rate
    total_tested = inconsistent_analysis['total_responses_tested']
    total_consistent = inconsistent_analysis['consistent_responses']
    inconsistent_analysis['overall_consistency_rate'] = total_consistent / total_tested if total_tested > 0 else 0.0

    return inconsistent_analysis

def identify_controversial_decisions(evaluations: List[Dict], confidence_threshold: float = 0.7) -> List[Dict]:
    """Identify controversial decisions based on low judge confidence."""

    controversial_decisions = []

    for eval_data in evaluations:
        try:
            # Extract judge response
            judge_response = json.loads(eval_data['judge']['response'])
            confidence = judge_response.get('confidence', 1.0)
            grade = judge_response['grade']

            # Only include main provider evaluations (not consistency tests)
            provider = eval_data['metadata']['real_provider_name']
            if 'ConsistencyTest' in provider:
                continue

            # Identify low-confidence (controversial) decisions
            if confidence <= confidence_threshold:
                controversial_decision = {
                    'question_id': eval_data['question_id'],
                    'question': eval_data['question'],
                    'target_answer': eval_data['target_answer'],
                    'predicted_answer': list(eval_data['provider_responses'].values())[0],
                    'provider': provider,
                    'grade': grade,
                    'confidence': confidence,
                    'reasoning': judge_response.get('reasoning', ''),
                    'timestamp': eval_data['timestamp']
                }
                controversial_decisions.append(controversial_decision)

        except Exception as e:
            continue

    # Sort by confidence (lowest first)
    controversial_decisions.sort(key=lambda x: x['confidence'])

    return controversial_decisions

def generate_consistency_expansion_report(inconsistent_analysis: Dict[str, Any],
                                        controversial_decisions: List[Dict],
                                        total_evaluations: int = 3000) -> str:
    """Generate comprehensive judge consistency expansion report."""

    report = []
    report.append("# JUDGE CONSISTENCY EXPANSION ANALYSIS")
    report.append("## Detailed Analysis of Inconsistent Cases and Controversial Decisions")
    report.append("=" * 80)
    report.append("")

    # Overall consistency summary
    report.append("## JUDGE CONSISTENCY SUMMARY")
    report.append("")

    total_tested = inconsistent_analysis['total_responses_tested']
    total_consistent = inconsistent_analysis['consistent_responses']
    total_inconsistent = inconsistent_analysis['inconsistent_responses']
    consistency_rate = inconsistent_analysis['overall_consistency_rate']

    report.append(f"**Overall Judge Consistency Performance:**")
    report.append(f"- Total responses tested: {total_tested}")
    report.append(f"- Consistent responses: {total_consistent}")
    report.append(f"- Inconsistent responses: {total_inconsistent}")
    report.append(f"- **Overall consistency rate: {consistency_rate:.1%}**")
    report.append("")

    # Consistency by validation run
    if inconsistent_analysis['consistency_rates_by_run']:
        rates = inconsistent_analysis['consistency_rates_by_run']
        report.append(f"**Consistency Rates by Validation Run:**")
        for i, rate in enumerate(rates, 1):
            report.append(f"- Run {i}: {rate:.1%}")
        report.append(f"- Mean: {np.mean(rates):.1%}")
        report.append(f"- Std Dev: {np.std(rates):.1%}")
        report.append("")

    # Detailed inconsistent case analysis
    report.append("## INCONSISTENT CASE ANALYSIS")
    report.append("")

    if inconsistent_analysis['inconsistent_cases']:
        report.append(f"**Total Inconsistent Cases Found: {len(inconsistent_analysis['inconsistent_cases'])}**")
        report.append("")

        for i, case in enumerate(inconsistent_analysis['inconsistent_cases'], 1):
            report.append(f"### Inconsistent Case #{i}")
            report.append(f"**Question ID:** {case['question_id']}")
            report.append(f"**Question:** {case['question']}")
            report.append(f"**Target Answer:** {case['target']}")
            report.append(f"**Predicted Answer:** {case['predicted_answer']}")
            report.append("")

            report.append("**Judge Evaluation Inconsistency:**")
            for j, evaluation in enumerate(case['evaluations'], 1):
                report.append(f"- Run {j}: Grade {evaluation['grade']} (Latency: {evaluation['latency_ms']:.0f}ms)")

            report.append(f"**Grade Distribution:** {case['grade_distribution']}")
            report.append(f"**Unique Grades:** {case['unique_grades']}")
            report.append("")

            # Analysis of why inconsistency occurred
            report.append("**Potential Causes of Inconsistency:**")
            grades = case['unique_grades']
            if 'A' in grades and 'B' in grades:
                report.append("- Judge uncertainty about correctness boundary")
                report.append("- Potential ambiguity in question or answer format")
                report.append("- Edge case requiring human expert review")
            report.append("")
    else:
        report.append("**No inconsistent cases found - Judge performance is highly reliable.**")
        report.append("")

    # Controversial decisions analysis
    report.append("## CONTROVERSIAL DECISIONS ANALYSIS")
    report.append("")

    if controversial_decisions:
        report.append(f"**Total Low-Confidence Decisions Found: {len(controversial_decisions)}**")
        report.append("")

        # Group by confidence ranges
        confidence_ranges = {
            'Very Low (0.0-0.5)': [d for d in controversial_decisions if d['confidence'] <= 0.5],
            'Low (0.5-0.6)': [d for d in controversial_decisions if 0.5 < d['confidence'] <= 0.6],
            'Moderate (0.6-0.7)': [d for d in controversial_decisions if 0.6 < d['confidence'] <= 0.7]
        }

        for range_name, decisions in confidence_ranges.items():
            if decisions:
                report.append(f"### {range_name} Confidence Decisions ({len(decisions)})")
                report.append("")

                # Show top 5 most controversial in each range
                for i, decision in enumerate(decisions[:5], 1):
                    report.append(f"**{i}. Question ID:** {decision['question_id']}")
                    report.append(f"   **Confidence:** {decision['confidence']:.2f}")
                    report.append(f"   **Grade:** {decision['grade']}")
                    report.append(f"   **Provider:** {decision['provider']}")
                    report.append(f"   **Question:** {decision['question'][:100]}...")
                    report.append(f"   **Target:** {decision['target_answer']}")
                    report.append(f"   **Predicted:** {decision['predicted_answer'][:100]}...")
                    report.append("")

                if len(decisions) > 5:
                    report.append(f"   ... and {len(decisions) - 5} more decisions in this range")
                    report.append("")

        # Provider breakdown for controversial decisions
        provider_controversial = {}
        for decision in controversial_decisions:
            provider = decision['provider']
            if provider not in provider_controversial:
                provider_controversial[provider] = []
            provider_controversial[provider].append(decision)

        report.append("### Controversial Decisions by Provider")
        report.append("")
        for provider, decisions in provider_controversial.items():
            avg_confidence = np.mean([d['confidence'] for d in decisions])
            report.append(f"**{provider}:** {len(decisions)} decisions (avg confidence: {avg_confidence:.2f})")
        report.append("")

    else:
        report.append("**No controversial decisions found - All judge decisions had high confidence.**")
        report.append("")

    # Key insights and recommendations
    report.append("## KEY INSIGHTS")
    report.append("")

    if consistency_rate >= 0.95:
        report.append(f"1. **Excellent judge consistency** ({consistency_rate:.1%}) meets academic standards")
    elif consistency_rate >= 0.90:
        report.append(f"1. **Good judge consistency** ({consistency_rate:.1%}) acceptable for most applications")
    else:
        report.append(f"1. **Judge consistency concerns** ({consistency_rate:.1%}) - needs investigation")

    if inconsistent_analysis['inconsistent_cases']:
        report.append(f"2. **{len(inconsistent_analysis['inconsistent_cases'])} inconsistent cases** require human expert review")
    else:
        report.append("2. **Zero inconsistent cases** - judge reliability excellent")

    if controversial_decisions:
        controversy_rate = len(controversial_decisions) / total_evaluations
        report.append(f"3. **{len(controversial_decisions)} controversial decisions** ({controversy_rate:.1%} of total)")

        # Most problematic provider
        if provider_controversial:
            most_controversial = max(provider_controversial.items(), key=lambda x: len(x[1]))
            report.append(f"4. **{most_controversial[0]}** has most controversial decisions ({len(most_controversial[1])})")
    else:
        report.append("3. **No controversial decisions** - all judges confident")

    report.append("")

    # Recommendations
    report.append("## RECOMMENDATIONS")
    report.append("")

    if consistency_rate < 0.95:
        report.append("### Judge Reliability Improvements")
        report.append("- Implement multi-judge evaluation for controversial cases")
        report.append("- Refine grading criteria for edge cases")
        report.append("- Add human expert validation for inconsistent cases")
        report.append("")

    if controversial_decisions:
        report.append("### Controversial Decision Handling")
        report.append("- Manual expert review for confidence < 0.6")
        report.append("- Multi-judge consensus for confidence < 0.7")
        report.append("- Improved grading rubric for ambiguous cases")
        report.append("")

    report.append("### External Validation")
    report.append("- Human expert panel for dataset quality validation")
    report.append("- Cross-validation with alternative judge models")
    report.append("- Domain expert review for technical questions")

    return "\\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Judge Consistency Expansion Analysis')
    parser.add_argument('--consistency-file', required=True, help='Path to judge consistency validation JSONL')
    parser.add_argument('--judge-file', required=True, help='Path to judge evaluations JSONL')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold for controversial decisions')

    args = parser.parse_args()

    # Load data
    print("Loading judge consistency validation...")
    validations = load_consistency_validation(args.consistency_file)
    print(f"Loaded {len(validations)} validation runs")

    print("Loading judge evaluations...")
    evaluations = load_judge_evaluations(args.judge_file)
    print(f"Loaded {len(evaluations)} evaluations")

    # Analyze inconsistent cases
    print("Analyzing inconsistent cases...")
    inconsistent_analysis = analyze_inconsistent_cases(validations)

    # Identify controversial decisions
    print(f"Identifying controversial decisions (confidence <= {args.confidence_threshold})...")
    controversial_decisions = identify_controversial_decisions(evaluations, args.confidence_threshold)

    # Generate report
    main_evaluations = len([e for e in evaluations if 'ConsistencyTest' not in e['metadata']['real_provider_name']])
    report_content = generate_consistency_expansion_report(inconsistent_analysis, controversial_decisions, main_evaluations)

    # Save results
    output_dir = Path(args.output_dir)

    # Save JSON analysis
    analysis_output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_type': 'judge_consistency_expansion',
        'inconsistent_analysis': inconsistent_analysis,
        'controversial_decisions': controversial_decisions[:50],  # Limit for JSON size
        'summary': {
            'overall_consistency_rate': inconsistent_analysis['overall_consistency_rate'],
            'total_inconsistent_cases': len(inconsistent_analysis['inconsistent_cases']),
            'total_controversial_decisions': len(controversial_decisions),
            'judge_reliability_assessment': 'EXCELLENT' if inconsistent_analysis['overall_consistency_rate'] >= 0.95 else
                                          'GOOD' if inconsistent_analysis['overall_consistency_rate'] >= 0.90 else 'NEEDS_IMPROVEMENT'
        }
    }

    json_file = output_dir / "judge_consistency_expansion.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)

    # Save report
    report_file = output_dir / "judge_consistency_expansion_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\\n=== JUDGE CONSISTENCY EXPANSION COMPLETE ===")
    print(f"JSON results: {json_file}")
    print(f"Report: {report_file}")

    # Print key findings
    print(f"\\n=== KEY FINDINGS ===")
    print(f"Overall consistency rate: {inconsistent_analysis['overall_consistency_rate']:.1%}")
    print(f"Inconsistent cases: {len(inconsistent_analysis['inconsistent_cases'])}")
    print(f"Controversial decisions: {len(controversial_decisions)}")
    print(f"Judge reliability: {analysis_output['summary']['judge_reliability_assessment']}")

if __name__ == "__main__":
    main()