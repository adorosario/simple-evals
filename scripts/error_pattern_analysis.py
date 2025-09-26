#!/usr/bin/env python3
"""
Error Pattern Analysis for 1000-Example RAG Benchmark Results
Comprehensive categorization and analysis of all incorrect responses.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Any

def load_judge_evaluations(judge_file: str) -> List[Dict]:
    """Load judge evaluations from JSONL file."""
    evaluations = []
    with open(judge_file, 'r') as f:
        for line in f:
            if line.strip():
                evaluations.append(json.loads(line))
    return evaluations

def categorize_error_type(question: str, predicted_answer: str, target: str, grade: str, reasoning: str) -> str:
    """Categorize the type of error based on content analysis."""

    # Clean and normalize text
    predicted_lower = predicted_answer.lower()
    target_lower = target.lower()
    question_lower = question.lower()
    reasoning_lower = reasoning.lower()

    # Since grade B = incorrect in our system, analyze the type of error
    if grade == 'B':
        # Check for explicit refusal/inability
        if "i don't" in predicted_lower or "i cannot" in predicted_lower or "i'm not" in predicted_lower or "not sure" in predicted_lower:
            return "EXPLICIT_REFUSAL"

        # Check for very short/insufficient responses
        elif len(predicted_answer.strip()) < 10:
            return "INSUFFICIENT_RESPONSE"

        # Check for uncertainty indicators
        elif any(word in predicted_lower for word in ["unclear", "ambiguous", "uncertain", "maybe", "possibly"]):
            return "UNCERTAINTY_RESPONSE"

        # Question type analysis for better categorization
        elif any(word in question_lower for word in ['when', 'what year', 'what date']):
            return "TEMPORAL_ERROR"
        elif any(word in question_lower for word in ['where', 'which city', 'which country', 'which state']):
            return "GEOGRAPHICAL_ERROR"
        elif any(word in question_lower for word in ['who', 'which person', 'whose']):
            return "PERSON_IDENTIFICATION_ERROR"
        elif any(word in question_lower for word in ['how many', 'what number', 'how much']):
            return "QUANTITATIVE_ERROR"
        elif any(word in question_lower for word in ['what is', 'what was', 'which', 'name']):
            return "FACTUAL_ERROR"

        # Check for numerical mismatches
        predicted_numbers = re.findall(r'\d+', predicted_answer)
        target_numbers = re.findall(r'\d+', target)
        if predicted_numbers and target_numbers and predicted_numbers != target_numbers:
            return "NUMERICAL_MISMATCH"

        # Check for date/year mismatches specifically
        if any(year in predicted_answer for year in ['19', '20']) and any(year in target for year in ['19', '20']):
            return "DATE_YEAR_ERROR"

        # Analyze reasoning for more context
        if "contradict" in reasoning_lower or "mismatch" in reasoning_lower:
            return "DIRECT_CONTRADICTION"
        elif "partial" in reasoning_lower or "incomplete" in reasoning_lower:
            return "PARTIAL_INFORMATION_ERROR"
        elif "wrong" in reasoning_lower or "incorrect" in reasoning_lower:
            return "FACTUAL_INACCURACY"

        # Default factual error
        return "GENERAL_FACTUAL_ERROR"

    else:
        return "OTHER_ERROR"

def analyze_error_patterns(evaluations: List[Dict]) -> Dict[str, Any]:
    """Comprehensive error pattern analysis."""

    # Organize by provider
    provider_errors = defaultdict(list)
    provider_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'by_grade': Counter()})

    for eval_data in evaluations:
        # Extract provider info from metadata
        provider = eval_data['metadata']['real_provider_name']

        # Extract grade from judge response
        try:
            judge_response = json.loads(eval_data['judge']['response'])
            grade = judge_response['grade']
            confidence = judge_response.get('confidence', None)
            reasoning = judge_response.get('reasoning', '')
        except:
            continue

        # Get predicted answer (first provider response)
        predicted_answer = list(eval_data['provider_responses'].values())[0]

        provider_stats[provider]['total'] += 1
        provider_stats[provider]['by_grade'][grade] += 1

        # Track errors (grade B is incorrect in our system)
        if grade == 'B':  # In our system, A=correct, B=incorrect
            provider_stats[provider]['errors'] += 1

            error_type = categorize_error_type(
                eval_data['question'],
                predicted_answer,
                eval_data['target_answer'],
                grade,
                reasoning
            )

            error_record = {
                'question_id': eval_data['question_id'],
                'question': eval_data['question'],
                'predicted_answer': predicted_answer,
                'target': eval_data['target_answer'],
                'grade': grade,
                'error_type': error_type,
                'reasoning': reasoning,
                'confidence': confidence
            }
            provider_errors[provider].append(error_record)

    # Analyze error patterns
    analysis = {
        'provider_error_summary': {},
        'error_type_analysis': {},
        'grade_distribution': {},
        'domain_analysis': {},
        'detailed_errors': provider_errors
    }

    # Provider error summary
    for provider, stats in provider_stats.items():
        error_rate = stats['errors'] / stats['total'] if stats['total'] > 0 else 0
        analysis['provider_error_summary'][provider] = {
            'total_responses': stats['total'],
            'total_errors': stats['errors'],
            'error_rate': error_rate,
            'accuracy': 1 - error_rate,
            'grade_distribution': dict(stats['by_grade'])
        }

    # Error type analysis by provider
    for provider, errors in provider_errors.items():
        error_types = Counter([error['error_type'] for error in errors])
        analysis['error_type_analysis'][provider] = {
            'error_type_counts': dict(error_types),
            'most_common_errors': error_types.most_common(5),
            'total_errors': len(errors)
        }

    # Grade distribution analysis
    for provider, stats in provider_stats.items():
        total = stats['total']
        analysis['grade_distribution'][provider] = {
            grade: {'count': count, 'percentage': count/total if total > 0 else 0}
            for grade, count in stats['by_grade'].items()
        }

    return analysis

def identify_systematic_failures(error_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Identify systematic failure patterns across providers."""

    systematic_patterns = {
        'common_failure_questions': {},
        'provider_specific_weaknesses': {},
        'error_severity_analysis': {},
        'abstention_effectiveness': {}
    }

    # Find questions that multiple providers got wrong
    question_failures = defaultdict(list)

    for provider, errors in error_analysis['detailed_errors'].items():
        for error in errors:
            question_failures[error['question_id']].append({
                'provider': provider,
                'grade': error['grade'],
                'error_type': error['error_type']
            })

    # Identify questions with multiple failures
    multi_failure_questions = {
        qid: failures for qid, failures in question_failures.items()
        if len(failures) > 1
    }

    systematic_patterns['common_failure_questions'] = {
        'total_multi_failure_questions': len(multi_failure_questions),
        'questions': multi_failure_questions
    }

    # Provider-specific weakness analysis
    for provider, error_data in error_analysis['error_type_analysis'].items():
        error_types = error_data['error_type_counts']
        total_errors = error_data['total_errors']

        # Calculate error type percentages
        error_percentages = {
            error_type: count/total_errors if total_errors > 0 else 0
            for error_type, count in error_types.items()
        }

        systematic_patterns['provider_specific_weaknesses'][provider] = {
            'primary_weakness': max(error_percentages.items(), key=lambda x: x[1]) if error_percentages else None,
            'error_type_percentages': error_percentages,
            'total_errors': total_errors
        }

    # Error severity analysis (for A/B grading system)
    for provider, summary in error_analysis['provider_error_summary'].items():
        grade_dist = summary['grade_distribution']

        # In A/B system: A=correct, B=incorrect
        b_grade_count = grade_dist.get('B', 0)
        total_responses = summary['total_responses']

        systematic_patterns['error_severity_analysis'][provider] = {
            'error_count': b_grade_count,
            'error_rate': b_grade_count / total_responses if total_responses > 0 else 0,
            'accuracy_rate': grade_dist.get('A', 0) / total_responses if total_responses > 0 else 0,
            'total_responses': total_responses
        }

    return systematic_patterns

def generate_error_analysis_report(error_analysis: Dict[str, Any], systematic_patterns: Dict[str, Any]) -> str:
    """Generate comprehensive error analysis report."""

    report = []
    report.append("# COMPREHENSIVE ERROR PATTERN ANALYSIS")
    report.append("## 1000-Example RAG Benchmark - Detailed Failure Mode Investigation")
    report.append("=" * 70)
    report.append("")

    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("")

    # Sort providers by error rate
    provider_summary = error_analysis['provider_error_summary']
    sorted_providers = sorted(provider_summary.items(), key=lambda x: x[1]['error_rate'])

    for provider, stats in sorted_providers:
        report.append(f"**{provider}**:")
        report.append(f"- Error Rate: {stats['error_rate']:.1%} ({stats['total_errors']}/{stats['total_responses']})")
        report.append(f"- Accuracy: {stats['accuracy']:.1%}")
        report.append("")

    # Error Type Analysis
    report.append("## ERROR TYPE BREAKDOWN BY PROVIDER")
    report.append("")

    for provider, error_data in error_analysis['error_type_analysis'].items():
        report.append(f"### {provider}")
        report.append(f"Total Errors: {error_data['total_errors']}")
        report.append("")

        if error_data['most_common_errors']:
            report.append("**Most Common Error Types:**")
            for error_type, count in error_data['most_common_errors']:
                percentage = count / error_data['total_errors'] * 100 if error_data['total_errors'] > 0 else 0
                report.append(f"1. {error_type}: {count} ({percentage:.1f}%)")
            report.append("")

    # Grade Distribution
    report.append("## GRADE DISTRIBUTION ANALYSIS")
    report.append("")

    for provider, grade_data in error_analysis['grade_distribution'].items():
        report.append(f"### {provider}")
        for grade in ['A', 'B']:  # Only A and B grades in our system
            if grade in grade_data:
                count = grade_data[grade]['count']
                percentage = grade_data[grade]['percentage'] * 100
                grade_label = "CORRECT" if grade == 'A' else "INCORRECT"
                report.append(f"- Grade {grade} ({grade_label}): {count} ({percentage:.1f}%)")
        report.append("")

    # Systematic Failure Analysis
    report.append("## SYSTEMATIC FAILURE PATTERNS")
    report.append("")

    # Common failure questions
    common_failures = systematic_patterns['common_failure_questions']
    report.append(f"**Questions with Multiple Provider Failures: {common_failures['total_multi_failure_questions']}**")
    report.append("")

    if common_failures['total_multi_failure_questions'] > 0:
        # Show top 5 most problematic questions
        question_failure_counts = {
            qid: len(failures)
            for qid, failures in common_failures['questions'].items()
        }
        top_problematic = sorted(question_failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        report.append("**Most Problematic Questions (Failed by Multiple Providers):**")
        for qid, failure_count in top_problematic:
            report.append(f"- {qid}: Failed by {failure_count} providers")
        report.append("")

    # Provider-specific weaknesses
    report.append("## PROVIDER-SPECIFIC WEAKNESS ANALYSIS")
    report.append("")

    for provider, weakness_data in systematic_patterns['provider_specific_weaknesses'].items():
        report.append(f"### {provider}")
        if weakness_data['primary_weakness']:
            weakness_type, percentage = weakness_data['primary_weakness']
            report.append(f"**Primary Weakness**: {weakness_type} ({percentage:.1%} of errors)")

        report.append("**Error Type Distribution:**")
        sorted_errors = sorted(weakness_data['error_type_percentages'].items(), key=lambda x: x[1], reverse=True)
        for error_type, percentage in sorted_errors[:3]:  # Top 3
            report.append(f"- {error_type}: {percentage:.1%}")
        report.append("")

    # Severity Analysis
    report.append("## ERROR RATE ANALYSIS")
    report.append("")

    severity_data = systematic_patterns['error_severity_analysis']
    sorted_severity = sorted(severity_data.items(), key=lambda x: x[1]['error_rate'])

    for provider, severity in sorted_severity:
        report.append(f"**{provider}**:")
        report.append(f"- Error Rate: {severity['error_rate']:.1%}")
        report.append(f"- Accuracy Rate: {severity['accuracy_rate']:.1%}")
        report.append(f"- Total Errors: {severity['error_count']}")
        report.append(f"- Total Responses: {severity['total_responses']}")
        report.append("")

    # Key Insights
    report.append("## KEY INSIGHTS")
    report.append("")

    best_performer = sorted_providers[0][0]
    worst_performer = sorted_providers[-1][0]

    report.append(f"1. **{best_performer}** has the lowest error rate ({sorted_providers[0][1]['error_rate']:.1%})")
    report.append(f"2. **{worst_performer}** has the highest error rate ({sorted_providers[-1][1]['error_rate']:.1%})")

    # Most common error type overall
    all_error_types = Counter()
    for provider_data in error_analysis['error_type_analysis'].values():
        for error_type, count in provider_data['error_type_counts'].items():
            all_error_types[error_type] += count

    if all_error_types:
        most_common_error = all_error_types.most_common(1)[0]
        report.append(f"3. **Most common error type across all providers**: {most_common_error[0]} ({most_common_error[1]} instances)")

    report.append(f"4. **{common_failures['total_multi_failure_questions']} questions failed by multiple providers** - indicating dataset difficulty")

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Error Pattern Analysis for RAG Benchmark')
    parser.add_argument('--judge-file', required=True, help='Path to judge evaluations JSONL file')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    args = parser.parse_args()

    # Load evaluations
    print("Loading judge evaluations...")
    evaluations = load_judge_evaluations(args.judge_file)
    print(f"Loaded {len(evaluations)} evaluations")

    # Perform error analysis
    print("Analyzing error patterns...")
    error_analysis = analyze_error_patterns(evaluations)

    # Identify systematic failures
    print("Identifying systematic failure patterns...")
    systematic_patterns = identify_systematic_failures(error_analysis)

    # Generate report
    report_content = generate_error_analysis_report(error_analysis, systematic_patterns)

    # Save results
    output_dir = Path(args.output_dir)

    # Save JSON results
    analysis_output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_type': 'error_pattern_analysis',
        'total_evaluations': len(evaluations),
        'error_analysis': error_analysis,
        'systematic_patterns': systematic_patterns,
        'summary': {
            'providers_analyzed': list(error_analysis['provider_error_summary'].keys()),
            'total_errors': sum(data['total_errors'] for data in error_analysis['provider_error_summary'].values()),
            'best_performer': min(error_analysis['provider_error_summary'].items(),
                                key=lambda x: x[1]['error_rate'])[0] if error_analysis['provider_error_summary'] else None
        }
    }

    json_file = output_dir / "error_pattern_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)

    # Save report
    report_file = output_dir / "error_pattern_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\n=== ERROR ANALYSIS COMPLETE ===")
    print(f"JSON results: {json_file}")
    print(f"Report: {report_file}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    for provider, stats in sorted(error_analysis['provider_error_summary'].items(),
                                key=lambda x: x[1]['error_rate']):
        print(f"{provider}: {stats['error_rate']:.1%} error rate ({stats['total_errors']} errors)")

if __name__ == "__main__":
    main()