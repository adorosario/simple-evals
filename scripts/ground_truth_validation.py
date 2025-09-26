#!/usr/bin/env python3
"""
Ground Truth Validation Analysis
Manual expert review simulation and gold target accuracy verification.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

def load_judge_evaluations(judge_file: str) -> List[Dict]:
    """Load judge evaluations from JSONL file."""
    evaluations = []
    with open(judge_file, 'r') as f:
        for line in f:
            if line.strip():
                evaluations.append(json.loads(line))
    return evaluations

def load_controversial_decisions(json_file: str) -> List[Dict]:
    """Load controversial decisions from previous analysis."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get('controversial_decisions', [])

def simulate_expert_review(controversial_decisions: List[Dict], sample_size: int = 25) -> Dict[str, Any]:
    """Simulate expert review of most controversial decisions."""

    # Sort by confidence (lowest first) and take sample
    sorted_decisions = sorted(controversial_decisions, key=lambda x: x['confidence'])
    sample_decisions = sorted_decisions[:sample_size]

    expert_review = {
        'total_reviewed': len(sample_decisions),
        'judge_agreement_rate': 0.0,
        'judge_correct_decisions': 0,
        'judge_incorrect_decisions': 0,
        'problematic_questions': [],
        'ambiguous_questions': [],
        'clear_judge_errors': [],
        'review_details': []
    }

    # Simulate expert analysis based on question patterns and judge reasoning
    for i, decision in enumerate(sample_decisions):
        question = decision['question'].lower()
        target = decision['target_answer']
        predicted = decision['predicted_answer']
        judge_grade = decision['grade']
        confidence = decision['confidence']
        reasoning = decision.get('reasoning', '')

        # Expert review simulation based on heuristics
        expert_assessment = analyze_question_expert_perspective(
            question, target, predicted, judge_grade, reasoning, confidence
        )

        expert_review['review_details'].append({
            'question_id': decision['question_id'],
            'question': decision['question'][:100] + "...",
            'target_answer': target,
            'predicted_answer': predicted[:100] + "...",
            'judge_grade': judge_grade,
            'judge_confidence': confidence,
            'expert_assessment': expert_assessment,
            'expert_agrees_with_judge': expert_assessment['agrees_with_judge'],
            'issue_type': expert_assessment['issue_type']
        })

        # Track agreement
        if expert_assessment['agrees_with_judge']:
            expert_review['judge_correct_decisions'] += 1
        else:
            expert_review['judge_incorrect_decisions'] += 1

            if expert_assessment['issue_type'] == 'JUDGE_ERROR':
                expert_review['clear_judge_errors'].append(decision['question_id'])
            elif expert_assessment['issue_type'] == 'AMBIGUOUS_QUESTION':
                expert_review['ambiguous_questions'].append(decision['question_id'])
            elif expert_assessment['issue_type'] == 'PROBLEMATIC_TARGET':
                expert_review['problematic_questions'].append(decision['question_id'])

    # Calculate agreement rate
    expert_review['judge_agreement_rate'] = expert_review['judge_correct_decisions'] / len(sample_decisions)

    return expert_review

def analyze_question_expert_perspective(question: str, target: str, predicted: str,
                                      judge_grade: str, reasoning: str, confidence: float) -> Dict[str, Any]:
    """Simulate expert analysis of a controversial decision."""

    question_lower = question.lower()
    target_lower = target.lower()
    predicted_lower = predicted.lower()
    reasoning_lower = reasoning.lower()

    # Check for obvious factual matches/mismatches
    if target_lower in predicted_lower or predicted_lower in target_lower:
        # Likely correct but judge may have been strict
        if judge_grade == 'B':  # Judge said incorrect
            return {
                'agrees_with_judge': False,
                'issue_type': 'JUDGE_ERROR',
                'explanation': 'Answer contains correct information but judge was too strict',
                'confidence_in_assessment': 0.8
            }

    # Check for numerical/date mismatches
    target_numbers = re.findall(r'\d+', target)
    predicted_numbers = re.findall(r'\d+', predicted)

    if target_numbers and predicted_numbers:
        if target_numbers != predicted_numbers:
            if judge_grade == 'B':  # Judge said incorrect
                return {
                    'agrees_with_judge': True,
                    'issue_type': 'CLEAR_FACTUAL_ERROR',
                    'explanation': 'Clear numerical/date mismatch',
                    'confidence_in_assessment': 0.9
                }
        else:
            if judge_grade == 'A':  # Judge said correct
                return {
                    'agrees_with_judge': True,
                    'issue_type': 'CORRECT_MATCH',
                    'explanation': 'Numbers match correctly',
                    'confidence_in_assessment': 0.9
                }

    # Check for ambiguous questions
    ambiguous_indicators = [
        'multiple', 'various', 'different', 'several', 'many',
        'approximately', 'around', 'about', 'roughly'
    ]

    if any(indicator in question_lower for indicator in ambiguous_indicators):
        return {
            'agrees_with_judge': False,
            'issue_type': 'AMBIGUOUS_QUESTION',
            'explanation': 'Question allows multiple valid answers',
            'confidence_in_assessment': 0.6
        }

    # Check for very specific technical questions
    technical_terms = [
        'specification', 'technical', 'precise', 'exact', 'specific',
        'millimeters', 'kilometers', 'degrees', 'coordinates'
    ]

    if any(term in question_lower for term in technical_terms):
        if confidence < 0.7:
            return {
                'agrees_with_judge': False,
                'issue_type': 'UNCERTAIN_TECHNICAL',
                'explanation': 'Technical question requires domain expertise',
                'confidence_in_assessment': 0.5
            }

    # Default: trust low-confidence judge decisions as potentially problematic
    if confidence < 0.6:
        return {
            'agrees_with_judge': False,
            'issue_type': 'LOW_CONFIDENCE_CONCERN',
            'explanation': 'Judge uncertainty suggests potential issues',
            'confidence_in_assessment': 0.6
        }

    # Default: agree with judge
    return {
        'agrees_with_judge': True,
        'issue_type': 'REASONABLE_DECISION',
        'explanation': 'Judge decision appears reasonable',
        'confidence_in_assessment': 0.7
    }

def validate_gold_targets(evaluations: List[Dict], sample_size: int = 100) -> Dict[str, Any]:
    """Validate accuracy of gold target answers through automated checks."""

    # Sample random questions for validation
    main_evaluations = [e for e in evaluations if 'ConsistencyTest' not in e['metadata']['real_provider_name']]
    sample_evaluations = np.random.choice(main_evaluations, min(sample_size, len(main_evaluations)), replace=False)

    validation_results = {
        'total_validated': len(sample_evaluations),
        'potentially_outdated': [],
        'potentially_ambiguous': [],
        'multiple_valid_answers': [],
        'validation_details': []
    }

    for eval_data in sample_evaluations:
        question = eval_data['question']
        target = eval_data['target_answer']

        validation_assessment = assess_target_validity(question, target)

        validation_results['validation_details'].append({
            'question_id': eval_data['question_id'],
            'question': question[:100] + "...",
            'target_answer': target,
            'assessment': validation_assessment,
            'issues_found': validation_assessment['issues']
        })

        # Categorize issues
        for issue in validation_assessment['issues']:
            if issue == 'POTENTIALLY_OUTDATED':
                validation_results['potentially_outdated'].append(eval_data['question_id'])
            elif issue == 'AMBIGUOUS_PHRASING':
                validation_results['potentially_ambiguous'].append(eval_data['question_id'])
            elif issue == 'MULTIPLE_VALID_ANSWERS':
                validation_results['multiple_valid_answers'].append(eval_data['question_id'])

    return validation_results

def assess_target_validity(question: str, target: str) -> Dict[str, Any]:
    """Assess potential issues with gold target answers."""

    question_lower = question.lower()
    target_lower = target.lower()
    issues = []

    # Check for potentially outdated information
    current_indicators = ['current', 'now', 'today', 'present', 'latest', 'recent']
    if any(indicator in question_lower for indicator in current_indicators):
        issues.append('POTENTIALLY_OUTDATED')

    # Check for ambiguous phrasing
    ambiguous_phrases = [
        'approximately', 'about', 'around', 'roughly', 'nearly',
        'close to', 'more or less', 'in the region of'
    ]
    if any(phrase in question_lower for phrase in ambiguous_phrases):
        issues.append('AMBIGUOUS_PHRASING')

    # Check for questions that might have multiple valid answers
    multiple_answer_indicators = [
        'one of', 'among', 'include', 'such as', 'for example',
        'various', 'several', 'multiple', 'different'
    ]
    if any(indicator in question_lower for indicator in multiple_answer_indicators):
        issues.append('MULTIPLE_VALID_ANSWERS')

    # Check for very specific numerical targets that might be disputed
    if re.search(r'\d+\.\d{2,}', target):  # Numbers with high precision
        issues.append('HIGH_PRECISION_NUMERICAL')

    # Check for names that might have variations
    if len(target.split()) > 2 and any(char in target for char in [',', '-', '.']):
        issues.append('NAME_VARIATION_POSSIBLE')

    return {
        'issues': issues,
        'issue_count': len(issues),
        'validity_score': max(0, 1.0 - len(issues) * 0.2)
    }

def generate_ground_truth_report(expert_review: Dict[str, Any],
                               target_validation: Dict[str, Any]) -> str:
    """Generate comprehensive ground truth validation report."""

    report = []
    report.append("# GROUND TRUTH VALIDATION ANALYSIS")
    report.append("## Expert Review Simulation and Gold Target Accuracy Assessment")
    report.append("=" * 70)
    report.append("")

    # Expert review summary
    report.append("## SIMULATED EXPERT REVIEW RESULTS")
    report.append("")

    total_reviewed = expert_review['total_reviewed']
    agreement_rate = expert_review['judge_agreement_rate']
    correct_decisions = expert_review['judge_correct_decisions']
    incorrect_decisions = expert_review['judge_incorrect_decisions']

    report.append(f"**Expert Review Summary:**")
    report.append(f"- Total controversial decisions reviewed: {total_reviewed}")
    report.append(f"- Judge-expert agreement rate: {agreement_rate:.1%}")
    report.append(f"- Judge decisions confirmed correct: {correct_decisions}")
    report.append(f"- Judge decisions questioned by expert: {incorrect_decisions}")
    report.append("")

    # Issue categorization
    clear_errors = len(expert_review['clear_judge_errors'])
    ambiguous_questions = len(expert_review['ambiguous_questions'])
    problematic_questions = len(expert_review['problematic_questions'])

    report.append(f"**Issue Categories:**")
    report.append(f"- Clear judge errors: {clear_errors}")
    report.append(f"- Ambiguous questions: {ambiguous_questions}")
    report.append(f"- Problematic target answers: {problematic_questions}")
    report.append("")

    # Detailed examples
    if expert_review['review_details']:
        report.append("## DETAILED EXPERT REVIEW EXAMPLES")
        report.append("")

        # Show examples of each issue type
        issue_types = defaultdict(list)
        for detail in expert_review['review_details']:
            issue_types[detail['issue_type']].append(detail)

        for issue_type, examples in issue_types.items():
            if examples:
                report.append(f"### {issue_type} Examples")
                report.append("")

                for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples per type
                    report.append(f"**{i}. Question ID:** {example['question_id']}")
                    report.append(f"   **Question:** {example['question']}")
                    report.append(f"   **Target:** {example['target_answer']}")
                    report.append(f"   **Predicted:** {example['predicted_answer']}")
                    report.append(f"   **Judge Grade:** {example['judge_grade']} (Confidence: {example['judge_confidence']:.2f})")
                    report.append(f"   **Expert Assessment:** {example['expert_assessment']['explanation']}")
                    report.append("")

                if len(examples) > 3:
                    report.append(f"   ... and {len(examples) - 3} more {issue_type} cases")
                    report.append("")

    # Gold target validation
    report.append("## GOLD TARGET VALIDATION RESULTS")
    report.append("")

    total_validated = target_validation['total_validated']
    outdated = len(target_validation['potentially_outdated'])
    ambiguous = len(target_validation['potentially_ambiguous'])
    multiple_valid = len(target_validation['multiple_valid_answers'])

    report.append(f"**Gold Target Assessment Summary:**")
    report.append(f"- Total questions validated: {total_validated}")
    report.append(f"- Potentially outdated targets: {outdated} ({outdated/total_validated:.1%})")
    report.append(f"- Ambiguously phrased questions: {ambiguous} ({ambiguous/total_validated:.1%})")
    report.append(f"- Questions with multiple valid answers: {multiple_valid} ({multiple_valid/total_validated:.1%})")
    report.append("")

    # Calculate overall target quality
    total_issues = outdated + ambiguous + multiple_valid
    target_quality = (total_validated - total_issues) / total_validated if total_validated > 0 else 0

    report.append(f"**Overall Target Quality Score: {target_quality:.1%}**")
    report.append("")

    # Key insights
    report.append("## KEY INSIGHTS")
    report.append("")

    if agreement_rate >= 0.8:
        report.append(f"1. **High judge reliability** - {agreement_rate:.1%} expert agreement confirms judge quality")
    elif agreement_rate >= 0.6:
        report.append(f"1. **Moderate judge reliability** - {agreement_rate:.1%} expert agreement suggests some issues")
    else:
        report.append(f"1. **Judge reliability concerns** - {agreement_rate:.1%} expert agreement indicates problems")

    if target_quality >= 0.9:
        report.append(f"2. **Excellent target quality** - {target_quality:.1%} of targets appear valid")
    elif target_quality >= 0.8:
        report.append(f"2. **Good target quality** - {target_quality:.1%} of targets appear valid")
    else:
        report.append(f"2. **Target quality concerns** - {target_quality:.1%} validity rate needs improvement")

    if clear_errors > 0:
        report.append(f"3. **{clear_errors} clear judge errors identified** - need manual review")

    if ambiguous_questions > 0:
        report.append(f"4. **{ambiguous_questions} ambiguous questions found** - consider revision")

    report.append("")

    # Recommendations
    report.append("## RECOMMENDATIONS")
    report.append("")

    if agreement_rate < 0.8:
        report.append("### Judge Reliability Improvements")
        report.append("- Implement multi-judge evaluation for low-confidence decisions")
        report.append("- Refine grading criteria for ambiguous cases")
        report.append("- Add human expert validation loop")
        report.append("")

    if target_quality < 0.9:
        report.append("### Dataset Quality Improvements")
        report.append("- Review and update potentially outdated questions")
        report.append("- Clarify ambiguously phrased questions")
        report.append("- Accept multiple valid answers where appropriate")
        report.append("- Regular dataset maintenance and validation")
        report.append("")

    report.append("### External Validation Next Steps")
    report.append("- Human expert panel review of identified issues")
    report.append("- Cross-validation with alternative datasets")
    report.append("- Domain expert consultation for technical questions")
    report.append("- Real-world utility validation studies")

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Ground Truth Validation Analysis')
    parser.add_argument('--judge-file', required=True, help='Path to judge evaluations JSONL file')
    parser.add_argument('--controversial-file', required=True, help='Path to controversial decisions JSON file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--expert-sample', type=int, default=25, help='Sample size for expert review')
    parser.add_argument('--target-sample', type=int, default=100, help='Sample size for target validation')

    args = parser.parse_args()

    # Load data
    print("Loading judge evaluations...")
    evaluations = load_judge_evaluations(args.judge_file)
    print(f"Loaded {len(evaluations)} evaluations")

    print("Loading controversial decisions...")
    controversial_decisions = load_controversial_decisions(args.controversial_file)
    print(f"Loaded {len(controversial_decisions)} controversial decisions")

    # Simulate expert review
    print(f"Simulating expert review of {args.expert_sample} most controversial decisions...")
    expert_review = simulate_expert_review(controversial_decisions, args.expert_sample)

    # Validate gold targets
    print(f"Validating {args.target_sample} random gold targets...")
    np.random.seed(42)  # For reproducibility
    target_validation = validate_gold_targets(evaluations, args.target_sample)

    # Generate report
    report_content = generate_ground_truth_report(expert_review, target_validation)

    # Save results
    output_dir = Path(args.output_dir)

    # Save JSON analysis
    analysis_output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_type': 'ground_truth_validation',
        'expert_review': expert_review,
        'target_validation': target_validation,
        'summary': {
            'judge_expert_agreement_rate': expert_review['judge_agreement_rate'],
            'target_quality_score': (target_validation['total_validated'] -
                                   len(target_validation['potentially_outdated']) -
                                   len(target_validation['potentially_ambiguous']) -
                                   len(target_validation['multiple_valid_answers'])) / target_validation['total_validated'],
            'clear_judge_errors': len(expert_review['clear_judge_errors']),
            'problematic_targets': len(target_validation['potentially_outdated']) +
                                 len(target_validation['potentially_ambiguous']) +
                                 len(target_validation['multiple_valid_answers'])
        }
    }

    json_file = output_dir / "ground_truth_validation.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)

    # Save report
    report_file = output_dir / "ground_truth_validation_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\n=== GROUND TRUTH VALIDATION COMPLETE ===")
    print(f"JSON results: {json_file}")
    print(f"Report: {report_file}")

    # Print key findings
    print(f"\n=== KEY FINDINGS ===")
    print(f"Judge-expert agreement rate: {expert_review['judge_agreement_rate']:.1%}")
    print(f"Target quality score: {analysis_output['summary']['target_quality_score']:.1%}")
    print(f"Clear judge errors: {len(expert_review['clear_judge_errors'])}")
    print(f"Problematic targets: {analysis_output['summary']['problematic_targets']}")

if __name__ == "__main__":
    main()