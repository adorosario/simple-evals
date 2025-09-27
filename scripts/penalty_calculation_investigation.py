#!/usr/bin/env python3
"""
Provider Penalty Calculation Investigation Script
Analyzes the actual penalty calculation mechanism for any provider.

CORRECTED UNDERSTANDING:
- Each incorrect answer (B grade) gets -4.0 quality score penalty
- Quality score = mean of individual quality scores per question
- Penalty ratio = 4.0 (configured in evaluation framework)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

def load_run_metadata(run_dir: str) -> Dict[str, Any]:
    """Load run metadata to get penalty information"""
    metadata_file = Path(run_dir) / "run_metadata.json"
    with open(metadata_file, 'r') as f:
        return json.load(f)

def load_judge_evaluations(run_dir: str) -> List[Dict[str, Any]]:
    """Load all judge evaluations for detailed analysis"""
    evaluations = []
    judge_file = Path(run_dir) / "judge_evaluations.jsonl"

    if judge_file.exists():
        with open(judge_file, 'r') as f:
            for line in f:
                if line.strip():
                    evaluations.append(json.loads(line))

    return evaluations

def get_provider_evaluations(evaluations: List[Dict], provider_name: str) -> List[Dict]:
    """Filter evaluations for specific provider"""
    provider_evals = []
    for eval_data in evaluations:
        real_provider = eval_data['metadata']['real_provider_name']
        if provider_name.lower() in real_provider.lower() and 'ConsistencyTest' not in real_provider:
            provider_evals.append(eval_data)

    return provider_evals

def analyze_penalty_calculation(run_dir: str, provider_name: str):
    """Investigate the penalty calculation mechanism for specified provider"""

    print(f"=== PENALTY CALCULATION INVESTIGATION ===")
    print(f"Run: {Path(run_dir).name}")
    print(f"Provider: {provider_name}")
    print("=" * 60)

    # Load data
    metadata = load_run_metadata(run_dir)
    evaluations = load_judge_evaluations(run_dir)

    # Find provider metrics
    provider_metrics = None
    sampler_name = None
    for result in metadata['results']['results']:
        if provider_name.lower() in result['sampler_name'].lower():
            provider_metrics = result['metrics']
            sampler_name = result['sampler_name']
            break

    if not provider_metrics:
        print(f"ERROR: {provider_name} metrics not found!")
        print("Available providers:")
        for result in metadata['results']['results']:
            print(f"  - {result['sampler_name']}")
        return

    # Get provider evaluations
    provider_evals = get_provider_evaluations(evaluations, provider_name)
    print(f"Found {len(provider_evals)} evaluations for {sampler_name}")

    # Extract key metrics
    volume_score = provider_metrics['volume_score']
    quality_score = provider_metrics['quality_score']
    n_correct = provider_metrics['n_correct']
    n_incorrect = provider_metrics['n_incorrect']
    n_abstained = provider_metrics['n_not_attempted']
    penalty_count = provider_metrics['overconfidence_penalty']  # This is just count, not penalty points
    penalty_ratio = provider_metrics['penalty_ratio']
    threshold = provider_metrics['threshold_value']

    # Analyze individual evaluations to understand quality score calculation
    correct_scores = []
    incorrect_scores = []
    abstained_scores = []

    for eval_data in provider_evals:
        try:
            judge_response = json.loads(eval_data['judge']['response'])
            grade = judge_response['grade']

            if grade == 'A':
                correct_scores.append(1.0)  # Correct = +1.0
            elif grade == 'B':
                incorrect_scores.append(-penalty_ratio)  # Incorrect = -4.0
            # Note: Abstentions (no judge evaluation) would get 0.0 but aren't in judge_evaluations.jsonl

        except Exception as e:
            print(f"Warning: Could not parse evaluation: {e}")
            continue

    # Add abstention scores (0.0 each)
    abstained_scores = [0.0] * n_abstained

    print("ACTUAL METRICS FROM RUN:")
    print(f"  Volume Score: {volume_score}")
    print(f"  Quality Score: {quality_score}")
    print(f"  Correct (A grades): {n_correct}")
    print(f"  Incorrect (B grades): {n_incorrect}")
    print(f"  Abstained: {n_abstained}")
    print(f"  Penalty Count (metadata): {penalty_count}")
    print(f"  Penalty Ratio: {penalty_ratio}")
    print(f"  Confidence Threshold: {threshold}")
    print()

    # Calculate quality score breakdown
    all_quality_scores = correct_scores + incorrect_scores + abstained_scores
    calculated_quality_score = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0

    print("CORRECTED PENALTY CALCULATION ANALYSIS:")
    print(f"  Individual Quality Scores:")
    print(f"    - {len(correct_scores)} correct answers × +1.0 = +{len(correct_scores)}")
    print(f"    - {len(incorrect_scores)} incorrect answers × -{penalty_ratio} = {sum(incorrect_scores)}")
    print(f"    - {len(abstained_scores)} abstentions × 0.0 = {sum(abstained_scores)}")
    print(f"  Total quality points: {sum(all_quality_scores)}")
    print(f"  Total questions: {len(all_quality_scores)}")
    print(f"  Calculated quality score: {calculated_quality_score:.3f}")
    print(f"  Reported quality score: {quality_score:.3f}")
    print(f"  Match: {'✓' if abs(calculated_quality_score - quality_score) < 0.001 else '✗'}")
    print()

    # Show penalty impact
    total_penalty_impact = sum(incorrect_scores)
    print("PENALTY IMPACT ANALYSIS:")
    print(f"  Total penalty from incorrect answers: {total_penalty_impact}")
    print(f"  Penalty per incorrect answer: -{penalty_ratio}")
    print(f"  Without penalties (volume score): {volume_score:.3f}")
    print(f"  With penalties (quality score): {quality_score:.3f}")
    print(f"  Net penalty impact per question: {total_penalty_impact / len(all_quality_scores):.3f}")
    print()

    # Quality score formula verification
    print("QUALITY SCORE FORMULA VERIFICATION:")
    print(f"  Expected formula: Quality = Mean of [+1.0 per correct, -{penalty_ratio} per incorrect, 0.0 per abstained]")
    print(f"  Calculated: {calculated_quality_score:.3f}")
    print(f"  Reported: {quality_score:.3f}")
    if abs(calculated_quality_score - quality_score) < 0.001:
        print(f"  ✓ Formula matches perfectly!")
    else:
        print(f"  ✗ Formula mismatch - difference: {abs(calculated_quality_score - quality_score):.6f}")
    print()

    # Show breakdown of failed evaluations
    if incorrect_scores:
        print("INCORRECT ANSWER BREAKDOWN:")
        incorrect_questions = []
        for eval_data in provider_evals:
            try:
                judge_response = json.loads(eval_data['judge']['response'])
                if judge_response['grade'] == 'B':
                    incorrect_questions.append({
                        'question_id': eval_data['question_id'],
                        'question': eval_data['question'][:100] + '...' if len(eval_data['question']) > 100 else eval_data['question']
                    })
            except:
                continue

        for i, q in enumerate(incorrect_questions, 1):
            print(f"  {i}. {q['question_id']}: {q['question']}")
        print()

    # Conclusion
    print("CONCLUSION:")
    print(f"  ✓ Each B grade (incorrect answer) contributes -{penalty_ratio} to quality score")
    print(f"  ✓ Quality score = mean of individual question scores")
    print(f"  ✓ {provider_name} lost {abs(total_penalty_impact):.1f} total penalty points from {n_incorrect} incorrect answers")
    print(f"  → Focus on fixing the {n_incorrect} incorrect answers to improve quality score")

def main():
    parser = argparse.ArgumentParser(description='Investigate penalty calculation mechanism')
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--provider', required=True, help='Provider name to analyze (e.g. customgpt, openai_rag, openai_vanilla)')

    args = parser.parse_args()

    analyze_penalty_calculation(args.run_dir, args.provider)

if __name__ == "__main__":
    main()