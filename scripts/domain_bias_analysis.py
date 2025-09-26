#!/usr/bin/env python3
"""
Domain Bias Analysis for RAG Benchmark
Analyze provider performance across different knowledge domains.
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

def categorize_question_domain(question: str, target_answer: str) -> str:
    """Categorize questions by knowledge domain based on content analysis."""

    question_lower = question.lower()
    target_lower = target_answer.lower()

    # Science and Technology
    if any(term in question_lower for term in [
        'molecule', 'chemical', 'protein', 'atom', 'compound', 'element',
        'physics', 'biology', 'chemistry', 'scientific', 'experiment',
        'research', 'laboratory', 'medical', 'disease', 'drug', 'pharmaceutical',
        'engineering', 'technology', 'computer', 'software', 'satellite', 'space'
    ]):
        return "SCIENCE_TECHNOLOGY"

    # History and Politics
    elif any(term in question_lower for term in [
        'war', 'battle', 'president', 'king', 'queen', 'emperor', 'dynasty',
        'revolution', 'independence', 'treaty', 'empire', 'colonial',
        'historical', 'ancient', 'medieval', 'century', 'founded', 'established',
        'politician', 'government', 'parliament', 'congress', 'minister'
    ]) or any(year in question for year in ['19', '18', '17', '16']):
        return "HISTORY_POLITICS"

    # Geography and Places
    elif any(term in question_lower for term in [
        'city', 'country', 'state', 'province', 'capital', 'continent',
        'mountain', 'river', 'ocean', 'island', 'located', 'situated',
        'geography', 'region', 'territory', 'border', 'population'
    ]):
        return "GEOGRAPHY_PLACES"

    # Entertainment and Media
    elif any(term in question_lower for term in [
        'movie', 'film', 'actor', 'actress', 'director', 'show', 'series',
        'television', 'tv', 'episode', 'season', 'character', 'netflix',
        'hollywood', 'oscar', 'award', 'album', 'song', 'musician', 'singer',
        'band', 'music', 'concert', 'album', 'novel', 'author', 'book'
    ]):
        return "ENTERTAINMENT_MEDIA"

    # Sports and Athletics
    elif any(term in question_lower for term in [
        'sport', 'game', 'team', 'player', 'athlete', 'championship',
        'olympic', 'football', 'basketball', 'soccer', 'baseball', 'tennis',
        'golf', 'swimming', 'coach', 'stadium', 'league', 'match', 'tournament'
    ]):
        return "SPORTS_ATHLETICS"

    # Business and Economics
    elif any(term in question_lower for term in [
        'company', 'corporation', 'business', 'ceo', 'founder', 'industry',
        'market', 'stock', 'economy', 'financial', 'bank', 'investment',
        'profit', 'revenue', 'brand', 'product', 'service', 'retail'
    ]):
        return "BUSINESS_ECONOMICS"

    # Arts and Culture
    elif any(term in question_lower for term in [
        'art', 'artist', 'painting', 'sculpture', 'museum', 'gallery',
        'culture', 'cultural', 'tradition', 'festival', 'religion',
        'philosophy', 'literature', 'poetry', 'theater', 'dance'
    ]):
        return "ARTS_CULTURE"

    # Nature and Environment
    elif any(term in question_lower for term in [
        'animal', 'species', 'wildlife', 'forest', 'nature', 'environment',
        'climate', 'weather', 'plant', 'tree', 'ecosystem', 'conservation'
    ]):
        return "NATURE_ENVIRONMENT"

    # Technology and Computing
    elif any(term in question_lower for term in [
        'computer', 'software', 'internet', 'website', 'digital', 'online',
        'algorithm', 'programming', 'code', 'data', 'artificial intelligence'
    ]):
        return "TECHNOLOGY_COMPUTING"

    # Dates and Numbers (for questions primarily about when/how much)
    elif any(term in question_lower for term in [
        'when', 'what year', 'what date', 'how many', 'what number',
        'how much', 'what time', 'which year'
    ]) and not any(other_domain in question_lower for other_domain in [
        'movie', 'sport', 'war', 'city', 'company'
    ]):
        return "TEMPORAL_QUANTITATIVE"

    # General Knowledge (fallback)
    else:
        return "GENERAL_KNOWLEDGE"

def analyze_domain_performance(evaluations: List[Dict]) -> Dict[str, Any]:
    """Analyze provider performance across different domains."""

    domain_analysis = {
        'domain_distribution': Counter(),
        'provider_domain_performance': defaultdict(lambda: defaultdict(lambda: {
            'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0.0
        })),
        'domain_difficulty': {},
        'provider_domain_strengths': {},
        'provider_domain_weaknesses': {}
    }

    # Process all evaluations
    for eval_data in evaluations:
        # Skip consistency tests
        provider = eval_data['metadata']['real_provider_name']
        if 'ConsistencyTest' in provider:
            continue

        # Extract evaluation data
        try:
            judge_response = json.loads(eval_data['judge']['response'])
            grade = judge_response['grade']
            question = eval_data['question']
            target = eval_data['target_answer']

            # Categorize by domain
            domain = categorize_question_domain(question, target)
            domain_analysis['domain_distribution'][domain] += 1

            # Track provider performance by domain
            domain_stats = domain_analysis['provider_domain_performance'][provider][domain]
            domain_stats['total'] += 1

            if grade == 'A':  # Correct
                domain_stats['correct'] += 1
            else:  # Incorrect
                domain_stats['incorrect'] += 1

        except Exception:
            continue

    # Calculate accuracy rates
    for provider, domains in domain_analysis['provider_domain_performance'].items():
        for domain, stats in domains.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']

    # Calculate domain difficulty (overall accuracy across all providers)
    for domain in domain_analysis['domain_distribution'].keys():
        total_questions = 0
        correct_answers = 0

        for provider, domains in domain_analysis['provider_domain_performance'].items():
            if domain in domains:
                total_questions += domains[domain]['total']
                correct_answers += domains[domain]['correct']

        if total_questions > 0:
            domain_analysis['domain_difficulty'][domain] = {
                'total_questions': total_questions,
                'overall_accuracy': correct_answers / total_questions,
                'difficulty_rank': 0  # Will be set later
            }

    # Rank domains by difficulty
    sorted_domains = sorted(
        domain_analysis['domain_difficulty'].items(),
        key=lambda x: x[1]['overall_accuracy']
    )

    for rank, (domain, stats) in enumerate(sorted_domains, 1):
        stats['difficulty_rank'] = rank

    # Identify provider strengths and weaknesses
    for provider, domains in domain_analysis['provider_domain_performance'].items():
        provider_accuracies = {domain: stats['accuracy'] for domain, stats in domains.items() if stats['total'] >= 3}

        if provider_accuracies:
            # Calculate provider's average accuracy
            avg_accuracy = np.mean(list(provider_accuracies.values()))

            # Identify strengths (significantly above average)
            strengths = {domain: acc for domain, acc in provider_accuracies.items()
                        if acc > avg_accuracy + 0.1 and acc > 0.7}

            # Identify weaknesses (significantly below average)
            weaknesses = {domain: acc for domain, acc in provider_accuracies.items()
                         if acc < avg_accuracy - 0.1 and acc < 0.6}

            domain_analysis['provider_domain_strengths'][provider] = strengths
            domain_analysis['provider_domain_weaknesses'][provider] = weaknesses

    return domain_analysis

def identify_domain_bias(domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Identify systematic domain biases across providers."""

    bias_analysis = {
        'dataset_bias': {},
        'provider_bias': {},
        'fairness_assessment': {},
        'systematic_advantages': {}
    }

    # Dataset bias analysis
    total_questions = sum(domain_analysis['domain_distribution'].values())

    bias_analysis['dataset_bias'] = {
        'domain_distribution': dict(domain_analysis['domain_distribution']),
        'domain_percentages': {
            domain: count / total_questions * 100
            for domain, count in domain_analysis['domain_distribution'].items()
        },
        'most_represented': domain_analysis['domain_distribution'].most_common(3),
        'least_represented': domain_analysis['domain_distribution'].most_common()[-3:],
        'total_questions': total_questions
    }

    # Provider bias analysis
    providers = list(domain_analysis['provider_domain_performance'].keys())

    for provider in providers:
        provider_domains = domain_analysis['provider_domain_performance'][provider]

        # Calculate domain-specific performance relative to other providers
        relative_performance = {}

        for domain in domain_analysis['domain_distribution'].keys():
            if domain in provider_domains and provider_domains[domain]['total'] >= 3:
                provider_acc = provider_domains[domain]['accuracy']

                # Calculate average accuracy of other providers in this domain
                other_providers_acc = []
                for other_provider in providers:
                    if other_provider != provider:
                        other_domains = domain_analysis['provider_domain_performance'][other_provider]
                        if domain in other_domains and other_domains[domain]['total'] >= 3:
                            other_providers_acc.append(other_domains[domain]['accuracy'])

                if other_providers_acc:
                    avg_other_acc = np.mean(other_providers_acc)
                    relative_performance[domain] = provider_acc - avg_other_acc

        bias_analysis['provider_bias'][provider] = {
            'relative_performance': relative_performance,
            'strongest_domains': sorted(relative_performance.items(), key=lambda x: x[1], reverse=True)[:3],
            'weakest_domains': sorted(relative_performance.items(), key=lambda x: x[1])[:3]
        }

    # Fairness assessment
    for domain in domain_analysis['domain_distribution'].keys():
        domain_accuracies = []
        for provider in providers:
            if domain in domain_analysis['provider_domain_performance'][provider]:
                stats = domain_analysis['provider_domain_performance'][provider][domain]
                if stats['total'] >= 3:
                    domain_accuracies.append(stats['accuracy'])

        if len(domain_accuracies) >= 2:
            bias_analysis['fairness_assessment'][domain] = {
                'provider_variance': np.var(domain_accuracies),
                'max_difference': max(domain_accuracies) - min(domain_accuracies),
                'fairness_score': 1 - (np.var(domain_accuracies) / 0.25)  # Normalized variance
            }

    return bias_analysis

def generate_domain_bias_report(domain_analysis: Dict[str, Any], bias_analysis: Dict[str, Any]) -> str:
    """Generate comprehensive domain bias analysis report."""

    report = []
    report.append("# DOMAIN BIAS ANALYSIS")
    report.append("## Provider Performance Across Knowledge Domains")
    report.append("=" * 60)
    report.append("")

    # Dataset composition
    report.append("## DATASET DOMAIN COMPOSITION")
    report.append("")

    total_questions = bias_analysis['dataset_bias']['total_questions']
    report.append(f"**Total Questions Analyzed: {total_questions}**")
    report.append("")

    report.append("**Domain Distribution:**")
    for domain, percentage in sorted(bias_analysis['dataset_bias']['domain_percentages'].items(),
                                   key=lambda x: x[1], reverse=True):
        count = bias_analysis['dataset_bias']['domain_distribution'][domain]
        report.append(f"- {domain}: {count} questions ({percentage:.1f}%)")
    report.append("")

    # Domain difficulty ranking
    report.append("## DOMAIN DIFFICULTY RANKING")
    report.append("")

    difficulty_ranking = sorted(domain_analysis['domain_difficulty'].items(),
                              key=lambda x: x[1]['overall_accuracy'])

    report.append("**Domains Ranked by Difficulty (Hardest to Easiest):**")
    for rank, (domain, stats) in enumerate(difficulty_ranking, 1):
        report.append(f"{rank}. **{domain}**: {stats['overall_accuracy']:.1%} overall accuracy "
                     f"({stats['total_questions']} questions)")
    report.append("")

    # Provider performance by domain
    report.append("## PROVIDER PERFORMANCE BY DOMAIN")
    report.append("")

    providers = list(domain_analysis['provider_domain_performance'].keys())
    for provider in providers:
        report.append(f"### {provider}")

        provider_domains = domain_analysis['provider_domain_performance'][provider]

        # Sort by accuracy
        sorted_domains = sorted(
            [(domain, stats) for domain, stats in provider_domains.items() if stats['total'] >= 3],
            key=lambda x: x[1]['accuracy'], reverse=True
        )

        if sorted_domains:
            report.append("**Domain Performance (≥3 questions):**")
            for domain, stats in sorted_domains:
                report.append(f"- {domain}: {stats['accuracy']:.1%} "
                             f"({stats['correct']}/{stats['total']})")

        # Strengths and weaknesses
        strengths = domain_analysis['provider_domain_strengths'].get(provider, {})
        weaknesses = domain_analysis['provider_domain_weaknesses'].get(provider, {})

        if strengths:
            report.append(f"**Strengths:** {', '.join(strengths.keys())}")
        if weaknesses:
            report.append(f"**Weaknesses:** {', '.join(weaknesses.keys())}")

        report.append("")

    # Provider bias analysis
    report.append("## PROVIDER BIAS ANALYSIS")
    report.append("")

    for provider, bias_data in bias_analysis['provider_bias'].items():
        report.append(f"### {provider} Relative Performance")

        strongest = bias_data['strongest_domains']
        weakest = bias_data['weakest_domains']

        if strongest:
            report.append("**Strongest Relative Domains:**")
            for domain, diff in strongest:
                report.append(f"- {domain}: {diff:+.1%} vs other providers")

        if weakest:
            report.append("**Weakest Relative Domains:**")
            for domain, diff in weakest:
                report.append(f"- {domain}: {diff:+.1%} vs other providers")

        report.append("")

    # Fairness assessment
    report.append("## DOMAIN FAIRNESS ASSESSMENT")
    report.append("")

    fairness_data = bias_analysis['fairness_assessment']
    sorted_fairness = sorted(fairness_data.items(), key=lambda x: x[1]['max_difference'], reverse=True)

    report.append("**Domains with Largest Provider Performance Gaps:**")
    for domain, fairness in sorted_fairness[:5]:
        report.append(f"- {domain}: {fairness['max_difference']:.1%} max difference between providers")
    report.append("")

    # Key insights
    report.append("## KEY INSIGHTS")
    report.append("")

    # Hardest and easiest domains
    hardest_domain = difficulty_ranking[0]
    easiest_domain = difficulty_ranking[-1]

    report.append(f"1. **Hardest domain**: {hardest_domain[0]} ({hardest_domain[1]['overall_accuracy']:.1%} accuracy)")
    report.append(f"2. **Easiest domain**: {easiest_domain[0]} ({easiest_domain[1]['overall_accuracy']:.1%} accuracy)")

    # Most/least represented domains
    most_rep = bias_analysis['dataset_bias']['most_represented'][0]
    least_rep = bias_analysis['dataset_bias']['least_represented'][0]

    report.append(f"3. **Most represented**: {most_rep[0]} ({most_rep[1]} questions)")
    report.append(f"4. **Least represented**: {least_rep[0]} ({least_rep[1]} questions)")

    # Provider with most domain biases
    max_bias_provider = max(bias_analysis['provider_bias'].items(),
                           key=lambda x: len(x[1]['relative_performance']))

    report.append(f"5. **Most domain variation**: {max_bias_provider[0]} with performance gaps across domains")

    # Dataset representativeness
    domain_variance = np.var(list(bias_analysis['dataset_bias']['domain_percentages'].values()))
    report.append(f"6. **Dataset balance**: {'Well-balanced' if domain_variance < 50 else 'Imbalanced'} "
                 f"domain distribution")

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Domain Bias Analysis for RAG Benchmark')
    parser.add_argument('--judge-file', required=True, help='Path to judge evaluations JSONL file')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    args = parser.parse_args()

    # Load evaluations
    print("Loading judge evaluations...")
    evaluations = load_judge_evaluations(args.judge_file)
    print(f"Loaded {len(evaluations)} evaluations")

    # Analyze domain performance
    print("Analyzing domain performance...")
    domain_analysis = analyze_domain_performance(evaluations)

    # Identify domain bias
    print("Identifying domain bias patterns...")
    bias_analysis = identify_domain_bias(domain_analysis)

    # Generate report
    report_content = generate_domain_bias_report(domain_analysis, bias_analysis)

    # Save results
    output_dir = Path(args.output_dir)

    # Save JSON analysis
    analysis_output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'analysis_type': 'domain_bias_analysis',
        'domain_analysis': domain_analysis,
        'bias_analysis': bias_analysis,
        'summary': {
            'total_domains': len(domain_analysis['domain_distribution']),
            'total_questions': sum(domain_analysis['domain_distribution'].values()),
            'hardest_domain': min(domain_analysis['domain_difficulty'].items(),
                                key=lambda x: x[1]['overall_accuracy'])[0],
            'easiest_domain': max(domain_analysis['domain_difficulty'].items(),
                                key=lambda x: x[1]['overall_accuracy'])[0]
        }
    }

    json_file = output_dir / "domain_bias_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)

    # Save report
    report_file = output_dir / "domain_bias_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"\n=== DOMAIN BIAS ANALYSIS COMPLETE ===")
    print(f"JSON results: {json_file}")
    print(f"Report: {report_file}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Domains analyzed: {len(domain_analysis['domain_distribution'])}")
    print(f"Total questions: {sum(domain_analysis['domain_distribution'].values())}")

    # Show top domains
    top_domains = domain_analysis['domain_distribution'].most_common(3)
    print(f"Top domains: {', '.join([f'{d[0]} ({d[1]})' for d in top_domains])}")

if __name__ == "__main__":
    main()