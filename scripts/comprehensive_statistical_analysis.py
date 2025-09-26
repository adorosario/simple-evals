#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for 1000-Example RAG Benchmark Results
Academic-grade statistical validation of provider performance differences.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import math
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse
import sys

def load_benchmark_results(results_file: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def extract_provider_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract key metrics for each provider into a DataFrame."""
    data = []
    for result in results['results']:
        provider = result['sampler_name']
        metrics = result['metrics']
        data.append({
            'provider': provider,
            'n_correct': metrics['n_correct'],
            'n_incorrect': metrics['n_incorrect'],
            'n_not_attempted': metrics['n_not_attempted'],
            'volume_score': metrics['volume_score'],
            'quality_score': metrics['quality_score'],
            'accuracy_given_attempted': metrics['accuracy_given_attempted'],
            'abstention_rate': metrics['abstention_rate'],
            'attempted_rate': metrics['attempted_rate'],
            'overconfidence_penalty': metrics['overconfidence_penalty']
        })
    return pd.DataFrame(data)

def calculate_confidence_intervals(n_correct: int, n_total: int, confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence intervals for accuracy using Wilson score interval."""
    if n_total == 0:
        return 0.0, 0.0

    p = n_correct / n_total
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Wilson score interval (more accurate for proportions)
    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator

    return centre - half_width, centre + half_width

def perform_statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive statistical tests comparing all provider pairs."""
    results = {
        'accuracy_comparisons': {},
        'volume_comparisons': {},
        'quality_score_comparisons': {},
        'abstention_comparisons': {},
        'chi_square_tests': {},
        'statistical_summary': {}
    }

    providers = df['provider'].tolist()

    # Accuracy comparisons (two-proportion z-tests)
    print("=== ACCURACY STATISTICAL TESTS ===")
    for i in range(len(providers)):
        for j in range(i + 1, len(providers)):
            provider1, provider2 = providers[i], providers[j]

            # Get data for both providers
            p1_data = df[df['provider'] == provider1].iloc[0]
            p2_data = df[df['provider'] == provider2].iloc[0]

            # Two-proportion z-test for accuracy
            n1_correct = p1_data['n_correct']
            n1_total = n1_correct + p1_data['n_incorrect']  # Only attempted
            n2_correct = p2_data['n_correct']
            n2_total = n2_correct + p2_data['n_incorrect']  # Only attempted

            if n1_total > 0 and n2_total > 0:
                # Two-proportion z-test
                p1 = n1_correct / n1_total
                p2 = n2_correct / n2_total

                pooled_p = (n1_correct + n2_correct) / (n1_total + n2_total)
                se = math.sqrt(pooled_p * (1 - pooled_p) * (1/n1_total + 1/n2_total))

                if se > 0:
                    z_stat = (p1 - p2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                    results['accuracy_comparisons'][f"{provider1}_vs_{provider2}"] = {
                        'provider1': provider1,
                        'provider2': provider2,
                        'p1_accuracy': p1,
                        'p2_accuracy': p2,
                        'p1_n': n1_total,
                        'p2_n': n2_total,
                        'z_statistic': z_stat,
                        'p_value': p_value,
                        'significant_at_05': p_value < 0.05,
                        'significant_at_01': p_value < 0.01,
                        'effect_size': p1 - p2
                    }

                    print(f"{provider1} vs {provider2}:")
                    print(f"  Accuracy: {p1:.3f} vs {p2:.3f} (diff: {p1-p2:+.3f})")
                    print(f"  Z-statistic: {z_stat:.3f}, p-value: {p_value:.6f}")
                    print(f"  Significant at α=0.05: {p_value < 0.05}")
                    print(f"  Significant at α=0.01: {p_value < 0.01}")
                    print()

    # Quality score comparisons (using quality scores as continuous variables)
    print("=== QUALITY SCORE ANALYSIS ===")
    quality_scores = {}
    for _, row in df.iterrows():
        provider = row['provider']
        quality_scores[provider] = row['quality_score']

        # Calculate confidence interval for quality score
        n_total = 1000  # All providers tested on 1000 examples
        ci_lower, ci_upper = calculate_confidence_intervals(
            max(0, int(row['quality_score'] * n_total + n_total)), n_total * 2
        )

        print(f"{provider}: Quality Score = {row['quality_score']:.3f}")
        print(f"  Volume Score: {row['volume_score']:.3f}")
        print(f"  Accuracy (attempted): {row['accuracy_given_attempted']:.3f}")
        print(f"  Abstention Rate: {row['abstention_rate']:.3f}")
        print()

    # Chi-square test for abstention patterns
    print("=== ABSTENTION PATTERN ANALYSIS ===")
    abstention_data = []
    attempted_data = []
    for _, row in df.iterrows():
        abstention_data.append(row['n_not_attempted'])
        attempted_data.append(1000 - row['n_not_attempted'])  # Total - abstentions

    # Chi-square test for independence of abstention patterns
    contingency_table = [attempted_data, abstention_data]
    chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)

    results['chi_square_tests']['abstention_independence'] = {
        'chi2_statistic': chi2_stat,
        'p_value': chi2_p_value,
        'degrees_of_freedom': dof,
        'significant_at_05': chi2_p_value < 0.05,
        'contingency_table': {
            'attempted': attempted_data,
            'abstained': abstention_data,
            'providers': providers
        }
    }

    print(f"Chi-square test for abstention independence:")
    print(f"  Chi-square statistic: {chi2_stat:.3f}")
    print(f"  p-value: {chi2_p_value:.6f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  Significant at α=0.05: {chi2_p_value < 0.05}")
    print()

    # Multiple comparison correction (Bonferroni)
    n_comparisons = len(results['accuracy_comparisons'])
    bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    print(f"=== MULTIPLE COMPARISON CORRECTION ===")
    print(f"Number of pairwise comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected α: {bonferroni_alpha:.6f}")

    significant_after_correction = 0
    for comparison, data in results['accuracy_comparisons'].items():
        data['significant_bonferroni'] = data['p_value'] < bonferroni_alpha
        if data['significant_bonferroni']:
            significant_after_correction += 1
            print(f"  {comparison}: p = {data['p_value']:.6f} < {bonferroni_alpha:.6f} (SIGNIFICANT)")

    results['statistical_summary'] = {
        'total_comparisons': n_comparisons,
        'bonferroni_alpha': bonferroni_alpha,
        'significant_before_correction': sum(1 for d in results['accuracy_comparisons'].values() if d['significant_at_05']),
        'significant_after_correction': significant_after_correction,
        'sample_size_per_provider': 1000
    }

    return results

def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> Dict[str, float]:
    """Calculate required sample size and achieved power for given effect size."""
    from scipy.stats import norm

    # For two-proportion test
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    # Estimate required sample size for detecting effect_size difference
    # Using conservative estimate with p1 ≈ p2 ≈ 0.7 (approximate average accuracy)
    p = 0.7
    required_n = 2 * (z_alpha + z_beta)**2 * p * (1 - p) / effect_size**2

    # Calculate achieved power with n=1000 per group
    current_n = 1000
    z_stat_threshold = effect_size * math.sqrt(current_n / (2 * p * (1 - p)))
    achieved_power = 1 - norm.cdf(z_alpha - z_stat_threshold) + norm.cdf(-z_alpha - z_stat_threshold)

    return {
        'effect_size': effect_size,
        'alpha': alpha,
        'target_power': power,
        'required_sample_size_per_group': math.ceil(required_n),
        'current_sample_size_per_group': current_n,
        'achieved_power': achieved_power,
        'adequately_powered': achieved_power >= power
    }

def generate_comprehensive_report(df: pd.DataFrame, stats_results: Dict[str, Any],
                                power_results: Dict[str, Dict[str, float]]) -> str:
    """Generate a comprehensive statistical report."""
    report = []
    report.append("# COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    report.append("## 1000-Example RAG Provider Benchmark Results")
    report.append("=" * 60)
    report.append("")

    # Provider summary
    report.append("## PROVIDER PERFORMANCE SUMMARY")
    report.append("")
    df_sorted = df.sort_values('quality_score', ascending=False)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report.append(f"{i}. **{row['provider']}**")
        report.append(f"   - Quality Score: {row['quality_score']:.3f}")
        report.append(f"   - Volume Score: {row['volume_score']:.3f}")
        report.append(f"   - Accuracy (attempted): {row['accuracy_given_attempted']:.3f}")
        report.append(f"   - Correct/Incorrect/Abstained: {row['n_correct']}/{row['n_incorrect']}/{row['n_not_attempted']}")
        report.append(f"   - Abstention Rate: {row['abstention_rate']:.3f}")
        report.append("")

    # Statistical significance results
    report.append("## STATISTICAL SIGNIFICANCE ANALYSIS")
    report.append("")

    for comparison, data in stats_results['accuracy_comparisons'].items():
        report.append(f"### {data['provider1']} vs {data['provider2']}")
        report.append(f"- Accuracy: {data['p1_accuracy']:.3f} vs {data['p2_accuracy']:.3f}")
        report.append(f"- Difference: {data['effect_size']:+.3f}")
        report.append(f"- Z-statistic: {data['z_statistic']:.3f}")
        report.append(f"- p-value: {data['p_value']:.6f}")
        report.append(f"- Significant (α=0.05): {'YES' if data['significant_at_05'] else 'NO'}")
        report.append(f"- Significant (α=0.01): {'YES' if data['significant_at_01'] else 'NO'}")
        report.append(f"- Significant (Bonferroni): {'YES' if data['significant_bonferroni'] else 'NO'}")
        report.append("")

    # Multiple comparisons summary
    summary = stats_results['statistical_summary']
    report.append("## MULTIPLE COMPARISONS CORRECTION")
    report.append(f"- Total pairwise comparisons: {summary['total_comparisons']}")
    report.append(f"- Bonferroni-corrected α: {summary['bonferroni_alpha']:.6f}")
    report.append(f"- Significant before correction: {summary['significant_before_correction']}")
    report.append(f"- Significant after correction: {summary['significant_after_correction']}")
    report.append("")

    # Power analysis
    report.append("## POWER ANALYSIS")
    report.append("")
    for effect_size, power_data in power_results.items():
        report.append(f"### Effect Size: {effect_size}")
        report.append(f"- Required sample size (power=0.8): {power_data['required_sample_size_per_group']:,}")
        report.append(f"- Current sample size: {power_data['current_sample_size_per_group']:,}")
        report.append(f"- Achieved power: {power_data['achieved_power']:.3f}")
        report.append(f"- Adequately powered: {'YES' if power_data['adequately_powered'] else 'NO'}")
        report.append("")

    # Abstention analysis
    chi2_data = stats_results['chi_square_tests']['abstention_independence']
    report.append("## ABSTENTION PATTERN ANALYSIS")
    report.append(f"- Chi-square statistic: {chi2_data['chi2_statistic']:.3f}")
    report.append(f"- p-value: {chi2_data['p_value']:.6f}")
    report.append(f"- Significant difference in abstention patterns: {'YES' if chi2_data['significant_at_05'] else 'NO'}")
    report.append("")

    # Key findings
    report.append("## KEY STATISTICAL FINDINGS")
    report.append("")

    # Find best performer
    best_performer = df_sorted.iloc[0]['provider']
    report.append(f"1. **{best_performer}** achieves highest quality score ({df_sorted.iloc[0]['quality_score']:.3f})")

    # Check for significant differences
    significant_pairs = [k for k, v in stats_results['accuracy_comparisons'].items()
                        if v['significant_bonferroni']]
    if significant_pairs:
        report.append(f"2. **Statistically significant differences found** (Bonferroni-corrected):")
        for pair in significant_pairs:
            data = stats_results['accuracy_comparisons'][pair]
            report.append(f"   - {data['provider1']} vs {data['provider2']}: p = {data['p_value']:.6f}")
    else:
        report.append("2. **No statistically significant differences** after multiple comparison correction")

    report.append("")

    # Sample size adequacy
    adequately_powered = sum(1 for p in power_results.values() if p['adequately_powered'])
    total_effect_sizes = len(power_results)
    report.append(f"3. **Sample size adequacy**: {adequately_powered}/{total_effect_sizes} effect sizes adequately powered")

    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Statistical Analysis of RAG Benchmark')
    parser.add_argument('--results-file', required=True, help='Path to benchmark results JSON file')
    parser.add_argument('--output-dir', default='results', help='Output directory for analysis')

    args = parser.parse_args()

    # Load results
    print("Loading benchmark results...")
    results = load_benchmark_results(args.results_file)
    df = extract_provider_metrics(results)

    print(f"Analyzing {len(df)} providers with 1000 samples each...")
    print()

    # Perform statistical tests
    stats_results = perform_statistical_tests(df)

    # Power analysis for different effect sizes
    print("=== POWER ANALYSIS ===")
    effect_sizes = [0.05, 0.1, 0.15, 0.2]  # Different effect sizes to test
    power_results = {}

    for effect_size in effect_sizes:
        power_data = power_analysis(effect_size)
        power_results[effect_size] = power_data
        print(f"Effect size {effect_size}:")
        print(f"  Required n (power=0.8): {power_data['required_sample_size_per_group']:,}")
        print(f"  Achieved power (n=1000): {power_data['achieved_power']:.3f}")
        print(f"  Adequately powered: {power_data['adequately_powered']}")
        print()

    # Generate comprehensive report
    report_content = generate_comprehensive_report(df, stats_results, power_results)

    # Save results
    run_id = results.get('run_id', 'unknown')
    output_dir = Path(args.output_dir)

    # Save JSON results (convert numpy types to Python native types)
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_output = {
        'run_id': run_id,
        'analysis_type': 'comprehensive_statistical_analysis',
        'timestamp': pd.Timestamp.now().isoformat(),
        'provider_metrics': convert_numpy_types(df.to_dict('records')),
        'statistical_tests': convert_numpy_types(stats_results),
        'power_analysis': convert_numpy_types(power_results),
        'summary': {
            'best_performer': str(df.sort_values('quality_score', ascending=False).iloc[0]['provider']),
            'statistically_significant_differences': sum(1 for d in stats_results['accuracy_comparisons'].values()
                                                       if d['significant_bonferroni']),
            'adequately_powered_effect_sizes': sum(1 for p in power_results.values() if p['adequately_powered']),
            'sample_size_adequate': all(p['adequately_powered'] for p in power_results.values())
        }
    }

    json_file = output_dir / f"comprehensive_statistical_analysis_{run_id}.json"
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)

    # Save report
    report_file = output_dir / f"comprehensive_statistical_report_{run_id}.md"
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"=== ANALYSIS COMPLETE ===")
    print(f"JSON results saved to: {json_file}")
    print(f"Report saved to: {report_file}")
    print()
    print("=== EXECUTIVE SUMMARY ===")
    print(report_content.split("## KEY STATISTICAL FINDINGS")[1].split("## ")[0])

if __name__ == "__main__":
    main()