#!/usr/bin/env python3
"""
Academic Statistical Analysis Module
Provides rigorous statistical analysis with confidence intervals for benchmark results.

Implements:
- Wilson score confidence intervals for accuracy metrics
- Statistical significance testing between providers
- Publication-ready tables and visualizations
- Academic-grade reporting suitable for peer review

Reference: Wilson, E. B. (1927). "Probable inference, the law of succession,
and statistical inference". Journal of the American Statistical Association.

Uses unified brand kit for consistent, Apple-inspired HTML reports.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import math

# Add parent directory to path for brand kit import
sys.path.insert(0, str(Path(__file__).parent.parent))
from brand_kit import (
    get_html_head,
    get_navigation_bar,
    get_page_header,
    format_timestamp
)


def _scan_available_reports(results_dir: str) -> dict:
    """
    Scan results directory for available reports.

    Returns:
        Dictionary mapping report types to file paths (relative to results_dir)
    """
    reports = {
        'quality_benchmark': None,
        'statistical_analysis': None,
        'forensics': {}
    }

    results_path = Path(results_dir)

    # Find quality benchmark report
    for file in results_path.glob("quality_benchmark_report_*.html"):
        reports['quality_benchmark'] = file.name
        break

    # Find statistical analysis report
    for file in results_path.glob("statistical_analysis_run_*.html"):
        reports['statistical_analysis'] = file.name
        break

    # Find forensic dashboards
    for forensic_dir in results_path.glob("*_forensics"):
        provider = forensic_dir.name.replace("_forensics", "")
        dashboard = forensic_dir / "forensic_dashboard.html"
        if dashboard.exists():
            reports['forensics'][provider] = f"{forensic_dir.name}/forensic_dashboard.html"

    return reports


@dataclass
class ProviderStatistics:
    """Statistical metrics for a provider"""
    provider_name: str
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    accuracy: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    penalty_points: float
    avg_confidence: float

def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a binomial proportion.

    This is more accurate than the normal approximation, especially for small sample sizes
    or proportions near 0 or 1.

    Args:
        successes: Number of successes
        n: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple

    Reference: Wilson, E. B. (1927). JASA 22(158): 209-212
    """
    if n == 0:
        return (0.0, 0.0)

    p = successes / n
    z = 1.96  # For 95% confidence (could use scipy.stats.norm.ppf((1 + confidence) / 2))

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def load_quality_benchmark_results(run_dir: Path) -> Dict[str, Any]:
    """Load quality benchmark results from the run directory"""
    results_file = run_dir / "quality_benchmark_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Quality benchmark results not found: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def calculate_provider_statistics(
    provider_results: Dict[str, Any],
    provider_name: str
) -> ProviderStatistics:
    """
    Calculate comprehensive statistics for a provider including confidence intervals

    Args:
        provider_results: Results dict for the provider (with accuracy, volume, etc.)
        provider_name: Display name of the provider

    Returns:
        ProviderStatistics object with all metrics
    """
    total_questions = provider_results.get('questions_answered', 0)
    accuracy = provider_results.get('accuracy', 0.0)

    # Calculate correct/incorrect from accuracy
    correct_answers = int(round(accuracy * total_questions))
    incorrect_answers = total_questions - correct_answers

    # Wilson score confidence interval for accuracy
    ci_lower, ci_upper = wilson_score_interval(correct_answers, total_questions, confidence=0.95)

    penalty_points = provider_results.get('penalty_points', 0.0)
    avg_confidence = provider_results.get('average_confidence', 0.0)

    return ProviderStatistics(
        provider_name=provider_name,
        total_questions=total_questions,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers,
        accuracy=accuracy,
        accuracy_ci_lower=ci_lower,
        accuracy_ci_upper=ci_upper,
        penalty_points=penalty_points,
        avg_confidence=avg_confidence
    )


def generate_publication_table_markdown(
    stats_by_provider: Dict[str, ProviderStatistics],
    confidence_threshold: float
) -> str:
    """
    Generate publication-ready markdown table

    Args:
        stats_by_provider: Dict mapping provider names to their statistics
        confidence_threshold: The confidence threshold for this table

    Returns:
        Markdown formatted table string
    """
    table = []
    table.append(f"## Results at Confidence Threshold {confidence_threshold}")
    table.append("")
    table.append("| Provider | Questions Answered | Accuracy | 95% CI | Penalty Points |")
    table.append("|----------|-------------------|----------|---------|----------------|")

    for provider_name, stats in sorted(stats_by_provider.items()):
        ci_str = f"[{stats.accuracy_ci_lower:.3f}, {stats.accuracy_ci_upper:.3f}]"
        table.append(
            f"| {stats.provider_name} | "
            f"{stats.total_questions} | "
            f"{stats.accuracy:.3f} | "
            f"{ci_str} | "
            f"{stats.penalty_points:.1f} |"
        )

    table.append("")
    return "\n".join(table)


def generate_publication_table_latex(
    stats_by_provider: Dict[str, ProviderStatistics],
    confidence_threshold: float
) -> str:
    """
    Generate publication-ready LaTeX table

    Args:
        stats_by_provider: Dict mapping provider names to their statistics
        confidence_threshold: The confidence threshold for this table

    Returns:
        LaTeX formatted table string
    """
    table = []
    table.append("\\begin{table}[h]")
    table.append("\\centering")
    table.append(f"\\caption{{Benchmark Results at Confidence Threshold {confidence_threshold}}}")
    table.append("\\begin{tabular}{lcccc}")
    table.append("\\hline")
    table.append("Provider & Questions & Accuracy & 95\\% CI & Penalty Points \\\\")
    table.append("\\hline")

    for provider_name, stats in sorted(stats_by_provider.items()):
        ci_str = f"[{stats.accuracy_ci_lower:.3f}, {stats.accuracy_ci_upper:.3f}]"
        # Escape underscores in provider names for LaTeX
        provider_latex = stats.provider_name.replace('_', '\\_')
        table.append(
            f"{provider_latex} & "
            f"{stats.total_questions} & "
            f"{stats.accuracy:.3f} & "
            f"{ci_str} & "
            f"{stats.penalty_points:.1f} \\\\"
        )

    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    table.append("")
    return "\n".join(table)


def check_statistical_significance(
    stats1: ProviderStatistics,
    stats2: ProviderStatistics
) -> Dict[str, Any]:
    """
    Check if difference between two providers is statistically significant

    Uses confidence interval overlap method:
    - If CIs don't overlap: significant difference
    - If CIs overlap: not definitively significant (conservative approach)

    Args:
        stats1: Statistics for first provider
        stats2: Statistics for second provider

    Returns:
        Dict with significance analysis
    """
    # Check CI overlap
    ci1_lower, ci1_upper = stats1.accuracy_ci_lower, stats1.accuracy_ci_upper
    ci2_lower, ci2_upper = stats2.accuracy_ci_lower, stats2.accuracy_ci_upper

    # No overlap means significant difference
    no_overlap = (ci1_upper < ci2_lower) or (ci2_upper < ci1_lower)

    # Calculate point estimate difference
    diff = stats1.accuracy - stats2.accuracy

    # Who is higher?
    if stats1.accuracy > stats2.accuracy:
        higher_provider = stats1.provider_name
        lower_provider = stats2.provider_name
    else:
        higher_provider = stats2.provider_name
        lower_provider = stats1.provider_name

    return {
        'provider1': stats1.provider_name,
        'provider2': stats2.provider_name,
        'accuracy_diff': abs(diff),
        'higher_provider': higher_provider,
        'lower_provider': lower_provider,
        'ci_overlap': not no_overlap,
        'statistically_significant': no_overlap,
        'interpretation': (
            f"{higher_provider} significantly outperforms {lower_provider} (95% CI)"
            if no_overlap
            else f"No significant difference between {stats1.provider_name} and {stats2.provider_name} (CIs overlap)"
        )
    }


def generate_statistical_analysis_report(
    run_dir: Path,
    output_dir: Optional[Path] = None
) -> str:
    """
    Generate comprehensive statistical analysis report

    Args:
        run_dir: Path to evaluation run directory
        output_dir: Optional custom output directory

    Returns:
        Path to generated report
    """
    if output_dir is None:
        output_dir = run_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    run_id = run_dir.name

    print(f"\n{'='*60}")
    print(f"Academic Statistical Analysis")
    print(f"Run: {run_id}")
    print(f"{'='*60}\n")

    # Load quality benchmark results
    print("Loading benchmark results...")
    results = load_quality_benchmark_results(run_dir)

    # Extract provider results organized by confidence threshold
    provider_name_mapping = {
        'CustomGPT_RAG': 'customgpt',
        'OpenAI_RAG': 'openai_rag',
        'OpenAI_Vanilla': 'openai_vanilla'
    }

    display_names = {
        'customgpt': 'CustomGPT',
        'openai_rag': 'OpenAI RAG',
        'openai_vanilla': 'OpenAI Vanilla'
    }

    # Container for all statistics
    all_stats = {}  # {confidence_threshold: {provider: ProviderStatistics}}

    # Get confidence threshold from configuration
    conf_threshold = results.get('configuration', {}).get('confidence_threshold', {}).get('threshold', 0.8)
    all_stats[conf_threshold] = {}

    # Process each provider from the results array
    for provider_result in results.get('results', []):
        sampler_name = provider_result.get('sampler_name')
        provider_key = provider_name_mapping.get(sampler_name)

        if provider_key:
            # Map metrics to expected format
            metrics = provider_result.get('metrics', {})
            provider_data = {
                'questions_answered': metrics.get('conversations', 0),
                'accuracy': metrics.get('accuracy_given_attempted', 0.0),
                'n_correct': metrics.get('n_correct', 0),
                'n_incorrect': metrics.get('n_incorrect', 0),
                'penalty_points': metrics.get('overconfidence_penalty', 0) * metrics.get('penalty_ratio', 4.0),
                'avg_confidence': 0.0  # Not available in this data format
            }

            stats = calculate_provider_statistics(provider_data, display_names[provider_key])
            all_stats[conf_threshold][provider_key] = stats

    # Generate markdown report
    print("Generating statistical report...")
    report = []
    report.append("# Academic Statistical Analysis Report")
    report.append(f"**Run ID:** `{run_id}`")
    report.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary with actual results
    report.append("## Executive Summary")
    report.append("")

    # Get the single threshold results for summary
    if all_stats:
        threshold = list(all_stats.keys())[0]
        stats = all_stats[threshold]

        # Sort by accuracy descending
        sorted_providers = sorted(stats.items(), key=lambda x: x[1].accuracy, reverse=True)

        report.append(f"**Benchmark Configuration:** {len(sorted_providers)} providers evaluated on {sorted_providers[0][1].total_questions if sorted_providers else 0} questions at {threshold*100:.0f}% confidence threshold")
        report.append("")
        report.append("**Key Findings:**")
        report.append("")

        for i, (provider_key, provider_stats) in enumerate(sorted_providers, 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            report.append(f"{i}. {rank_emoji} **{provider_stats.provider_name}**: {provider_stats.accuracy:.1%} accuracy [{provider_stats.accuracy_ci_lower:.3f}-{provider_stats.accuracy_ci_upper:.3f}] ({provider_stats.penalty_points:.0f} penalty points)")

        report.append("")
        report.append("**Statistical Methodology:** Wilson Score Confidence Intervals (95% CI) for binomial proportions, conservative significance testing via CI overlap")

    report.append("")

    # Results - remove redundant header when only one threshold
    if len(all_stats) == 1:
        # Single threshold - skip the parent header
        pass
    else:
        report.append("## Results by Confidence Threshold")
        report.append("")

    for conf_threshold in sorted(all_stats.keys()):
        stats_by_provider = all_stats[conf_threshold]

        # Markdown table
        report.append(generate_publication_table_markdown(stats_by_provider, conf_threshold))

        # Statistical significance analysis
        report.append(f"### Statistical Significance Analysis (Threshold {conf_threshold})")
        report.append("")

        # Pairwise comparisons
        provider_pairs = [
            ('customgpt', 'openai_rag'),
            ('customgpt', 'openai_vanilla'),
            ('customgpt', 'google_gemini_rag'),
            ('openai_rag', 'openai_vanilla'),
            ('openai_rag', 'google_gemini_rag'),
            ('openai_vanilla', 'google_gemini_rag')
        ]

        for p1, p2 in provider_pairs:
            if p1 in stats_by_provider and p2 in stats_by_provider:
                sig_result = check_statistical_significance(
                    stats_by_provider[p1],
                    stats_by_provider[p2]
                )

                icon = "✅" if sig_result['statistically_significant'] else "⚠️"
                report.append(f"{icon} **{sig_result['interpretation']}**")
                report.append(f"   - Accuracy difference: {sig_result['accuracy_diff']:.3f}")
                report.append("")

        report.append("---")
        report.append("")

    # LaTeX tables appendix
    report.append("## Appendix: LaTeX Tables")
    report.append("")
    report.append("Copy-paste ready LaTeX tables for publication:")
    report.append("")
    report.append("```latex")

    for conf_threshold in sorted(all_stats.keys()):
        stats_by_provider = all_stats[conf_threshold]
        latex_table = generate_publication_table_latex(stats_by_provider, conf_threshold)
        report.append(latex_table)

    report.append("```")
    report.append("")

    # Methodology
    report.append("## Statistical Methodology")
    report.append("")
    report.append("### Confidence Intervals")
    report.append("")
    report.append("We use **Wilson score confidence intervals** for binomial proportions (accuracy).")
    report.append("This method is superior to the normal approximation for small to medium sample sizes ")
    report.append("and for proportions near 0 or 1.")
    report.append("")
    report.append("**Formula:**")
    report.append("```")
    report.append("CI = (p + z²/2n ± z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)")
    report.append("```")
    report.append("Where:")
    report.append("- p = sample proportion (accuracy)")
    report.append("- n = sample size (number of questions)")
    report.append("- z = 1.96 (for 95% confidence)")
    report.append("")
    report.append("**Reference:** Wilson, E. B. (1927). \"Probable inference, the law of succession, and statistical inference\". ")
    report.append("*Journal of the American Statistical Association*, 22(158), 209-212.")
    report.append("")

    report.append("### Significance Testing")
    report.append("")
    report.append("We use a **conservative approach** based on confidence interval overlap:")
    report.append("- If 95% CIs do not overlap → statistically significant difference (p < 0.05)")
    report.append("- If 95% CIs overlap → no definitive conclusion (requires further testing)")
    report.append("")
    report.append("This is more conservative than a formal hypothesis test, reducing false positives.")
    report.append("")

    # Save report
    md_file = output_dir / f"statistical_analysis_{run_id}.md"
    with open(md_file, 'w') as f:
        f.write("\n".join(report))
    print(f"✓ Statistical analysis report saved: {md_file}")

    # Save structured data as JSON
    json_data = {
        "run_id": run_id,
        "analysis_timestamp": datetime.now().isoformat(),
        "methodology": {
            "confidence_interval_method": "Wilson Score",
            "confidence_level": 0.95,
            "significance_testing": "CI Overlap (Conservative)"
        },
        "results_by_threshold": {}
    }

    for conf_threshold in sorted(all_stats.keys()):
        threshold_key = f"confidence_{conf_threshold}"
        json_data["results_by_threshold"][threshold_key] = {}

        for provider, stats in all_stats[conf_threshold].items():
            json_data["results_by_threshold"][threshold_key][provider] = {
                "provider_name": stats.provider_name,
                "total_questions": stats.total_questions,
                "correct_answers": stats.correct_answers,
                "incorrect_answers": stats.incorrect_answers,
                "accuracy": stats.accuracy,
                "accuracy_ci_95": {
                    "lower": stats.accuracy_ci_lower,
                    "upper": stats.accuracy_ci_upper
                },
                "penalty_points": stats.penalty_points,
                "average_confidence": stats.avg_confidence
            }

    json_file = output_dir / f"statistical_analysis_{run_id}.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Statistical analysis data saved: {json_file}")

    # Generate HTML version for easy viewing
    html_content = markdown_to_html("\n".join(report), run_dir=str(run_dir), run_id=run_id)
    html_file = output_dir / f"statistical_analysis_{run_id}.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    print(f"✓ HTML report saved: {html_file}")

    print(f"\n{'='*60}")
    print(f"Statistical Analysis Complete")
    print(f"{'='*60}\n")

    return str(md_file)


def markdown_to_html(markdown_text: str, run_dir: str = "", run_id: str = "unknown") -> str:
    """Convert markdown to HTML with Bootstrap styling"""
    # Simple markdown to HTML conversion
    import re

    # Scan for available reports for navigation
    available_reports = _scan_available_reports(run_dir) if run_dir else {}

    html_lines = []
    in_table = False
    in_code_block = False
    code_block_lang = ""

    lines = markdown_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_block_lang = line[3:].strip()
                html_lines.append('<pre><code>')
            else:
                in_code_block = False
                html_lines.append('</code></pre>')
            i += 1
            continue

        if in_code_block:
            html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue

        # Headers
        if line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        # Tables
        elif '|' in line and not in_table:
            # Start of table
            in_table = True
            headers = [h.strip() for h in line.split('|')[1:-1]]
            html_lines.append('<table class="table table-striped table-bordered">')
            html_lines.append('<thead><tr>')
            for h in headers:
                html_lines.append(f'<th>{h}</th>')
            html_lines.append('</tr></thead>')
            # Skip separator line
            i += 1
            html_lines.append('<tbody>')
        elif '|' in line and in_table:
            if line.strip().startswith('|---'):
                # Skip separator
                pass
            else:
                # Table row
                cells = [c.strip() for c in line.split('|')[1:-1]]
                html_lines.append('<tr>')
                for c in cells:
                    # Process inline formatting
                    c = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', c)
                    c = re.sub(r'`(.*?)`', r'<code>\1</code>', c)
                    html_lines.append(f'<td>{c}</td>')
                html_lines.append('</tr>')
        elif in_table and not ('|' in line):
            # End of table
            in_table = False
            html_lines.append('</tbody></table>')
            continue
        # Horizontal rule
        elif line.strip() == '---':
            html_lines.append('<hr>')
        # Lists
        elif line.startswith('- '):
            # Check if we're starting a list
            if i == 0 or not lines[i-1].startswith('- '):
                html_lines.append('<ul>')
            item = line[2:]
            # Process inline formatting
            item = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'`(.*?)`', r'<code>\1</code>', item)
            html_lines.append(f'<li>{item}</li>')
            # Check if we're ending a list
            if i == len(lines) - 1 or (i + 1 < len(lines) and not lines[i+1].startswith('- ')):
                html_lines.append('</ul>')
        # Paragraphs
        elif line.strip():
            # Process inline formatting
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'`(.*?)`', r'<code>\1</code>', line)
            line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line)
            # Check for emoji bullets
            if line.startswith('✅') or line.startswith('⚠️'):
                html_lines.append(f'<p class="alert {"alert-success" if line.startswith("✅") else "alert-warning"}">{line}</p>')
            else:
                html_lines.append(f'<p>{line}</p>')
        # Skip blank lines - CSS margins will handle spacing

        i += 1

    # Close any open table
    if in_table:
        html_lines.append('</tbody></table>')

    html_content = '\n'.join(html_lines)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 40px; background-color: #f5f7fa; }}
        .container {{ max-width: 1200px; background-color: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; margin-bottom: 20px; margin-top: 0; }}
        h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h3 {{ color: #555; margin-top: 30px; margin-bottom: 15px; }}
        p {{ margin: 10px 0; }}
        p:first-of-type {{ margin-top: 0; }}
        table {{
            margin: 20px 0;
            border-collapse: collapse;
            width: 100%;
        }}
        table thead {{
            background-color: #34495e;
            color: white;
        }}
        table th, table td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        table tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        table tbody tr:hover {{
            background-color: #e9ecef;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            overflow-x: auto;
        }}
        code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        hr {{
            margin: 40px 0;
            border: 0;
            border-top: 2px solid #ecf0f1;
        }}
        strong {{
            color: #2c3e50;
        }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .alert-success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        ul {{
            margin: 15px 0;
            padding-left: 30px;
        }}
        li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
    """

    # ACTUALLY: Use brand kit instead of inline styles (LEGACY CODE ABOVE)
    # Wrap content in brand kit template for consistency
    html_with_brand_kit = get_html_head(
        title="Statistical Analysis Report",
        description="Academic-grade statistical analysis with Wilson score confidence intervals"
    )

    html_with_brand_kit += f"""
<body>
    {get_navigation_bar(
        active_page='statistical',
        run_id=run_id,
        base_path="",
        quality_report=available_reports.get('quality_benchmark'),
        statistical_report=available_reports.get('statistical_analysis'),
        forensic_reports=available_reports.get('forensics', {})
    )}

    <div class="main-container">
        {get_page_header(
            title="Statistical Analysis Report",
            subtitle="Wilson score confidence intervals and statistical significance testing",
            meta_info=f"Generated: {format_timestamp()}"
        )}

        <div class="content-section">
            <div class="info-box">
                {html_content}
            </div>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center text-muted mb-4">
                <p>
                    <strong>Statistical Analysis</strong> |
                    Generated: {format_timestamp()}
                </p>
            </div>
        </div>
    </div>
</body>
</html>
"""

    return html_with_brand_kit


def main():
    parser = argparse.ArgumentParser(
        description='Generate academic statistical analysis with confidence intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python academic_statistical_analysis.py --run-dir results/run_XXX
  python academic_statistical_analysis.py --run-dir results/run_XXX --output-dir custom_output/
        """
    )
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--output-dir', help='Output directory (default: run directory)')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    generate_statistical_analysis_report(run_dir, output_dir)


if __name__ == "__main__":
    main()
