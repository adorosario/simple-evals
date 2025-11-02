"""
Main Dashboard Hub Generator
============================

Generates the central dashboard hub (index.html) that links to all sub-dashboards:
- Quality Benchmark Report
- Statistical Analysis
- Forensic Dashboards (per provider)
- Individual Question Reports

This is the entry point for the entire RAG benchmarking system.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brand_kit import (
    get_html_head,
    get_navigation_bar,
    get_page_header,
    wrap_html_document,
    format_timestamp,
    BRAND_COLORS
)


def generate_main_dashboard(results_dir: str, run_metadata: dict):
    """
    Generate main dashboard hub for a benchmark run.

    Args:
        results_dir: Directory containing all benchmark results
        run_metadata: Dictionary with run configuration and metadata
    """
    run_id = run_metadata.get("run_id", "")
    timestamp = run_metadata.get("timestamp", datetime.now())
    providers = run_metadata.get("providers", [])
    total_questions = run_metadata.get("total_questions", 0)
    confidence_threshold = run_metadata.get("confidence_threshold", 0.8)

    # Collect available reports
    available_reports = scan_available_reports(results_dir)

    html = get_html_head(
        title="Why RAGs Hallucinate - Main Dashboard",
        description="Academic-grade RAG benchmark analysis using OpenAI's penalty-aware scoring methodology"
    )

    html += f"""
<body>
    {get_navigation_bar(
        active_page='home',
        run_id=run_id,
        base_path="",
        quality_report=available_reports.get('quality_benchmark'),
        statistical_report=available_reports.get('statistical_analysis'),
        forensic_reports=available_reports.get('forensics', {})
    )}

    <div class="main-container">
        {get_page_header(
            title="Why RAGs Hallucinate",
            subtitle="Academic RAG Benchmark Analysis",
            meta_info=f"Run ID: <code>{run_id}</code> | Generated: {format_timestamp(timestamp)}"
        )}

        <div class="content-section">
            <!-- Executive Summary -->
            <h2 class="section-header">
                <i class="fas fa-chart-line me-2"></i>Executive Summary
            </h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <h3><i class="fas fa-users me-2"></i>Providers Tested</h3>
                    <div class="value">{len(providers)}</div>
                    <div class="description">{', '.join(providers)}</div>
                </div>

                <div class="metric-card">
                    <h3><i class="fas fa-question-circle me-2"></i>Questions Per Provider</h3>
                    <div class="value">{total_questions}</div>
                    <div class="description">SimpleQA benchmark questions</div>
                </div>

                <div class="metric-card">
                    <h3><i class="fas fa-calculator me-2"></i>Total Evaluations</h3>
                    <div class="value">{len(providers) * total_questions}</div>
                    <div class="description">Questions × Providers</div>
                </div>

                <div class="metric-card">
                    <h3><i class="fas fa-shield-alt me-2"></i>Confidence Threshold</h3>
                    <div class="value">{int(confidence_threshold * 100)}%</div>
                    <div class="description">Judge confidence threshold (OpenAI recommended)</div>
                </div>
            </div>

            <!-- Research Methodology -->
            <h2 class="section-header">
                <i class="fas fa-flask me-2"></i>Methodology
            </h2>

            <div class="info-box">
                <h4><strong>Research Foundation</strong></h4>
                <p>
                    This benchmark implements OpenAI's <strong>penalty-aware scoring methodology</strong> from
                    <a href="https://openai.com/index/why-language-models-hallucinate/" target="_blank" rel="noopener">
                        "Why Language Models Hallucinate" (arXiv:2509.04664v1)
                    </a>.
                    The research demonstrates that traditional accuracy metrics create an "I Don't Know" tax that
                    penalizes appropriately calibrated systems.
                </p>

                <div class="row mt-4">
                    <div class="col-md-6">
                        <h5><i class="fas fa-trophy me-2" style="color: {BRAND_COLORS['quality']};"></i>Quality Strategy (Penalty-Aware)</h5>
                        <ul>
                            <li><strong>Scoring:</strong> Correct = +1, Wrong = -4, Abstain = 0</li>
                            <li><strong>Philosophy:</strong> Rewards appropriate uncertainty and calibration</li>
                            <li><strong>Best for:</strong> High-stakes applications where accuracy matters</li>
                            <li><strong>Key insight:</strong> Penalizes overconfident incorrect responses</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h5><i class="fas fa-chart-line me-2" style="color: {BRAND_COLORS['volume']};"></i>Volume Strategy (Traditional)</h5>
                        <ul>
                            <li><strong>Scoring:</strong> Correct = +1, Wrong = 0, Abstain = 0</li>
                            <li><strong>Philosophy:</strong> Rewards guessing and penalizes uncertainty</li>
                            <li><strong>Best for:</strong> High-volume, low-stakes applications</li>
                            <li><strong>Problem:</strong> Creates "I Don't Know" tax for conservative systems</li>
                        </ul>
                    </div>
                </div>

                <div class="alert alert-info mt-3">
                    <strong><i class="fas fa-info-circle me-2"></i>Why 80% Confidence?</strong>
                    OpenAI's research recommends an 80% confidence threshold for LLM-as-a-Judge evaluations.
                    This conservative threshold filters out borderline decisions that could distort results,
                    ensuring only high-confidence judgments are included in the final scores.
                </div>
            </div>

            <div class="info-box">
                <h4><strong>Technical Implementation</strong></h4>
                <div class="row">
                    <div class="col-md-3">
                        <strong><i class="fas fa-robot me-1"></i>LLM Provider:</strong><br>
                        GPT-4.1 (standardized across all RAG systems)
                    </div>
                    <div class="col-md-3">
                        <strong><i class="fas fa-gavel me-1"></i>Judge Model:</strong><br>
                        GPT-5 Standard Tier
                    </div>
                    <div class="col-md-3">
                        <strong><i class="fas fa-database me-1"></i>Dataset:</strong><br>
                        <a href="https://github.com/openai/simple-evals" target="_blank">SimpleQA</a>
                    </div>
                    <div class="col-md-3">
                        <strong><i class="fas fa-calculator me-1"></i>Penalty Ratio:</strong><br>
                        4.0 (wrong = -4 points)
                    </div>
                </div>
            </div>

            <!-- Available Reports -->
            <h2 class="section-header">
                <i class="fas fa-file-alt me-2"></i>Available Reports
            </h2>

            <div class="row">
                <!-- Quality Benchmark Report -->
                <div class="col-md-4 mb-4">
                    <div class="card h-100 metric-card">
                        <div class="card-body">
                            <h3 class="card-title">
                                <i class="fas fa-trophy me-2" style="color: {BRAND_COLORS['primary']};"></i>
                                Quality Benchmark
                            </h3>
                            <p class="card-text">
                                Provider leaderboard with quality vs volume scoring comparison.
                                Shows abstention rates, penalty points, and strategy assessment.
                            </p>
                            {_get_report_link(available_reports, 'quality_benchmark')}
                        </div>
                    </div>
                </div>

                <!-- Statistical Analysis Report -->
                <div class="col-md-4 mb-4">
                    <div class="card h-100 metric-card">
                        <div class="card-body">
                            <h3 class="card-title">
                                <i class="fas fa-chart-bar me-2" style="color: {BRAND_COLORS['success']};"></i>
                                Statistical Analysis
                            </h3>
                            <p class="card-text">
                                Academic-grade statistical analysis with Wilson score confidence intervals,
                                significance testing, and domain-level performance breakdown.
                            </p>
                            {_get_report_link(available_reports, 'statistical_analysis')}
                        </div>
                    </div>
                </div>

                <!-- Forensic Analysis -->
                <div class="col-md-4 mb-4">
                    <div class="card h-100 metric-card">
                        <div class="card-body">
                            <h3 class="card-title">
                                <i class="fas fa-bug me-2" style="color: {BRAND_COLORS['danger']};"></i>
                                Forensic Analysis
                            </h3>
                            <p class="card-text">
                                Deep-dive investigation of penalty cases. Includes question-by-question
                                analysis, competitive comparison, and failure pattern identification.
                            </p>
                            {_get_forensic_links(available_reports, providers)}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Key Findings -->
            <h2 class="section-header">
                <i class="fas fa-lightbulb me-2"></i>Key Findings
            </h2>

            <div class="alert alert-success">
                <h5><strong><i class="fas fa-check-circle me-2"></i>Quality Strategy Validates RAG Architectures</strong></h5>
                <p>
                    The penalty-aware scoring methodology reveals that well-calibrated RAG systems
                    achieve high accuracy <em>without</em> overconfident hallucinations. Traditional
                    volume-focused metrics can mislead by rewarding guessing behavior over appropriate
                    uncertainty.
                </p>
            </div>

            <div class="alert alert-warning">
                <h5><strong><i class="fas fa-exclamation-triangle me-2"></i>The "I Don't Know" Tax is Real</strong></h5>
                <p>
                    Systems that appropriately abstain when uncertain are penalized under traditional
                    scoring (volume strategy). This creates perverse incentives for RAG systems to
                    guess rather than acknowledge knowledge gaps, leading to increased hallucinations.
                </p>
            </div>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center text-muted">
                <p>
                    <strong>SimpleEvals RAG Benchmark</strong> |
                    <a href="https://github.com/openai/simple-evals" target="_blank">GitHub Repository</a> |
                    <a href="https://openai.com/index/why-language-models-hallucinate/" target="_blank">Research Paper</a>
                </p>
                <p>
                    Generated: {format_timestamp(timestamp)}<br>
                    Run ID: <code>{run_id}</code>
                </p>
            </div>
        </div>
    </div>
</body>
</html>"""

    # Write to file
    output_file = os.path.join(results_dir, "index.html")
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Main dashboard generated: {output_file}")
    return output_file


def scan_available_reports(results_dir: str) -> dict:
    """
    Scan results directory for available reports.

    Returns:
        Dictionary mapping report types to file paths
    """
    reports = {
        'quality_benchmark': None,
        'statistical_analysis': None,
        'forensics': {}
    }

    # Find quality benchmark report
    for file in Path(results_dir).glob("quality_benchmark_report_*.html"):
        reports['quality_benchmark'] = file.name
        break

    # Find statistical analysis report
    for file in Path(results_dir).glob("statistical_analysis_run_*.html"):
        reports['statistical_analysis'] = file.name
        break

    # Find forensic dashboards
    for forensic_dir in Path(results_dir).glob("*_forensics"):
        provider = forensic_dir.name.replace("_forensics", "")
        dashboard = forensic_dir / "forensic_dashboard.html"
        if dashboard.exists():
            reports['forensics'][provider] = f"{forensic_dir.name}/forensic_dashboard.html"

    return reports


def _get_report_link(reports: dict, report_type: str) -> str:
    """Generate HTML link to report if available"""
    report_path = reports.get(report_type)

    if report_path:
        return f'''
            <a href="{report_path}" class="btn btn-primary w-100">
                <i class="fas fa-arrow-right me-2"></i>View Report
            </a>'''
    else:
        return '''
            <button class="btn btn-secondary w-100" disabled>
                <i class="fas fa-clock me-2"></i>Report Not Available
            </button>'''


def _get_forensic_links(reports: dict, providers: list) -> str:
    """Generate HTML links to forensic dashboards"""
    forensics = reports.get('forensics', {})

    if not forensics:
        return '''
            <button class="btn btn-secondary w-100" disabled>
                <i class="fas fa-clock me-2"></i>Forensics Not Available
            </button>'''

    links = []
    for provider in providers:
        provider_key = provider.lower().replace("_", "")
        if provider_key in forensics or provider in forensics:
            path = forensics.get(provider_key) or forensics.get(provider)
            links.append(f'''
                <a href="{path}" class="btn btn-sm btn-outline-primary mb-2 w-100">
                    <i class="fas fa-microscope me-2"></i>{provider} Forensics
                </a>''')

    return '<div class="d-grid gap-2">' + ''.join(links) + '</div>'


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate main dashboard hub")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--run-id", default="", help="Run ID")
    parser.add_argument("--providers", nargs="+", default=["CustomGPT", "OpenAI_RAG", "OpenAI_Vanilla"])
    parser.add_argument("--questions", type=int, default=200, help="Number of questions")
    parser.add_argument("--threshold", type=float, default=0.8, help="Confidence threshold")

    args = parser.parse_args()

    metadata = {
        "run_id": args.run_id,
        "timestamp": datetime.now(),
        "providers": args.providers,
        "total_questions": args.questions,
        "confidence_threshold": args.threshold
    }

    generate_main_dashboard(args.results_dir, metadata)
