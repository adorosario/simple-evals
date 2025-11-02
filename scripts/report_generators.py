"""
Report Generators Using Unified Brand Kit
=========================================

This module provides functions to generate HTML reports with consistent branding.
Replaces inline HTML generation with brand-kit-aware report generation.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brand_kit import (
    get_html_head,
    get_navigation_bar,
    get_page_header,
    format_timestamp,
    get_provider_badge_class,
    BRAND_COLORS
)


def _scan_available_reports(results_dir: str) -> dict:
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


def generate_quality_benchmark_report_v2(results, output_dir, run_metadata):
    """
    Generate provider-focused quality benchmark report using unified brand kit.

    This is the new version that replaces the old inline HTML generation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"quality_benchmark_report_{timestamp}.html")

    # Build detailed analysis
    successful_results = [r for r in results if r["success"]]
    run_id = run_metadata.get('run_id', '')

    # Scan for available reports for navigation
    available_reports = _scan_available_reports(output_dir)

    # Pre-compute values for template (avoid nested f-string issues)
    timestamp_str = format_timestamp()
    meta_info_str = f"Generated: {timestamp_str} | Run ID: <code style='color: white;'>{run_id}</code>"

    # Start HTML document with brand kit
    html = get_html_head(
        title="Why RAGs Hallucinate - Quality Benchmark",
        description="Quality vs Volume strategy comparison using OpenAI's penalty-aware scoring methodology"
    )

    html += f"""
<body>
    {get_navigation_bar(
        active_page='quality',
        run_id=run_id,
        base_path="",
        quality_report=available_reports.get('quality_benchmark'),
        statistical_report=available_reports.get('statistical_analysis'),
        forensic_reports=available_reports.get('forensics', {})
    )}

    <div class="main-container">
        {get_page_header(
            title="Why RAGs Hallucinate",
            subtitle="Quality Benchmark Report - Penalty-Aware Scoring Analysis",
            meta_info=meta_info_str
        )}

        <!-- Content Section -->
        <div class="content-section">
            <!-- Executive Summary -->
            <div class="metric-grid">
                <div class="metric-card">
                    <h3><i class="fas fa-users me-2"></i>Providers Tested</h3>
                    <div class="value">{len(successful_results)}</div>
                    <div class="description">RAG providers evaluated</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-question-circle me-2"></i>Questions Per Provider</h3>
                    <div class="value">{run_metadata.get('samples_per_provider', 'N/A')}</div>
                    <div class="description">SimpleQA benchmark questions</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-calculator me-2"></i>Total Evaluations</h3>
                    <div class="value">{run_metadata.get('actual_total_evaluations', 'N/A')}</div>
                    <div class="description">Questions × Providers</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-check-circle me-2"></i>Coverage Status</h3>
                    <div class="value" style="color: {'var(--success)' if run_metadata.get('evaluation_coverage_complete', False) else 'var(--warning)'};">
                        {'✅ Complete' if run_metadata.get('evaluation_coverage_complete', False) else '⚠️ Partial'}
                    </div>
                    <div class="description">{run_metadata.get('actual_total_evaluations', 0)}/{run_metadata.get('expected_total_evaluations', 0)} evaluations</div>
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
                        GPT-4.1 (standardized)
                    </div>
                    <div class="col-md-3">
                        <strong><i class="fas fa-gavel me-1"></i>Judge Model:</strong><br>
                        GPT-5 {"Flex Tier" if run_metadata.get("use_flex_tier", False) else "Standard"}
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

            <!-- Provider Leaderboard -->
            <h2 class="section-header">
                <i class="fas fa-trophy me-2"></i>Provider Performance Leaderboard
            </h2>

            <div class="table-responsive">
                <table id="providerTable" class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Provider</th>
                            <th><i class="fas fa-trophy me-1" style="color: {BRAND_COLORS['quality']};"></i>Quality Score</th>
                            <th><i class="fas fa-chart-line me-1" style="color: {BRAND_COLORS['volume']};"></i>Volume Score</th>
                            <th>Correct</th>
                            <th>Wrong</th>
                            <th>Abstain</th>
                            <th>Penalty Points</th>
                        </tr>
                    </thead>
                    <tbody>"""

    # Sort providers by quality score (descending)
    sorted_results = sorted(successful_results, key=lambda x: x['metrics']['quality_score'], reverse=True)

    # Add provider data to table
    for rank, result in enumerate(sorted_results, 1):
        provider_name = result["sampler_name"]
        metrics = result["metrics"]

        # Determine score styling
        quality_score = metrics['quality_score']
        volume_score = metrics['volume_score']

        quality_class = "score-high" if quality_score > 0.6 else ("score-medium" if quality_score > 0 else "score-low")
        volume_class = "score-high" if volume_score > 0.8 else ("score-medium" if volume_score > 0.5 else "score-low")

        # Calculate counts (using actual field names from JSON)
        correct_count = metrics.get('n_correct', 0)
        wrong_count = metrics.get('n_incorrect', 0)
        abstain_count = metrics.get('n_not_attempted', 0)
        penalty_points = metrics.get('overconfidence_penalty', wrong_count * 4)

        # Provider badge
        provider_badge_class = get_provider_badge_class(provider_name)

        html += f"""
                            <tr>
                                <td><strong>#{rank}</strong></td>
                                <td><span class="{provider_badge_class}">{provider_name}</span></td>
                                <td><span class="{quality_class}">{quality_score:.3f}</span></td>
                                <td><span class="{volume_class}">{volume_score:.3f}</span></td>
                                <td><span class="score-high">{correct_count}</span></td>
                                <td><span class="score-low">{wrong_count}</span></td>
                                <td><span style="color: {BRAND_COLORS['text_secondary']};">{abstain_count}</span></td>
                                <td><span class="score-low">-{penalty_points}</span></td>
                            </tr>"""

    html += """
                    </tbody>
                </table>
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
            </div>
        </div>
    </div>

    <!-- JavaScript for Interactive Features -->
    <script>
        // Initialize DataTables for provider leaderboard
        $(document).ready(function() {{
            $('#providerTable').DataTable({{
                responsive: true,
                pageLength: 25,
                order: [[2, 'desc']], // Sort by Quality Score descending
                columnDefs: [
                    {{ targets: [2, 3], type: 'num' }}
                ],
                language: {{
                    search: "Search providers:",
                    lengthMenu: "Show _MENU_ entries"
                }}
            }});
        }});
    </script>
</body>
</html>"""

    # Write to file
    with open(report_file, 'w') as f:
        f.write(html)

    print(f"✅ Quality benchmark report generated: {report_file}")
    return report_file


def generate_statistical_analysis_report_v2(output_dir, run_id, analysis_data):
    """
    Generate statistical analysis report using unified brand kit.

    Args:
        output_dir: Directory to write report
        run_id: Run identifier
        analysis_data: Dictionary with statistical analysis results
    """
    report_file = os.path.join(output_dir, f"statistical_analysis_run_{run_id}.html")

    html = get_html_head(
        title="Why RAGs Hallucinate - Statistical Analysis",
        description="Academic-grade statistical analysis with confidence intervals and significance testing"
    )

    html += f"""
<body>
    {get_navigation_bar(active_page='statistical', run_id=run_id)}

    <div class="main-container">
        {get_page_header(
            title="Statistical Analysis Report",
            subtitle="Wilson Score Confidence Intervals & Significance Testing",
            meta_info=f"Run ID: <code>{run_id}</code> | Generated: {format_timestamp()}"
        )}

        <div class="content-section">
            <h2 class="section-header">
                <i class="fas fa-chart-bar me-2"></i>Executive Summary
            </h2>

            <div class="info-box">
                <p><strong>Statistical Methodology:</strong> Wilson Score Confidence Intervals (95% CI) for binomial proportions,
                conservative significance testing via CI overlap.</p>
                <p><strong>Benchmark Configuration:</strong> {analysis_data.get('provider_count', 'N/A')} providers evaluated
                on {analysis_data.get('question_count', 'N/A')} questions at 80% confidence threshold.</p>
            </div>

            <!-- Analysis Content -->
            <div class="mt-4">
                {analysis_data.get('html_content', '<p>No analysis data available</p>')}
            </div>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center text-muted">
                <p>Generated: {format_timestamp()} | Run ID: <code>{run_id}</code></p>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open(report_file, 'w') as f:
        f.write(html)

    print(f"✅ Statistical analysis report generated: {report_file}")
    return report_file


def generate_forensic_dashboard_v2(penalty_cases, provider, output_dir, run_id):
    """
    Generate forensic dashboard for a provider using unified brand kit.

    Args:
        penalty_cases: List of penalty case dictionaries
        provider: Provider name (e.g., 'CustomGPT', 'OpenAI_RAG')
        output_dir: Directory to write the dashboard
        run_id: Run identifier for navigation
    """
    total_failures = len(penalty_cases)
    total_penalty = total_failures * 4  # Each failure = 4 penalty points

    provider_display = provider.replace('_', ' ')

    html = get_html_head(
        title=f"Why RAGs Hallucinate - {provider_display} Forensics",
        description=f"Forensic analysis of {provider_display} penalty cases"
    )

    html += f"""
<body>
    {get_navigation_bar(active_page='forensic', run_id=run_id)}

    <div class="main-container">
        {get_page_header(
            title=f"{provider_display} Forensic Analysis",
            subtitle="Deep-dive investigation of all penalty cases",
            meta_info=f"Run ID: <code>{run_id}</code> | Generated: {format_timestamp()}"
        )}

        <div class="content-section">
            <!-- Executive Summary -->
            <div class="metric-grid">
                <div class="metric-card">
                    <h3><i class="fas fa-exclamation-triangle me-2"></i>Total Failures</h3>
                    <div class="value" style="color: var(--danger);">{total_failures}</div>
                    <div class="description">Wrong answers requiring investigation</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-calculator me-2"></i>Penalty Points</h3>
                    <div class="value" style="color: var(--danger);">{total_penalty}</div>
                    <div class="description">Total penalty impact (4 points each)</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-microscope me-2"></i>Cases Analyzed</h3>
                    <div class="value">{total_failures}</div>
                    <div class="description">Complete forensic investigations</div>
                </div>
            </div>

            <!-- Penalty Cases Table -->
            <h2 class="section-header">
                <i class="fas fa-table me-2"></i>All Penalty Cases
            </h2>

            <div class="table-responsive">
                <table class="table table-striped table-hover" id="penaltyTable">
                    <thead>
                        <tr>
                            <th>Question ID</th>
                            <th>Question</th>
                            <th>Grade</th>
                            <th>Penalty Points</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>"""

    # Add penalty cases
    for case in penalty_cases:
        question_id = case.get('question_id', 'N/A')
        question = case.get('question', 'N/A')
        grade = case.get('grade', 'F')

        # Truncate long questions
        if len(question) > 100:
            question = question[:100] + "..."

        html += f"""
                        <tr>
                            <td><code>{question_id}</code></td>
                            <td>{question}</td>
                            <td><span class="grade-badge grade-{grade}">{grade}</span></td>
                            <td><span class="score-low fw-bold">4.0</span></td>
                            <td>
                                <a href="forensic_question_{question_id}.html" class="btn btn-sm btn-outline-danger">
                                    <i class="fas fa-search me-1"></i>Investigate
                                </a>
                            </td>
                        </tr>"""

    html += """
                    </tbody>
                </table>
            </div>

            <!-- Key Insights -->
            <h2 class="section-header">
                <i class="fas fa-lightbulb me-2"></i>Key Insights
            </h2>

            <div class="alert alert-info">
                <h5><strong><i class="fas fa-info-circle me-2"></i>About This Analysis</strong></h5>
                <p>
                    Each penalty case represents a question where the provider gave an incorrect answer with high confidence.
                    Under the penalty-aware scoring system, these failures cost 4 points each, making them significantly
                    more costly than abstaining (0 points).
                </p>
                <p>
                    Click "Investigate" on any question to see:
                </p>
                <ul>
                    <li>Full question and correct answer</li>
                    <li>Provider's incorrect answer</li>
                    <li>Judge's evaluation and confidence</li>
                    <li>Competitive comparison with other providers</li>
                    <li>Provider-specific metadata (citations, confidence scores, etc.)</li>
                </ul>
            </div>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center text-muted">
                <p>
                    <strong>{provider_display} Forensic Analysis</strong> |
                    <a href="../index.html">Return to Main Dashboard</a>
                </p>
                <p>Generated: {format_timestamp()} | Run ID: <code>{run_id}</code></p>
            </div>
        </div>
    </div>

    <!-- JavaScript for DataTables -->
    <script>
        $(document).ready(function() {{
            $('#penaltyTable').DataTable({{
                responsive: true,
                pageLength: 25,
                order: [[0, 'asc']],
                language: {{
                    search: "Search penalty cases:",
                    lengthMenu: "Show _MENU_ cases"
                }}
            }});
        }});
    </script>
</body>
</html>"""

    # Write to file
    report_file = os.path.join(output_dir, "forensic_dashboard.html")
    with open(report_file, 'w') as f:
        f.write(html)

    print(f"✅ Forensic dashboard generated: {report_file}")
    return report_file


def generate_forensic_question_report_v2(question_data, provider, output_file, run_id):
    """
    Generate individual question forensic report using unified brand kit.

    Args:
        question_data: Dictionary with question details and all provider answers
        provider: Provider name being investigated
        output_file: Path to write the HTML file
        run_id: Run identifier for navigation
    """
    question_id = question_data.get('question_id', 'N/A')
    question = question_data.get('question', 'N/A')
    correct_answer = question_data.get('correct_answer', 'N/A')

    provider_display = provider.replace('_', ' ')
    provider_key = provider.lower().replace(' ', '_')

    # Get provider-specific data
    provider_answer = question_data.get(f'{provider_key}_answer', 'N/A')
    provider_grade = question_data.get(f'{provider_key}_grade', 'F')

    # Get competitive data
    competitors = ['customgpt', 'openai_rag', 'openai_vanilla']
    competitors = [c for c in competitors if c != provider_key]

    html = get_html_head(
        title=f"Why RAGs Hallucinate - {question_id}",
        description=f"Forensic analysis of {provider_display} failure on {question_id}"
    )

    html += f"""
<body>
    {get_navigation_bar(active_page='forensic', run_id=run_id)}

    <div class="main-container">
        {get_page_header(
            title=f"Forensic Analysis: {question_id}",
            subtitle=f"{provider_display} failure investigation with competitive analysis",
            meta_info=f"Run ID: <code>{run_id}</code> | Generated: {format_timestamp()}"
        )}

        <div class="content-section">
            <!-- Question Details -->
            <h2 class="section-header">
                <i class="fas fa-question-circle me-2"></i>Question Details
            </h2>

            <div class="info-box">
                <h5><strong>Question:</strong></h5>
                <p class="lead">{question}</p>

                <h5 class="mt-4"><strong>Correct Answer:</strong></h5>
                <p class="text-success fw-bold">{correct_answer}</p>
            </div>

            <!-- Provider's Answer -->
            <h2 class="section-header">
                <i class="fas fa-times-circle me-2" style="color: var(--danger);"></i>{provider_display}'s Answer
            </h2>

            <div class="alert alert-danger">
                <h5><strong>Grade: <span class="grade-badge grade-{provider_grade}">{provider_grade}</span></strong></h5>
                <p class="mt-3">{provider_answer}</p>
                <hr>
                <p class="mb-0"><strong>Penalty:</strong> <span class="score-low fw-bold">4.0 points</span></p>
            </div>

            <!-- Competitive Comparison -->
            <h2 class="section-header">
                <i class="fas fa-users me-2"></i>Competitive Comparison
            </h2>

            <div class="row">"""

    # Add competitor cards
    for competitor in competitors:
        comp_answer = question_data.get(f'{competitor}_answer', 'Not available')
        comp_grade = question_data.get(f'{competitor}_grade', 'N/A')
        comp_display = competitor.replace('_', ' ').title()

        grade_class = 'success' if comp_grade in ['A', 'B'] else ('warning' if comp_grade == 'C' else 'danger')

        html += f"""
                <div class="col-md-6 mb-3">
                    <div class="card h-100">
                        <div class="card-header bg-{grade_class} text-white">
                            <h5 class="mb-0">{comp_display}</h5>
                        </div>
                        <div class="card-body">
                            <p><strong>Grade:</strong> <span class="grade-badge grade-{comp_grade}">{comp_grade}</span></p>
                            <p><strong>Answer:</strong></p>
                            <p class="text-muted">{comp_answer}</p>
                        </div>
                    </div>
                </div>"""

    html += f"""
            </div>

            <!-- Analysis -->
            <h2 class="section-header">
                <i class="fas fa-microscope me-2"></i>Failure Analysis
            </h2>

            <div class="info-box">
                <h5><strong>Why This Matters</strong></h5>
                <p>
                    This penalty case demonstrates a critical failure mode: the system provided an incorrect answer
                    with apparent confidence, rather than abstaining or expressing uncertainty. Under penalty-aware
                    scoring, this costs <strong>4 points</strong> versus <strong>0 points</strong> for abstaining.
                </p>
                <p>
                    Comparing with competitors helps identify whether this was a systemic knowledge gap (all providers
                    failed) or a provider-specific issue (others succeeded).
                </p>
            </div>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center">
                <a href="forensic_dashboard.html" class="btn btn-primary me-2">
                    <i class="fas fa-arrow-left me-1"></i>Back to Forensic Dashboard
                </a>
                <a href="../index.html" class="btn btn-outline-secondary">
                    <i class="fas fa-home me-1"></i>Main Dashboard
                </a>
            </div>
            <div class="text-center text-muted mt-3">
                <p>Generated: {format_timestamp()} | Run ID: <code>{run_id}</code></p>
            </div>
        </div>
    </div>
</body>
</html>"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ Forensic question report generated: {output_file}")
    return output_file
