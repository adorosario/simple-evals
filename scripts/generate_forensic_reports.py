#!/usr/bin/env python3
"""
Forensic Report Generator
Creates comprehensive browser-viewable forensic reports for penalty analysis.

This script generates:
1. Main forensic dashboard with links to all reports
2. Individual HTML reports for each failed question
3. HTML conversion of engineering report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import markdown
import re


def get_confidence_display_properties(confidence: float) -> dict:
    """
    Get Bootstrap styling properties for judge confidence level visualization

    Args:
        confidence: Float between 0.0 and 1.0 representing judge confidence

    Returns:
        dict with: badge_color, progress_class, interpretation, icon

    Confidence Levels:
    - 0.90-1.00: Very High (success/green)
    - 0.75-0.89: High (info/blue)
    - 0.60-0.74: Medium (warning/yellow)
    - 0.00-0.59: Low (danger/red)
    """
    if confidence >= 0.90:
        return {
            "badge_color": "success",
            "progress_class": "bg-success",
            "interpretation": "⭐ Very High Confidence - Judge is very certain about this evaluation",
            "icon": "fa-check-circle",
            "level": "VERY_HIGH"
        }
    elif confidence >= 0.75:
        return {
            "badge_color": "info",
            "progress_class": "bg-info",
            "interpretation": "✓ High Confidence - Judge is confident about this evaluation",
            "icon": "fa-info-circle",
            "level": "HIGH"
        }
    elif confidence >= 0.60:
        return {
            "badge_color": "warning",
            "progress_class": "bg-warning text-dark",
            "interpretation": "⚠ Medium Confidence - Judge has some uncertainty about this evaluation",
            "icon": "fa-exclamation-triangle",
            "level": "MEDIUM"
        }
    else:
        return {
            "badge_color": "danger",
            "progress_class": "bg-danger",
            "interpretation": "❗ Low Confidence - Judge is uncertain, may need human review",
            "icon": "fa-exclamation-circle",
            "level": "LOW"
        }


def extract_customgpt_metadata(audit_log_path: Path, question_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract CustomGPT metadata (IDs and citations) from audit logs

    Args:
        audit_log_path: Path to provider_requests.jsonl audit log
        question_id: Question ID to search for (e.g., 'simpleqa_0016')

    Returns:
        dict: CustomGPT metadata including prompt_id, citations, debug URLs
        None: If no CustomGPT request found for this question
    """
    if not audit_log_path.exists():
        return None

    try:
        with open(audit_log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())

                    # Match CustomGPT provider for this specific question
                    if (log_entry.get('provider') == 'CustomGPT_RAG' and
                        log_entry.get('question_id') == question_id):

                        metadata = log_entry.get('metadata', {})

                        return {
                            'project_id': metadata.get('project_id'),
                            'session_id': metadata.get('session_id'),
                            'conversation_id': metadata.get('conversation_id'),
                            'prompt_id': metadata.get('prompt_id'),
                            'message_id': metadata.get('message_id'),
                            'external_id': metadata.get('external_id'),
                            'citations': metadata.get('citations', []),
                            'citation_count': metadata.get('citation_count', 0),
                            'debug_urls': metadata.get('debug_urls', {})
                        }
                except json.JSONDecodeError:
                    continue

        return None
    except Exception as e:
        print(f"Warning: Failed to extract CustomGPT metadata: {e}")
        return None


def load_all_data(run_dir: str, provider: str = "customgpt") -> Dict[str, Any]:
    """
    Load all available data sources

    Adapts data from actual penalty analysis schema to expected format.
    Makes GPT-5 analysis optional for fault tolerance.
    """
    run_path = Path(run_dir)
    run_id = run_path.name

    # Load penalty analysis (REQUIRED) - try multiple locations
    penalty_file = None
    possible_penalty_locations = [
        run_path / f"{provider}_penalty_analysis" / f"{provider}_penalty_analysis_{run_id}.json",
        Path(f"{provider}_penalty_analysis") / f"{provider}_penalty_analysis_{run_id}.json"
    ]

    for location in possible_penalty_locations:
        if location.exists():
            penalty_file = location
            break

    if not penalty_file:
        raise FileNotFoundError(
            f"Penalty analysis file not found. This is required.\n"
            f"Tried: {[str(f) for f in possible_penalty_locations]}"
        )

    with open(penalty_file, 'r') as f:
        penalty_raw = json.load(f)

    # Adapt penalty analysis schema
    # Actual schema: {metadata, penalty_cases, analysis}
    # Expected schema: {penalty_analysis: {penalty_cases: [...]}}
    penalty_data = {
        'penalty_cases': penalty_raw.get('penalty_cases', []),
        'metadata': penalty_raw.get('metadata', {}),
        'summary_analysis': penalty_raw.get('analysis', '')
    }

    # Load GPT-5 analysis (OPTIONAL) - try multiple locations
    gpt5_data = None
    possible_gpt5_locations = [
        run_path / f"{provider}_penalty_analysis" / f"gpt5_failure_analysis_{run_id}.json",
        Path(f"{provider}_penalty_analysis") / f"gpt5_failure_analysis_{run_id}.json"
    ]

    for location in possible_gpt5_locations:
        if location.exists():
            try:
                with open(location, 'r') as f:
                    gpt5_data = json.load(f)
                print(f"✓ Loaded GPT-5 analysis from {location}")
                break
            except Exception as e:
                print(f"⚠️  Warning: Failed to load GPT-5 analysis: {e}")

    if gpt5_data is None:
        print(f"⚠️  No GPT-5 analysis found (optional)")
        print(f"   Tried: {[str(f) for f in possible_gpt5_locations]}")
        # Create empty stub for compatibility
        gpt5_data = {'failure_analyses': []}

    # Load run metadata (OPTIONAL)
    metadata = {}
    metadata_file = run_path / "run_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load run metadata: {e}")

    return {
        'gpt5_analysis': gpt5_data,
        'penalty_analysis': penalty_data,
        'metadata': metadata,
        'run_id': run_id
    }

def get_html_template() -> str:
    """Get the base HTML template with Bootstrap styling"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>

    <!-- External Dependencies -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <!-- JavaScript Dependencies -->
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <style>
        .question-card {{ border-left: 4px solid #dc3545; }}
        .metric-card {{ border-left: 4px solid #0d6efd; }}
        .analysis-section {{ background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .grade-badge {{ font-size: 1.1em; }}
        .confidence-bar {{ height: 20px; border-radius: 10px; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-size: 0.9em; }}
        .forensic-nav {{ background-color: #212529; }}
        .forensic-nav .nav-link {{ color: #adb5bd; }}
        .forensic-nav .nav-link:hover {{ color: #fff; }}
        .forensic-nav .nav-link.active {{ color: #fff; background-color: #495057; }}
    </style>
</head>
<body>
    {content}

    <script>
        $(document).ready(function() {{
            $('.table').DataTable({{
                pageLength: 25,
                order: [[0, 'asc']],
                responsive: true
            }});
        }});
    </script>
</body>
</html>
"""

def generate_forensic_dashboard(data: Dict[str, Any], output_file: str, run_dir: str, provider: str = "customgpt") -> str:
    """Generate the main forensic dashboard"""
    penalty_cases = data['penalty_analysis']['penalty_cases']
    run_id = data['run_id']

    # Dynamically find quality benchmark report
    quality_report_file = None
    quality_report_link = ""
    run_path = Path(run_dir)
    for html_file in run_path.glob("quality_benchmark_report_*.html"):
        quality_report_file = html_file.name
        quality_report_link = f'<li class="list-group-item"><a href="{quality_report_file}" class="text-decoration-none"><i class="fas fa-chart-bar"></i> Quality Benchmark Report</a></li>'
        break

    # Calculate summary statistics
    total_failures = len(penalty_cases)
    total_penalty_points = sum(case['penalty_points'] for case in penalty_cases)
    avg_confidence = sum(case['customgpt_confidence'] for case in penalty_cases) / len(penalty_cases)

    # Domain breakdown
    domain_counts = {}
    for case in penalty_cases:
        domain = case['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Competitor success rates
    openai_rag_success = sum(1 for case in penalty_cases if case['openai_rag_grade'] == 'A')
    openai_vanilla_success = sum(1 for case in penalty_cases if case['openai_vanilla_grade'] == 'A')

    content = f"""
    <nav class="navbar navbar-expand-lg forensic-nav">
        <div class="container-fluid">
            <a class="navbar-brand text-white" href="#">
                <i class="fas fa-microscope"></i> {provider.upper()} Forensic Analysis
            </a>
            <span class="navbar-text text-light">Run ID: {run_id}</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">
                    <i class="fas fa-exclamation-triangle text-danger"></i>
                    {provider.upper()} Penalty Case Forensic Analysis
                </h1>
                <p class="lead">Complete forensic investigation of all {total_failures} penalty cases from evaluation run {run_id}</p>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-danger">{total_failures}</h2>
                        <p class="card-text">Total Failures</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-danger">{total_penalty_points:.1f}</h2>
                        <p class="card-text">Penalty Points</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-warning">{avg_confidence:.3f}</h2>
                        <p class="card-text">Avg Confidence</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-info">{openai_rag_success + openai_vanilla_success}/{total_failures * 2}</h2>
                        <p class="card-text">Competitor Successes</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Navigation to Individual Reports -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-file-medical"></i> Individual Question Forensic Reports</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
"""

    # Add links to individual question reports
    for i, case in enumerate(penalty_cases):
        question_id = case['question_id']
        domain = case['domain']
        confidence = case['customgpt_confidence']

        content += f"""
                            <div class="col-md-4 mb-3">
                                <div class="card question-card h-100">
                                    <div class="card-body">
                                        <h6 class="card-title">{question_id}</h6>
                                        <p class="card-text small">
                                            <strong>Domain:</strong> {domain.title()}<br>
                                            <strong>Confidence:</strong> {confidence:.3f}<br>
                                            <strong>Question:</strong> {case['question'][:80]}...
                                        </p>
                                        <a href="forensic_question_{question_id}.html" class="btn btn-sm btn-danger">
                                            <i class="fas fa-search"></i> View Forensic Report
                                        </a>
                                    </div>
                                </div>
                            </div>
"""

    content += """
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- All Available Reports -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-folder-open"></i> All Available Reports</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5><i class="fas fa-file-alt"></i> Main Reports</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">
                                        <a href="{provider}_engineering_report.html" class="text-decoration-none">
                                            <i class="fas fa-tools"></i> Engineering Post-Mortem Report (HTML)
                                        </a>
                                    </li>
                                    <li class="list-group-item">
                                        <a href="{provider}_engineering_report_{run_id}.md" class="text-decoration-none">
                                            <i class="fab fa-markdown"></i> Engineering Post-Mortem Report (Markdown)
                                        </a>
                                    </li>
                                    {quality_report_link}
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5><i class="fas fa-database"></i> Raw Data Files</h5>
                                <ul class="list-group">
                                    <li class="list-group-item">
                                        <a href="{provider}_penalty_analysis/gpt5_failure_analysis_{run_id}.json" class="text-decoration-none">
                                            <i class="fas fa-brain"></i> GPT-5 Failure Analysis (JSON)
                                        </a>
                                    </li>
                                    <li class="list-group-item">
                                        <a href="{provider}_penalty_analysis/{provider}_penalty_analysis_{run_id}.json" class="text-decoration-none">
                                            <i class="fas fa-exclamation-circle"></i> Penalty Analysis Data (JSON)
                                        </a>
                                    </li>
                                    <li class="list-group-item">
                                        <a href="{provider}_penalty_analysis/{provider}_penalty_cases_{run_id}.csv" class="text-decoration-none">
                                            <i class="fas fa-table"></i> Penalty Cases (CSV)
                                        </a>
                                    </li>
                                    <li class="list-group-item">
                                        <a href="run_metadata.json" class="text-decoration-none">
                                            <i class="fas fa-info-circle"></i> Run Metadata (JSON)
                                        </a>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Domain Breakdown -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-chart-pie"></i> Failure Analysis by Domain</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
"""

    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_failures) * 100
        content += f"""
                            <div class="col-md-3 mb-3">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h4 class="text-danger">{count}</h4>
                                        <p class="card-text">
                                            <strong>{domain.title()}</strong><br>
                                            {percentage:.1f}% of failures<br>
                                            {count * 4.0:.1f} penalty points
                                        </p>
                                    </div>
                                </div>
                            </div>
"""

    content += f"""
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Overview Table -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-table"></i> All Penalty Cases Overview</h3>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Question ID</th>
                                    <th>Domain</th>
                                    <th>Question</th>
                                    <th>Confidence</th>
                                    <th>OpenAI RAG</th>
                                    <th>OpenAI Vanilla</th>
                                    <th>Penalty Points</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
"""

    for case in penalty_cases:
        question_preview = case['question'][:60] + "..." if len(case['question']) > 60 else case['question']
        confidence_color = "danger" if case['customgpt_confidence'] > 0.95 else "warning"

        content += f"""
                                <tr>
                                    <td><code>{case['question_id']}</code></td>
                                    <td><span class="badge bg-secondary">{case['domain'].title()}</span></td>
                                    <td>{question_preview}</td>
                                    <td>
                                        <span class="badge bg-{confidence_color}">{case['customgpt_confidence']:.3f}</span>
                                    </td>
                                    <td>
                                        <span class="badge bg-{'success' if case['openai_rag_grade'] == 'A' else 'danger'}">{case['openai_rag_grade']}</span>
                                    </td>
                                    <td>
                                        <span class="badge bg-{'success' if case['openai_vanilla_grade'] == 'A' else 'danger'}">{case['openai_vanilla_grade']}</span>
                                    </td>
                                    <td><span class="text-danger fw-bold">{case['penalty_points']:.1f}</span></td>
                                    <td>
                                        <a href="forensic_question_{case['question_id']}.html" class="btn btn-sm btn-outline-danger">
                                            <i class="fas fa-microscope"></i> Forensic
                                        </a>
                                    </td>
                                </tr>
"""

    content += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-light text-center py-3 mt-5">
        <p>&copy; 2025 {provider.upper()} Forensic Analysis - Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </footer>
"""

    # Generate the full HTML
    html_content = get_html_template().format(
        title=f"{provider.upper()} Forensic Dashboard - Penalty Case Analysis",
        content=content
    )

    # Save the file
    with open(output_file, 'w') as f:
        f.write(html_content)

    return output_file

def generate_individual_question_report(question_data: Dict, gpt5_analysis: Dict, output_file: str, audit_log_path: Optional[Path] = None) -> str:
    """Generate detailed forensic report for a single question"""
    question_id = question_data['question_id']

    # Extract GPT-5 analysis for this question
    gpt5_content = None
    for analysis in gpt5_analysis['failure_analyses']:
        if analysis['question_id'] == question_id:
            gpt5_content = analysis['analysis']
            break

    if not gpt5_content:
        gpt5_content = "GPT-5 analysis not available for this question."

    # Convert markdown-like content to HTML
    gpt5_html = markdown.markdown(gpt5_content)

    # Extract judge confidence and get visual properties
    judge_confidence = question_data.get('judge_confidence', question_data.get('customgpt_confidence', 0.0))
    conf_props = get_confidence_display_properties(judge_confidence)

    # Extract CustomGPT metadata (IDs and citations) from audit logs
    customgpt_meta = None
    if audit_log_path:
        customgpt_meta = extract_customgpt_metadata(audit_log_path, question_id)

    content = f"""
    <nav class="navbar navbar-expand-lg forensic-nav">
        <div class="container-fluid">
            <a class="navbar-brand text-white" href="forensic_dashboard.html">
                <i class="fas fa-arrow-left"></i> Back to Dashboard
            </a>
            <span class="navbar-text text-light">Question: {question_id}</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">
                    <i class="fas fa-microscope text-danger"></i>
                    Forensic Analysis: {question_id}
                </h1>
                <p class="lead">Complete forensic investigation of penalty case {question_id}</p>
            </div>
        </div>

        <!-- Question Details -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card question-card">
                    <div class="card-header">
                        <h3><i class="fas fa-question-circle"></i> Question Details</h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>Question ID:</strong> <code>{question_data['question_id']}</code></p>
                                <p><strong>Domain:</strong> <span class="badge bg-secondary">{question_data['domain'].title()}</span></p>
                                <p><strong>Complexity:</strong> {question_data['complexity']:.3f}</p>
                                <p><strong>Penalty Points:</strong> <span class="text-danger fw-bold">{question_data['penalty_points']:.1f}</span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>CustomGPT Confidence:</strong>
                                    <span class="badge bg-{'danger' if question_data['customgpt_confidence'] > 0.95 else 'warning'}">{question_data['customgpt_confidence']:.3f}</span>
                                </p>
                                <p><strong>CustomGPT Grade:</strong> <span class="badge bg-danger grade-badge">{question_data['customgpt_grade']}</span></p>
                                <p><strong>Penalty Type:</strong> {question_data['penalty_type'].replace('_', ' ').title()}</p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <h5>Question:</h5>
                            <div class="alert alert-info">
                                {question_data['question']}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Answer Comparison -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-balance-scale"></i> Answer Comparison</h3>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Provider</th>
                                        <th>Answer</th>
                                        <th>Grade</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="table-success">
                                        <td><strong>Target Answer</strong></td>
                                        <td><em>{question_data['target_answer']}</em></td>
                                        <td><span class="badge bg-success">CORRECT</span></td>
                                        <td><i class="fas fa-check text-success"></i> Gold Standard</td>
                                    </tr>
                                    <tr class="table-danger">
                                        <td><strong>CustomGPT</strong></td>
                                        <td>{question_data['customgpt_answer']}</td>
                                        <td><span class="badge bg-danger">{question_data['customgpt_grade']}</span></td>
                                        <td><i class="fas fa-times text-danger"></i> FAILED (-4.0 points)</td>
                                    </tr>
                                    <tr class="{'table-success' if question_data['openai_rag_grade'] == 'A' else 'table-danger'}">
                                        <td><strong>OpenAI RAG</strong></td>
                                        <td>{question_data.get('openai_rag_answer', 'Not available')}</td>
                                        <td><span class="badge bg-{'success' if question_data['openai_rag_grade'] == 'A' else 'danger'}">{question_data['openai_rag_grade']}</span></td>
                                        <td><i class="fas fa-{'check' if question_data['openai_rag_grade'] == 'A' else 'times'} text-{'success' if question_data['openai_rag_grade'] == 'A' else 'danger'}"></i> {'PASSED' if question_data['openai_rag_grade'] == 'A' else 'FAILED'}</td>
                                    </tr>
                                    <tr class="{'table-success' if question_data['openai_vanilla_grade'] == 'A' else 'table-danger'}">
                                        <td><strong>OpenAI Vanilla</strong></td>
                                        <td>{question_data.get('openai_vanilla_answer', 'Not available')}</td>
                                        <td><span class="badge bg-{'success' if question_data['openai_vanilla_grade'] == 'A' else 'danger'}">{question_data['openai_vanilla_grade']}</span></td>
                                        <td><i class="fas fa-{'check' if question_data['openai_vanilla_grade'] == 'A' else 'times'} text-{'success' if question_data['openai_vanilla_grade'] == 'A' else 'danger'}"></i> {'PASSED' if question_data['openai_vanilla_grade'] == 'A' else 'FAILED'}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""

    # Add CustomGPT Server-Side Debugging Section (conditional)
    if customgpt_meta:
        content += f"""
        <!-- CustomGPT Server-Side Debugging Info -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h3><i class="fas fa-server"></i> CustomGPT Server-Side Debugging</h3>
                        <small>IDs and Citations for server-side investigation</small>
                    </div>
                    <div class="card-body">
                        <!-- CustomGPT IDs -->
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <h5><i class="fas fa-fingerprint"></i> CustomGPT API IDs</h5>
                                <table class="table table-sm table-bordered">
                                    <tbody>
                                        <tr>
                                            <td><strong>Project ID:</strong></td>
                                            <td><code>{customgpt_meta['project_id'] or 'N/A'}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Session ID:</strong></td>
                                            <td><code>{customgpt_meta['session_id'] or 'N/A'}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Prompt ID:</strong></td>
                                            <td><code>{customgpt_meta['prompt_id'] or 'N/A'}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>External ID:</strong></td>
                                            <td><code>{customgpt_meta['external_id'] or 'N/A'}</code></td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h5><i class="fas fa-link"></i> Debug API Endpoints</h5>
                                <p class="text-muted small">Use these URLs to query the CustomGPT API directly</p>
"""
        if customgpt_meta['debug_urls'].get('message_endpoint'):
            content += f"""
                                <div class="mb-2">
                                    <strong>Message Endpoint:</strong><br>
                                    <a href="{customgpt_meta['debug_urls']['message_endpoint']}" target="_blank" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-external-link-alt"></i> View Message
                                    </a>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="navigator.clipboard.writeText('{customgpt_meta['debug_urls']['message_endpoint']}')">
                                        <i class="fas fa-copy"></i> Copy URL
                                    </button>
                                </div>
"""
        if customgpt_meta['debug_urls'].get('conversation_endpoint'):
            content += f"""
                                <div class="mb-2">
                                    <strong>Conversation Endpoint:</strong><br>
                                    <a href="{customgpt_meta['debug_urls']['conversation_endpoint']}" target="_blank" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-external-link-alt"></i> View Conversation
                                    </a>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="navigator.clipboard.writeText('{customgpt_meta['debug_urls']['conversation_endpoint']}')">
                                        <i class="fas fa-copy"></i> Copy URL
                                    </button>
                                </div>
"""

        content += """
                            </div>
                        </div>

                        <!-- Citations -->
                        <div class="row">
                            <div class="col-12">
                                <h5><i class="fas fa-quote-right"></i> Knowledge Base Citations</h5>
"""
        if customgpt_meta['citation_count'] > 0:
            content += f"""
                                <div class="alert alert-info">
                                    <i class="fas fa-info-circle"></i> CustomGPT returned <strong>{customgpt_meta['citation_count']} citation(s)</strong> from the knowledge base
                                </div>
                                <table class="table table-sm table-striped">
                                    <thead class="table-dark">
                                        <tr>
                                            <th>#</th>
                                            <th>Citation Data</th>
                                        </tr>
                                    </thead>
                                    <tbody>
"""
            for idx, citation in enumerate(customgpt_meta['citations'], 1):
                citation_json = json.dumps(citation, indent=2)
                content += f"""
                                        <tr>
                                            <td>{idx}</td>
                                            <td><pre class="mb-0"><code>{citation_json}</code></pre></td>
                                        </tr>
"""
            content += """
                                    </tbody>
                                </table>
"""
        else:
            content += """
                                <div class="alert alert-warning">
                                    <i class="fas fa-exclamation-triangle"></i> <strong>No citations found!</strong> CustomGPT did not return any knowledge base citations for this answer.
                                    <br><small>This may indicate the answer was generated without retrieving relevant documents from the knowledge base.</small>
                                </div>
"""

        content += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
"""

    content += f"""
        <!-- Judge Evaluation with Confidence Visualization -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-light">
                        <h3><i class="fas fa-gavel"></i> LLM-As-A-Judge Evaluation</h3>
                        <small class="text-muted">Automated evaluation by {question_data.get('judge_model', 'gpt-5')}</small>
                    </div>
                    <div class="card-body">
                        <!-- Confidence Score with Visual Progress Bar -->
                        <div class="mb-4">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h5 class="mb-0">
                                    <i class="fas {conf_props['icon']}"></i> Judge Confidence:
                                </h5>
                                <span class="badge bg-{conf_props['badge_color']} fs-5 px-3 py-2">
                                    {judge_confidence:.1%}
                                </span>
                            </div>

                            <div class="progress" style="height: 30px; font-size: 14px;">
                                <div class="progress-bar {conf_props['progress_class']} progress-bar-striped"
                                     role="progressbar"
                                     style="width: {judge_confidence*100:.1f}%"
                                     aria-valuenow="{judge_confidence*100:.1f}"
                                     aria-valuemin="0"
                                     aria-valuemax="100">
                                    {judge_confidence:.1%}
                                </div>
                            </div>

                            <div class="alert alert-{conf_props['badge_color']} mt-2 mb-0" role="alert">
                                <small><i class="fas fa-info-circle"></i> {conf_props['interpretation']}</small>
                            </div>
                        </div>

                        <!-- Judge Reasoning -->
                        <div class="mt-4">
                            <h5><i class="fas fa-comment-dots"></i> Judge Reasoning:</h5>
                            <div class="alert alert-secondary">
                                <p class="mb-0" style="white-space: pre-wrap;">{question_data['judge_reasoning']}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- GPT-5 Forensic Analysis -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-brain"></i> GPT-5 Deep Forensic Analysis</h3>
                        <small class="text-muted">Advanced AI-powered failure analysis</small>
                    </div>
                    <div class="card-body analysis-section">
                        {gpt5_html}
                    </div>
                </div>
            </div>
        </div>

        <!-- Competitive Analysis Summary -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-chart-line"></i> Competitive Performance Summary</h3>
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h2 class="text-danger">FAILED</h2>
                                        <p><strong>CustomGPT</strong></p>
                                        <p>Confidence: {question_data['customgpt_confidence']:.3f}</p>
                                        <p>Penalty: -4.0 points</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h2 class="text-{'success' if question_data['openai_rag_grade'] == 'A' else 'danger'}">{'PASSED' if question_data['openai_rag_grade'] == 'A' else 'FAILED'}</h2>
                                        <p><strong>OpenAI RAG</strong></p>
                                        <p>Grade: {question_data['openai_rag_grade']}</p>
                                        <p>Status: {'Outperformed CustomGPT' if question_data['openai_rag_grade'] == 'A' else 'Also failed'}</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body">
                                        <h2 class="text-{'success' if question_data['openai_vanilla_grade'] == 'A' else 'danger'}">{'PASSED' if question_data['openai_vanilla_grade'] == 'A' else 'FAILED'}</h2>
                                        <p><strong>OpenAI Vanilla</strong></p>
                                        <p>Grade: {question_data['openai_vanilla_grade']}</p>
                                        <p>Status: {'Outperformed CustomGPT' if question_data['openai_vanilla_grade'] == 'A' else 'Also failed'}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Navigation -->
        <div class="row">
            <div class="col-12 text-center">
                <a href="forensic_dashboard.html" class="btn btn-primary">
                    <i class="fas fa-arrow-left"></i> Back to Dashboard
                </a>
                <a href="#top" class="btn btn-secondary">
                    <i class="fas fa-arrow-up"></i> Back to Top
                </a>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-light text-center py-3 mt-5">
        <p>&copy; 2025 CustomGPT Forensic Analysis - Question {question_id} - Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </footer>
"""

    # Generate the full HTML
    html_content = get_html_template().format(
        title=f"Forensic Analysis - {question_id}",
        content=content
    )

    # Save the file
    with open(output_file, 'w') as f:
        f.write(html_content)

    return output_file

def convert_engineering_report_to_html(run_dir: str, output_file: str, provider: str = "customgpt") -> str:
    """Convert the markdown engineering report to HTML"""
    md_file = Path(run_dir) / f"{provider}_engineering_report_{Path(run_dir).name}.md"

    with open(md_file, 'r') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    content = f"""
    <nav class="navbar navbar-expand-lg forensic-nav">
        <div class="container-fluid">
            <a class="navbar-brand text-white" href="forensic_dashboard.html">
                <i class="fas fa-arrow-left"></i> Back to Dashboard
            </a>
            <span class="navbar-text text-light">Engineering Report</span>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="card">
            <div class="card-body">
                {html_body}
            </div>
        </div>
    </div>

    <footer class="bg-dark text-light text-center py-3 mt-5">
        <p>&copy; 2025 {provider.upper()} Engineering Report - Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </footer>
"""

    # Generate the full HTML
    html_content = get_html_template().format(
        title=f"{provider.upper()} Engineering Post-Mortem Report",
        content=content
    )

    # Save the file
    with open(output_file, 'w') as f:
        f.write(html_content)

    return output_file


def generate_individual_question_json(question_data: Dict, gpt5_analysis: Dict, output_file: str, audit_log_path: Optional[Path] = None) -> str:
    """
    Generate machine-readable JSON file for a single question

    This JSON contains EVERYTHING needed for programmatic debugging:
    - Complete question and answer data
    - All CustomGPT IDs for API lookup
    - Citations array from CustomGPT knowledge base
    - Judge evaluation with confidence score
    - Competitor answers and grades
    - GPT-5 forensic analysis
    - Debug URLs for manual API access
    """
    question_id = question_data['question_id']

    # Extract GPT-5 analysis for this question
    gpt5_content = None
    for analysis in gpt5_analysis.get('failure_analyses', []):
        if analysis['question_id'] == question_id:
            gpt5_content = analysis['analysis']
            break

    if not gpt5_content:
        gpt5_content = "GPT-5 analysis not available for this question."

    # Extract judge confidence
    judge_confidence = question_data.get('judge_confidence', question_data.get('customgpt_confidence', 0.0))
    conf_props = get_confidence_display_properties(judge_confidence)

    # Extract CustomGPT metadata (IDs and citations) from audit logs
    customgpt_meta = None
    if audit_log_path:
        customgpt_meta = extract_customgpt_metadata(audit_log_path, question_id)

    # Build complete debug JSON structure
    debug_json = {
        # ============ QUESTION IDENTIFICATION ============
        "question_id": question_id,
        "question": question_data['question'],
        "target_answer": question_data['target_answer'],

        # ============ CUSTOMGPT IDS (for API debugging) ============
        "customgpt_ids": {
            "project_id": customgpt_meta['project_id'] if customgpt_meta else None,
            "session_id": customgpt_meta['session_id'] if customgpt_meta else None,
            "conversation_id": customgpt_meta['conversation_id'] if customgpt_meta else None,
            "prompt_id": customgpt_meta['prompt_id'] if customgpt_meta else None,
            "message_id": customgpt_meta['message_id'] if customgpt_meta else None,
            "external_id": customgpt_meta['external_id'] if customgpt_meta else None
        },

        # ============ CITATIONS (for KB debugging) ============
        "citations": customgpt_meta['citations'] if customgpt_meta else [],
        "citation_count": customgpt_meta['citation_count'] if customgpt_meta else 0,

        # ============ CUSTOMGPT RESPONSE ============
        "customgpt": {
            "answer": question_data['customgpt_answer'],
            "grade": question_data['customgpt_grade'],
            "confidence": question_data.get('customgpt_confidence', 0.0),
            "status": "FAILED" if question_data['customgpt_grade'] not in ['A'] else "PASSED"
        },

        # ============ COMPETITOR RESPONSES ============
        "competitors": {
            "openai_rag": {
                "answer": question_data.get('openai_rag_answer', 'N/A'),
                "grade": question_data.get('openai_rag_grade', 'N/A'),
                "status": "PASSED" if question_data.get('openai_rag_grade') == 'A' else "FAILED"
            },
            "openai_vanilla": {
                "answer": question_data.get('openai_vanilla_answer', 'N/A'),
                "grade": question_data.get('openai_vanilla_grade', 'N/A'),
                "status": "PASSED" if question_data.get('openai_vanilla_grade') == 'A' else "FAILED"
            }
        },

        # ============ JUDGE EVALUATION ============
        "judge": {
            "reasoning": question_data['judge_reasoning'],
            "confidence": judge_confidence,
            "confidence_level": conf_props['level'],  # VERY_HIGH, HIGH, MEDIUM, LOW
            "model": question_data.get('judge_model', 'gpt-5'),
            "evaluation_timestamp": question_data.get('evaluation_timestamp')
        },

        # ============ GPT-5 FORENSIC ANALYSIS ============
        "gpt5_forensic_analysis": gpt5_content,

        # ============ METADATA ============
        "metadata": {
            "domain": question_data['domain'],
            "complexity": question_data['complexity'],
            "penalty_points": question_data['penalty_points'],
            "penalty_type": question_data['penalty_type']
        },

        # ============ DEBUG URLS (for manual API access) ============
        "debug_urls": customgpt_meta['debug_urls'] if customgpt_meta else {
            "message_endpoint": None,
            "conversation_endpoint": None,
        },

        # ============ SCHEMA VERSION ============
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat()
    }

    # Write JSON with pretty printing for human readability
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(debug_json, f, indent=2, ensure_ascii=False)

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive forensic reports')
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--provider', default='customgpt', help='Provider name to analyze')
    parser.add_argument('--output-dir', help='Output directory for HTML files (default: run directory)')

    args = parser.parse_args()

    # Load all data
    print("Loading data...")
    data = load_all_data(args.run_dir, args.provider)

    # Default output to run directory to keep root clean
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.run_dir)

    output_dir.mkdir(exist_ok=True)

    # Generate main dashboard
    print("Generating forensic dashboard...")
    dashboard_file = output_dir / "forensic_dashboard.html"
    generate_forensic_dashboard(data, dashboard_file, args.run_dir, args.provider)
    print(f"✓ Dashboard saved to: {dashboard_file}")

    # Generate individual question reports (both HTML and JSON)
    print("Generating individual question forensic reports (HTML + JSON)...")
    penalty_cases = data['penalty_analysis']['penalty_cases']
    gpt5_analysis = data['gpt5_analysis']

    # Construct path to audit log for CustomGPT metadata extraction
    audit_log_path = Path(args.run_dir) / "provider_requests.jsonl"
    if not audit_log_path.exists():
        print(f"⚠️  Warning: Audit log not found at {audit_log_path}")
        print(f"   CustomGPT IDs and citations will not be available in reports")
        audit_log_path = None

    html_files = []
    json_files = []
    for case in penalty_cases:
        question_id = case['question_id']

        # Generate HTML report with CustomGPT metadata
        html_file = output_dir / f"forensic_question_{question_id}.html"
        generate_individual_question_report(case, gpt5_analysis, html_file, audit_log_path)
        html_files.append(html_file)

        # Generate JSON report with CustomGPT metadata
        json_file = output_dir / f"forensic_question_{question_id}.json"
        generate_individual_question_json(case, gpt5_analysis, json_file, audit_log_path)
        json_files.append(json_file)

        print(f"✓ Generated: {question_id} (HTML + JSON)")

    # Convert engineering report to HTML (OPTIONAL)
    eng_report_file = None
    try:
        print("Converting engineering report to HTML...")
        eng_report_file = output_dir / f"{args.provider}_engineering_report.html"
        convert_engineering_report_to_html(args.run_dir, eng_report_file, args.provider)
        print(f"✓ Engineering report saved to: {eng_report_file}")
    except FileNotFoundError as e:
        print(f"⚠️  Engineering report not found (optional): {e}")
        eng_report_file = None

    print(f"\n{'='*80}")
    print(f"🎉 Generated Complete Forensic Report System:")
    print(f"{'='*80}")
    print(f"   📊 Main dashboard: {dashboard_file}")
    print(f"   🔍 Individual HTML reports: {len(html_files)} files")
    print(f"   📄 Individual JSON reports: {len(json_files)} files (NEW)")
    print(f"   📋 Engineering report: {eng_report_file}")
    print(f"\n💡 For programmatic debugging:")
    print(f"   - JSON files: {output_dir}/forensic_question_*.json")
    print(f"   - Example: jq '.' {output_dir}/forensic_question_simpleqa_0001.json")
    print(f"\n🌐 Start HTTP server and navigate to: {dashboard_file.name}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()