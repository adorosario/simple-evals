#!/usr/bin/env python3
"""
Explainability Post-Mortem Report Generator

Generates comprehensive HTML reports from explainability post-mortem analysis results.
Uses the unified brand kit for consistent Apple-inspired styling.

Usage:
    python scripts/generate_explainability_report.py --input results/run_XXX/explainability_postmortem/postmortem_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brand_kit import (
    get_html_head,
    get_page_header,
    wrap_html_document,
    format_timestamp,
    BRAND_COLORS
)


# =============================================================================
# Report Configuration
# =============================================================================

ROOT_CAUSE_ICONS = {
    "hallucination": "fa-ghost",
    "partial_knowledge": "fa-puzzle-piece",
    "kb_gap": "fa-hole",
    "retrieval_miss": "fa-search-minus",
    "reasoning_error": "fa-brain",
    "specificity_failure": "fa-bullseye",
    "unknown": "fa-question-circle"
}

ROOT_CAUSE_COLORS = {
    "hallucination": BRAND_COLORS["danger"],
    "partial_knowledge": BRAND_COLORS["warning"],
    "kb_gap": BRAND_COLORS["warning"],
    "retrieval_miss": BRAND_COLORS["info"],
    "reasoning_error": BRAND_COLORS["quality"],
    "specificity_failure": BRAND_COLORS["info"],
    "unknown": BRAND_COLORS["text_secondary"]
}

ROOT_CAUSE_DESCRIPTIONS = {
    "hallucination": "Model fabricated information without KB source",
    "partial_knowledge": "KB has related but incomplete information",
    "kb_gap": "Required information not in knowledge base",
    "retrieval_miss": "KB has info but retrieval failed to find it",
    "reasoning_error": "Retrieved correct info but drew wrong conclusion",
    "specificity_failure": "Got general answer when specific was needed",
    "unknown": "Unable to determine root cause"
}

PRIORITY_COLORS = {
    "HIGH": BRAND_COLORS["danger"],
    "MEDIUM": BRAND_COLORS["warning"],
    "LOW": BRAND_COLORS["info"]
}


# =============================================================================
# HTML Components
# =============================================================================

def get_custom_css() -> str:
    """Additional CSS specific to explainability reports"""
    return """
<style>
/* Root Cause Cards */
.root-cause-card {
    border-left: 4px solid var(--card-border-color);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.root-cause-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Claims List */
.claims-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.claim-item {
    padding: 12px 16px;
    border-left: 3px solid #e2e8f0;
    margin-bottom: 8px;
    background: #f8fafc;
    border-radius: 0 8px 8px 0;
}
.claim-item.flagged {
    border-left-color: #ef4444;
    background: #fef2f2;
}
.claim-item.sourced {
    border-left-color: #10b981;
    background: #f0fdf4;
}

/* Stakeholder Grid */
.stakeholder-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}
.stakeholder-card {
    padding: 12px;
    border-radius: 8px;
    background: #f8fafc;
    text-align: center;
}
.stakeholder-card.ok { background: #f0fdf4; border: 1px solid #10b981; }
.stakeholder-card.concern { background: #fffbeb; border: 1px solid #f59e0b; }
.stakeholder-card.flag { background: #fef2f2; border: 1px solid #ef4444; }

/* Evidence Chain */
.evidence-chain {
    position: relative;
    padding-left: 24px;
}
.evidence-chain::before {
    content: '';
    position: absolute;
    left: 8px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: linear-gradient(to bottom, #3b82f6, #8b5cf6);
}
.evidence-item {
    position: relative;
    padding: 8px 0 8px 16px;
    margin-bottom: 8px;
}
.evidence-item::before {
    content: '';
    position: absolute;
    left: -20px;
    top: 14px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #3b82f6;
    border: 2px solid white;
    box-shadow: 0 0 0 2px #3b82f6;
}

/* Recommendation Box */
.recommendation-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px;
}
.recommendation-box.high-priority {
    background: linear-gradient(135deg, #fef2f2 0%, #f8fafc 100%);
    border-color: #ef4444;
}
.recommendation-box.medium-priority {
    background: linear-gradient(135deg, #fffbeb 0%, #f8fafc 100%);
    border-color: #f59e0b;
}

/* Summary Stats */
.stat-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 500;
    margin-right: 8px;
    margin-bottom: 8px;
}

/* Competitor Comparison */
.competitor-row {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 4px;
}
.competitor-row.passed { background: #f0fdf4; }
.competitor-row.failed { background: #fef2f2; }

/* Trust Score Gauge */
.trust-gauge {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}
.trust-gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
</style>
"""


def generate_executive_summary(metadata: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """Generate executive summary section"""

    # Count root causes
    root_cause_counts = {}
    for r in results:
        if r.get("root_cause"):
            cat = r["root_cause"].get("primary_category", "unknown")
            root_cause_counts[cat] = root_cause_counts.get(cat, 0) + 1

    # Count priorities
    priority_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        if r.get("recommendations"):
            priority = r["recommendations"].get("priority", "MEDIUM")
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

    # Build root cause pills
    root_cause_pills = ""
    for cat, count in sorted(root_cause_counts.items(), key=lambda x: -x[1]):
        color = ROOT_CAUSE_COLORS.get(cat, BRAND_COLORS["text_secondary"])
        icon = ROOT_CAUSE_ICONS.get(cat, "fa-question")
        root_cause_pills += f"""
        <span class="stat-pill" style="background: {color}20; color: {color};">
            <i class="fas {icon} me-2"></i>{cat.replace('_', ' ').title()}: {count}
        </span>"""

    # Build priority pills
    priority_pills = ""
    for priority, count in priority_counts.items():
        if count > 0:
            color = PRIORITY_COLORS.get(priority, BRAND_COLORS["info"])
            priority_pills += f"""
            <span class="stat-pill" style="background: {color}20; color: {color};">
                {priority}: {count}
            </span>"""

    return f"""
<div class="card mb-4">
    <div class="card-header">
        <h5 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Executive Summary</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-md-4">
                <div class="stat-box text-center p-3">
                    <div class="stat-value" style="font-size: 3rem; color: {BRAND_COLORS['danger']};">
                        {len(results)}
                    </div>
                    <div class="stat-label">Failed Queries Analyzed</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-box text-center p-3">
                    <div class="stat-value" style="font-size: 3rem; color: {BRAND_COLORS['quality']};">
                        {len(root_cause_counts)}
                    </div>
                    <div class="stat-label">Root Cause Categories</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="stat-box text-center p-3">
                    <div class="stat-value" style="font-size: 3rem; color: {BRAND_COLORS['warning']};">
                        {len(results) * 4}
                    </div>
                    <div class="stat-label">Analysis Queries Used</div>
                </div>
            </div>
        </div>

        <hr class="my-4">

        <h6 class="mb-3"><i class="fas fa-tags me-2"></i>Root Cause Breakdown</h6>
        <div class="mb-4">
            {root_cause_pills}
        </div>

        <h6 class="mb-3"><i class="fas fa-flag me-2"></i>Priority Distribution</h6>
        <div>
            {priority_pills}
        </div>
    </div>
</div>
"""


def generate_failure_card(result: Dict[str, Any], index: int) -> str:
    """Generate detailed card for a single failure"""

    question_id = result.get("question_id", "Unknown")
    question = result.get("question", "")
    target = result.get("target_answer", "")
    actual = result.get("customgpt_answer", "")
    judge_reasoning = result.get("judge_reasoning", "")

    root_cause = result.get("root_cause", {})
    primary_cat = root_cause.get("primary_category", "unknown")
    evidence = root_cause.get("evidence_chain", [])
    secondary = root_cause.get("secondary_categories", [])

    explainability = result.get("explainability", {})
    claims = explainability.get("claims", [])
    trust_score = explainability.get("trust_score", {})
    stakeholder = explainability.get("stakeholder_analysis", {})

    recommendations = result.get("recommendations", {})
    priority = recommendations.get("priority", "MEDIUM")

    competitor_context = result.get("competitor_context", {})

    cat_color = ROOT_CAUSE_COLORS.get(primary_cat, BRAND_COLORS["text_secondary"])
    cat_icon = ROOT_CAUSE_ICONS.get(primary_cat, "fa-question")
    priority_color = PRIORITY_COLORS.get(priority, BRAND_COLORS["info"])

    # Build claims HTML
    claims_html = ""
    if claims:
        for claim in claims:
            flagged = claim.get("flagged", False)
            sourced = claim.get("source") is not None
            cls = "flagged" if flagged else ("sourced" if sourced else "")
            icon = "fa-times-circle text-danger" if flagged else ("fa-check-circle text-success" if sourced else "fa-circle text-muted")
            claims_html += f"""
            <li class="claim-item {cls}">
                <i class="fas {icon} me-2"></i>
                <span>{claim.get('text', '')}</span>
            </li>"""
    else:
        claims_html = '<li class="claim-item"><em>No claims data available - explainability analysis may not have run for this message</em></li>'

    # Build evidence HTML
    evidence_html = ""
    for ev in evidence:
        evidence_html += f'<div class="evidence-item">{ev}</div>'
    if not evidence_html:
        evidence_html = '<div class="evidence-item"><em>No evidence collected</em></div>'

    # Build competitor HTML
    competitor_html = ""
    for prov, data in competitor_context.items():
        status = data.get("status", "UNKNOWN")
        grade = data.get("grade", "?")
        cls = "passed" if status == "PASSED" else "failed"
        icon = "fa-check text-success" if status == "PASSED" else "fa-times text-danger"
        competitor_html += f"""
        <div class="competitor-row {cls}">
            <i class="fas {icon} me-2"></i>
            <strong>{prov.replace('_', ' ')}</strong>
            <span class="ms-auto badge bg-secondary">{grade}</span>
        </div>"""

    # Trust score gauge
    trust_overall = trust_score.get("overall", 0) if trust_score else 0
    trust_pct = trust_overall * 100
    trust_color = BRAND_COLORS["success"] if trust_pct >= 70 else (BRAND_COLORS["warning"] if trust_pct >= 40 else BRAND_COLORS["danger"])

    return f"""
<div class="card mb-4 root-cause-card" style="--card-border-color: {cat_color};">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">
            <i class="fas {cat_icon} me-2" style="color: {cat_color};"></i>
            {question_id}
        </h5>
        <div>
            <span class="badge" style="background: {cat_color};">{primary_cat.replace('_', ' ').title()}</span>
            <span class="badge" style="background: {priority_color};">{priority}</span>
        </div>
    </div>
    <div class="card-body">
        <!-- Question & Answers -->
        <div class="row mb-4">
            <div class="col-12">
                <h6><i class="fas fa-question-circle me-2"></i>Question</h6>
                <p class="lead">{question}</p>
            </div>
        </div>
        <div class="row mb-4">
            <div class="col-md-6">
                <h6><i class="fas fa-bullseye me-2 text-success"></i>Expected Answer</h6>
                <div class="alert alert-success">{target}</div>
            </div>
            <div class="col-md-6">
                <h6><i class="fas fa-robot me-2 text-danger"></i>CustomGPT Answer</h6>
                <div class="alert alert-danger">{actual}</div>
            </div>
        </div>

        <!-- Judge Reasoning -->
        <div class="mb-4">
            <h6><i class="fas fa-gavel me-2"></i>Judge Reasoning</h6>
            <blockquote class="blockquote" style="border-left: 3px solid {BRAND_COLORS['primary']}; padding-left: 16px; font-size: 0.95rem; color: {BRAND_COLORS['text_secondary']};">
                {judge_reasoning}
            </blockquote>
        </div>

        <hr>

        <!-- Explainability Analysis -->
        <div class="row mb-4">
            <div class="col-md-6">
                <h6><i class="fas fa-list-check me-2"></i>Claims Analysis</h6>
                <ul class="claims-list">
                    {claims_html}
                </ul>
            </div>
            <div class="col-md-6">
                <h6><i class="fas fa-shield-halved me-2"></i>Trust Score</h6>
                <div class="mb-3">
                    <div class="d-flex justify-content-between mb-1">
                        <span>Overall Trust</span>
                        <span style="color: {trust_color};">{trust_pct:.0f}%</span>
                    </div>
                    <div class="trust-gauge">
                        <div class="trust-gauge-fill" style="width: {trust_pct}%; background: {trust_color};"></div>
                    </div>
                </div>
                <div class="small text-muted">
                    Sourced claims: {trust_score.get('sourced_claims_ratio', 0) * 100:.0f}% |
                    Flagged: {trust_score.get('flagged_claims_count', 0)}
                </div>
            </div>
        </div>

        <!-- Root Cause Evidence -->
        <div class="row mb-4">
            <div class="col-md-6">
                <h6><i class="fas fa-search me-2"></i>Root Cause Evidence</h6>
                <div class="evidence-chain">
                    {evidence_html}
                </div>
            </div>
            <div class="col-md-6">
                <h6><i class="fas fa-users me-2"></i>Competitor Results</h6>
                {competitor_html if competitor_html else '<p class="text-muted">No competitor data</p>'}
            </div>
        </div>

        <!-- Recommendations -->
        <div class="recommendation-box {priority.lower()}-priority">
            <h6><i class="fas fa-lightbulb me-2"></i>Recommendations</h6>
            <div class="row">
                <div class="col-md-4">
                    <strong>KB Remediation:</strong>
                    <p class="mb-0 small">{recommendations.get('kb_remediation', 'N/A')}</p>
                </div>
                <div class="col-md-4">
                    <strong>Retrieval Improvement:</strong>
                    <p class="mb-0 small">{recommendations.get('retrieval_improvement', 'N/A')}</p>
                </div>
                <div class="col-md-4">
                    <strong>Response Quality:</strong>
                    <p class="mb-0 small">{recommendations.get('response_quality', 'N/A')}</p>
                </div>
            </div>
            <div class="mt-2">
                <strong>Expected Impact:</strong> {recommendations.get('expected_impact', 'N/A')}
            </div>
        </div>
    </div>
</div>
"""


def generate_report(input_file: Path, output_file: Path = None) -> Path:
    """Generate HTML report from post-mortem JSON results"""

    # Load results
    with open(input_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", [])

    run_id = metadata.get("run_id", "Unknown")
    timestamp = metadata.get("analysis_timestamp", datetime.now().isoformat())

    # Generate report sections
    html_head = get_html_head(
        title=f"Explainability Post-Mortem: {run_id}",
        description="CustomGPT failure analysis with explainability insights"
    )

    custom_css = get_custom_css()

    page_header = get_page_header(
        title="Explainability Post-Mortem",
        subtitle="Root cause analysis of CustomGPT failures using Verify Responses API",
        meta_info=f"Run: {run_id} | Generated: {format_timestamp(datetime.fromisoformat(timestamp))}"
    )

    executive_summary = generate_executive_summary(metadata, results)

    # Generate all failure cards
    failure_cards = ""
    for i, result in enumerate(results):
        failure_cards += generate_failure_card(result, i)

    # Assemble full report
    html = f"""{html_head}
{custom_css}
<body>
    <div class="container-fluid px-4 py-4">
        {page_header}
        {executive_summary}

        <h4 class="mb-4"><i class="fas fa-microscope me-2"></i>Detailed Failure Analysis</h4>
        {failure_cards}

        <div class="text-center text-muted py-4">
            <small>Generated by CustomGPT Explainability Post-Mortem System</small>
        </div>
    </div>
"""

    html = wrap_html_document(html)

    # Determine output path
    if output_file is None:
        output_file = input_file.parent / f"explainability_report_{run_id}.html"

    # Write report
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Report generated: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report from explainability post-mortem results"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to postmortem_results_*.json file"
    )
    parser.add_argument(
        "--output",
        help="Output HTML file path (default: auto-generated)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    output_file = Path(args.output) if args.output else None
    generate_report(input_file, output_file)


if __name__ == "__main__":
    main()
