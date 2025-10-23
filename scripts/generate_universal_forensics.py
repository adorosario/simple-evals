#!/usr/bin/env python3
"""
Universal Forensic Report Generator
Creates comprehensive browser-viewable forensic reports for ANY provider's penalty cases.

Academic-grade forensic analysis suitable for peer review:
- Works with CustomGPT, OpenAI RAG, OpenAI Vanilla
- Provider-specific debugging sections
- Comparative analysis across all providers
- Complete data for reproducibility

This script generates:
1. Main forensic dashboard with links to all reports
2. Individual HTML reports for each failed question
3. Individual JSON reports for programmatic analysis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import markdown
import os
import time
import requests

# Provider display names
PROVIDER_DISPLAY_NAMES = {
    'customgpt': 'CustomGPT',
    'openai_rag': 'OpenAI RAG',
    'openai_vanilla': 'OpenAI Vanilla'
}

# Provider keys for audit log lookup
PROVIDER_KEYS = {
    'customgpt': 'CustomGPT_RAG',
    'openai_rag': 'OpenAI_RAG',
    'openai_vanilla': 'OpenAI_Vanilla'
}

# Citation content cache to avoid redundant API calls
_CITATION_CACHE = {}


def fetch_citation_content(
    project_id: str,
    citation_id: int,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch full citation content from CustomGPT Citations API

    Args:
        project_id: CustomGPT project ID
        citation_id: Numeric citation ID
        api_key: CustomGPT API key (reads from env if not provided)

    Returns:
        Citation object with content, source, metadata, or None if fetch fails

    API Reference:
        https://docs.customgpt.ai/reference/get_api-v1-projects-projectid-citations-citationid
    """
    # Check cache first
    cache_key = f"{project_id}_{citation_id}"
    if cache_key in _CITATION_CACHE:
        return _CITATION_CACHE[cache_key]

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('CUSTOMGPT_API_KEY')

    if not api_key:
        print(f"  ⚠️  Warning: CUSTOMGPT_API_KEY not set, cannot fetch citation {citation_id}")
        return None

    url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/citations/{citation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    try:
        # Rate limiting: be nice to the API
        time.sleep(0.1)

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            data_fields = data.get("data", {})

            # Extract relevant fields from API response
            # Map API field names to our schema
            citation_obj = {
                "citation_id": citation_id,
                "source_url": data_fields.get("url") or data_fields.get("source_url"),
                "page_name": data_fields.get("title") or data_fields.get("page_name"),
                "page_content": data_fields.get("content") or data_fields.get("page_content"),
                "page_description": data_fields.get("description") or data_fields.get("page_description"),
                "image_url": data_fields.get("image"),
                "sitemap_url": data_fields.get("sitemap_url"),
                "created_at": data_fields.get("created_at"),
                "updated_at": data_fields.get("updated_at"),
                "raw_response": data  # Include full response for debugging
            }

            # Cache the result
            _CITATION_CACHE[cache_key] = citation_obj
            return citation_obj

        elif response.status_code == 404:
            print(f"  ⚠️  Citation {citation_id} not found (404)")
            return None
        elif response.status_code == 401:
            print(f"  ❌ Authentication failed (401) - check CUSTOMGPT_API_KEY")
            return None
        else:
            print(f"  ⚠️  Failed to fetch citation {citation_id}: HTTP {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Network error fetching citation {citation_id}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Unexpected error fetching citation {citation_id}: {e}")
        return None


def get_confidence_display_properties(confidence: float) -> dict:
    """
    Get Bootstrap styling properties for judge confidence level visualization

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
            "interpretation": "⭐ Very High Confidence - Judge is very certain",
            "icon": "fa-check-circle",
            "level": "VERY_HIGH"
        }
    elif confidence >= 0.75:
        return {
            "badge_color": "info",
            "progress_class": "bg-info",
            "interpretation": "✓ High Confidence - Judge is confident",
            "icon": "fa-info-circle",
            "level": "HIGH"
        }
    elif confidence >= 0.60:
        return {
            "badge_color": "warning",
            "progress_class": "bg-warning text-dark",
            "interpretation": "⚠ Medium Confidence - Judge has some uncertainty",
            "icon": "fa-exclamation-triangle",
            "level": "MEDIUM"
        }
    else:
        return {
            "badge_color": "danger",
            "progress_class": "bg-danger",
            "interpretation": "❗ Low Confidence - Judge is uncertain, needs human review",
            "icon": "fa-exclamation-circle",
            "level": "LOW"
        }


def extract_provider_metadata(audit_log_path: Path, question_id: str, provider_key: str) -> Optional[Dict[str, Any]]:
    """
    Extract provider-specific metadata from audit logs

    Returns provider-specific debugging information:
    - CustomGPT: Citations, API IDs, conversation endpoints
    - OpenAI RAG: Vector store ID, model info
    - OpenAI Vanilla: Model info, API details
    """
    if not audit_log_path.exists():
        return None

    try:
        with open(audit_log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())

                    if (log_entry.get('provider') == provider_key and
                        log_entry.get('question_id') == question_id):

                        metadata = log_entry.get('metadata', {})

                        # Build provider-specific metadata structure
                        result = {
                            'provider': provider_key,
                            'raw_metadata': metadata
                        }

                        # CustomGPT-specific fields
                        if provider_key == 'CustomGPT_RAG':
                            project_id = metadata.get('project_id')
                            citation_ids = metadata.get('citations', [])

                            # Enrich citations with full content from API
                            enriched_citations = []
                            if project_id and citation_ids:
                                print(f"  Fetching {len(citation_ids)} citation(s) for {question_id}...")
                                for cit_id in citation_ids:
                                    citation_content = fetch_citation_content(project_id, cit_id)
                                    if citation_content:
                                        enriched_citations.append(citation_content)
                                    else:
                                        # Fallback to ID-only if fetch fails
                                        enriched_citations.append({
                                            "citation_id": cit_id,
                                            "error": "Failed to fetch citation content"
                                        })

                            result.update({
                                'project_id': project_id,
                                'session_id': metadata.get('session_id'),
                                'conversation_id': metadata.get('conversation_id'),
                                'prompt_id': metadata.get('prompt_id'),
                                'message_id': metadata.get('message_id'),
                                'external_id': metadata.get('external_id'),
                                'citations': enriched_citations if enriched_citations else citation_ids,
                                'citation_ids': citation_ids,  # Keep original IDs
                                'citation_count': metadata.get('citation_count', 0),
                                'debug_urls': metadata.get('debug_urls', {})
                            })

                        # OpenAI RAG-specific fields
                        elif provider_key == 'OpenAI_RAG':
                            result.update({
                                'vector_store_id': metadata.get('vector_store_id'),
                                'provider_type': metadata.get('provider_type'),
                                'api_endpoint': metadata.get('api_endpoint')
                            })

                        # OpenAI Vanilla-specific fields
                        elif provider_key == 'OpenAI_Vanilla':
                            result.update({
                                'provider_type': metadata.get('provider_type'),
                                'api_endpoint': metadata.get('api_endpoint')
                            })

                        return result

                except json.JSONDecodeError:
                    continue

        return None
    except Exception as e:
        print(f"Warning: Failed to extract provider metadata: {e}")
        return None


def load_penalty_data(run_dir: Path, provider: str) -> Dict[str, Any]:
    """
    Load penalty analysis data for the specified provider

    Returns:
        dict with penalty_cases, metadata, analysis
    """
    run_id = run_dir.name
    penalty_analysis_dir = run_dir / f"{provider}_penalty_analysis"

    penalty_file = penalty_analysis_dir / f"{provider}_penalty_analysis_{run_id}.json"

    if not penalty_file.exists():
        raise FileNotFoundError(
            f"Penalty analysis file not found: {penalty_file}\n"
            f"Run universal_penalty_analyzer.py first with --provider {provider}"
        )

    with open(penalty_file, 'r') as f:
        data = json.load(f)

    return data


def generate_provider_debugging_section(provider: str, metadata: Optional[Dict[str, Any]]) -> str:
    """
    Generate provider-specific debugging HTML section

    Args:
        provider: Provider name (customgpt, openai_rag, openai_vanilla)
        metadata: Provider-specific metadata from audit logs

    Returns:
        HTML string with provider-specific debugging information
    """
    if not metadata:
        return ""

    provider_display = PROVIDER_DISPLAY_NAMES[provider]
    provider_key = PROVIDER_KEYS[provider]

    html = f"""
        <!-- {provider_display} Server-Side Debugging Info -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-primary">
                    <div class="card-header bg-primary text-white">
                        <h3><i class="fas fa-server"></i> {provider_display} Debugging Information</h3>
                        <small>Provider-specific metadata for server-side investigation</small>
                    </div>
                    <div class="card-body">
    """

    # CustomGPT-specific debugging
    if provider == 'customgpt':
        html += f"""
                        <!-- CustomGPT API IDs -->
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <h5><i class="fas fa-fingerprint"></i> CustomGPT API IDs</h5>
                                <table class="table table-sm table-bordered">
                                    <tbody>
                                        <tr>
                                            <td><strong>Project ID:</strong></td>
                                            <td><code>{metadata.get('project_id', 'N/A')}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Session ID:</strong></td>
                                            <td><code>{metadata.get('session_id', 'N/A')}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Prompt ID:</strong></td>
                                            <td><code>{metadata.get('prompt_id', 'N/A')}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>External ID:</strong></td>
                                            <td><code>{metadata.get('external_id', 'N/A')}</code></td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <h5><i class="fas fa-link"></i> Debug API Endpoints</h5>
                                <p class="text-muted small">Use these URLs to query the CustomGPT API directly</p>
        """

        debug_urls = metadata.get('debug_urls', {})
        if debug_urls.get('message_endpoint'):
            html += f"""
                                <div class="mb-2">
                                    <strong>Message Endpoint:</strong><br>
                                    <a href="{debug_urls['message_endpoint']}" target="_blank" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-external-link-alt"></i> View Message
                                    </a>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="navigator.clipboard.writeText('{debug_urls['message_endpoint']}')">
                                        <i class="fas fa-copy"></i> Copy URL
                                    </button>
                                </div>
            """

        html += """
                            </div>
                        </div>
        """

        # Citations
        citations = metadata.get('citations', [])
        citation_count = metadata.get('citation_count', 0)

        html += """
                        <!-- Knowledge Base Citations -->
                        <div class="row">
                            <div class="col-12">
                                <h5><i class="fas fa-quote-right"></i> Knowledge Base Citations</h5>
        """

        if citation_count > 0:
            html += f"""
                                <div class="alert alert-info">
                                    <i class="fas fa-info-circle"></i> CustomGPT returned <strong>{citation_count} citation(s)</strong>
                                </div>
            """

            # Check if citations are enriched (dicts) or just IDs (ints)
            if citations and isinstance(citations[0], dict):
                # Enriched citations with full content
                html += """
                                <div class="accordion" id="citationsAccordion">
                """
                for idx, citation in enumerate(citations, 1):
                    citation_id = citation.get('citation_id', 'unknown')
                    source_url = citation.get('source_url') or 'N/A'
                    page_name = citation.get('page_name') or 'Untitled'
                    page_description = citation.get('page_description') or ''
                    page_content = citation.get('page_content') or ''
                    error = citation.get('error')

                    # Truncate name for accordion header
                    page_name_display = page_name[:50] + "..." if len(page_name) > 50 else page_name

                    html += f"""
                                    <div class="accordion-item">
                                        <h2 class="accordion-header" id="heading{idx}">
                                            <button class="accordion-button {'collapsed' if idx > 1 else ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{idx}" aria-expanded="{'true' if idx == 1 else 'false'}" aria-controls="collapse{idx}">
                                                <strong>Citation {idx}:</strong> &nbsp; <code>{citation_id}</code> &nbsp; - &nbsp; {page_name_display}
                                            </button>
                                        </h2>
                                        <div id="collapse{idx}" class="accordion-collapse collapse {'show' if idx == 1 else ''}" aria-labelledby="heading{idx}" data-bs-parent="#citationsAccordion">
                                            <div class="accordion-body">
                    """

                    if error:
                        html += f"""
                                                <div class="alert alert-warning">
                                                    <i class="fas fa-exclamation-triangle"></i> {error}
                                                </div>
                        """
                    else:
                        html += f"""
                                                <table class="table table-sm table-borderless mb-3">
                                                    <tbody>
                                                        <tr>
                                                            <td><strong>Citation ID:</strong></td>
                                                            <td><code>{citation_id}</code></td>
                                                        </tr>
                                                        <tr>
                                                            <td><strong>Source URL:</strong></td>
                                                            <td><a href="{source_url}" target="_blank" rel="noopener">{source_url}</a></td>
                                                        </tr>
                                                        <tr>
                                                            <td><strong>Page Name:</strong></td>
                                                            <td>{page_name}</td>
                                                        </tr>
                        """
                        if page_description:
                            html += f"""
                                                        <tr>
                                                            <td><strong>Description:</strong></td>
                                                            <td>{page_description}</td>
                                                        </tr>
                            """
                        html += """
                                                    </tbody>
                                                </table>
                        """

                        if page_content:
                            html += f"""
                                                <div class="mt-3">
                                                    <strong><i class="fas fa-file-alt"></i> Retrieved Content:</strong>
                                                    <div class="card mt-2">
                                                        <div class="card-body">
                                                            <pre class="mb-0" style="white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto;"><code>{page_content}</code></pre>
                                                        </div>
                                                    </div>
                                                </div>
                            """

                    html += """
                                            </div>
                                        </div>
                                    </div>
                    """

                html += """
                                </div>
                """
            else:
                # Fallback to ID-only display (original behavior)
                html += """
                                <table class="table table-sm table-striped">
                                    <thead class="table-dark">
                                        <tr>
                                            <th>#</th>
                                            <th>Citation ID</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                """
                for idx, citation in enumerate(citations, 1):
                    html += f"""
                                        <tr>
                                            <td>{idx}</td>
                                            <td><pre class="mb-0"><code>{citation}</code></pre></td>
                                        </tr>
                    """
                html += """
                                    </tbody>
                                </table>
                """
        else:
            html += """
                                <div class="alert alert-warning">
                                    <i class="fas fa-exclamation-triangle"></i> <strong>No citations found!</strong>
                                    <br><small>This may indicate the answer was generated without retrieving documents.</small>
                                </div>
            """

        html += """
                            </div>
                        </div>
        """

    # OpenAI RAG-specific debugging
    elif provider == 'openai_rag':
        vector_store_id = metadata.get('vector_store_id', 'N/A')
        api_endpoint = metadata.get('api_endpoint', 'N/A')

        html += f"""
                        <div class="row">
                            <div class="col-12">
                                <h5><i class="fas fa-database"></i> OpenAI RAG Configuration</h5>
                                <table class="table table-sm table-bordered">
                                    <tbody>
                                        <tr>
                                            <td><strong>Vector Store ID:</strong></td>
                                            <td><code>{vector_store_id}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>API Endpoint:</strong></td>
                                            <td><code>{api_endpoint}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Provider Type:</strong></td>
                                            <td><code>{metadata.get('provider_type', 'N/A')}</code></td>
                                        </tr>
                                    </tbody>
                                </table>
                                <div class="alert alert-info mt-3">
                                    <i class="fas fa-info-circle"></i> <strong>Note:</strong> OpenAI RAG uses file_search tool with the specified vector store.
                                    Citation extraction not implemented in current version.
                                </div>
                            </div>
                        </div>
        """

    # OpenAI Vanilla-specific debugging
    elif provider == 'openai_vanilla':
        api_endpoint = metadata.get('api_endpoint', 'N/A')

        html += f"""
                        <div class="row">
                            <div class="col-12">
                                <h5><i class="fas fa-cog"></i> OpenAI Vanilla Configuration</h5>
                                <table class="table table-sm table-bordered">
                                    <tbody>
                                        <tr>
                                            <td><strong>API Endpoint:</strong></td>
                                            <td><code>{api_endpoint}</code></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Provider Type:</strong></td>
                                            <td><code>{metadata.get('provider_type', 'N/A')}</code></td>
                                        </tr>
                                    </tbody>
                                </table>
                                <div class="alert alert-info mt-3">
                                    <i class="fas fa-info-circle"></i> <strong>Note:</strong> OpenAI Vanilla is a standard LLM without RAG capabilities.
                                    Performance relies solely on pre-training knowledge.
                                </div>
                            </div>
                        </div>
        """

    html += """
                    </div>
                </div>
            </div>
        </div>
    """

    return html


def generate_individual_question_html(
    question_data: Dict[str, Any],
    provider: str,
    output_file: Path,
    audit_log_path: Optional[Path] = None
) -> str:
    """
    Generate individual HTML forensic report for a single question

    Args:
        question_data: Penalty case data
        provider: Provider name (customgpt, openai_rag, openai_vanilla)
        output_file: Where to save the HTML
        audit_log_path: Path to audit log for metadata extraction

    Returns:
        Path to generated HTML file
    """
    question_id = question_data['question_id']
    provider_display = PROVIDER_DISPLAY_NAMES[provider]
    provider_key = PROVIDER_KEYS[provider]

    # Extract judge confidence
    judge_confidence = question_data.get('judge_confidence', 0.0)
    conf_props = get_confidence_display_properties(judge_confidence)

    # Extract provider metadata from audit logs
    provider_meta = None
    if audit_log_path:
        provider_meta = extract_provider_metadata(audit_log_path, question_id, provider_key)

    # Get provider-specific fields with fallback
    provider_answer = question_data.get(f'{provider}_answer', 'N/A')
    provider_grade = question_data.get(f'{provider}_grade', 'N/A')
    provider_confidence = question_data.get(f'{provider}_confidence', 0.0)

    content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Analysis - {question_id} - {provider_display}</title>

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
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-size: 0.9em; }}
        .forensic-nav {{ background-color: #212529; }}
        .forensic-nav .nav-link {{ color: #adb5bd; }}
        .forensic-nav .nav-link:hover {{ color: #fff; }}
        .forensic-nav .nav-link.active {{ color: #fff; background-color: #495057; }}
    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg forensic-nav">
        <div class="container-fluid">
            <a class="navbar-brand text-white" href="forensic_dashboard.html">
                <i class="fas fa-arrow-left"></i> Back to Dashboard
            </a>
            <span class="navbar-text text-light">Question: {question_id} - Provider: {provider_display}</span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">
                    <i class="fas fa-microscope text-danger"></i>
                    Forensic Analysis: {question_id}
                </h1>
                <p class="lead">{provider_display} failure investigation with competitive analysis</p>
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
                                <p><strong>Question ID:</strong> <code>{question_id}</code></p>
                                <p><strong>Domain:</strong> <span class="badge bg-secondary">{question_data.get('domain', 'Unknown')}</span></p>
                                <p><strong>Complexity:</strong> {question_data.get('complexity', 0.0):.3f}</p>
                                <p><strong>Penalty Points:</strong> <span class="text-danger fw-bold">{question_data.get('penalty_points', 0.0)}</span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>{provider_display} Confidence:</strong>
                                    <span class="badge bg-warning">{provider_confidence:.3f}</span>
                                </p>
                                <p><strong>{provider_display} Grade:</strong> <span class="badge bg-danger grade-badge">{provider_grade}</span></p>
                                <p><strong>Penalty Type:</strong> {question_data.get('penalty_type', 'Unknown').replace('_', ' ').title()}</p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <h5>Question:</h5>
                            <div class="alert alert-info">
                                {question_data.get('question', 'N/A')}
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
                                        <td><em>{question_data.get('target_answer', 'N/A')}</em></td>
                                        <td><span class="badge bg-success">CORRECT</span></td>
                                        <td><i class="fas fa-check text-success"></i> Gold Standard</td>
                                    </tr>
                                    <tr class="table-danger">
                                        <td><strong>{provider_display}</strong></td>
                                        <td>{provider_answer}</td>
                                        <td><span class="badge bg-danger">{provider_grade}</span></td>
                                        <td><i class="fas fa-times text-danger"></i> FAILED (-{question_data.get('penalty_points', 0.0)} points)</td>
                                    </tr>
    """

    # Add competitor results
    competitor_results = question_data.get('competitor_results', {})
    for comp_key, comp_data in competitor_results.items():
        comp_display = PROVIDER_DISPLAY_NAMES.get(comp_key.lower().replace('_rag', '_rag').replace('_vanilla', '_vanilla'), comp_key)
        if 'CustomGPT' in comp_key:
            comp_name = 'customgpt'
        elif 'OpenAI_RAG' in comp_key:
            comp_name = 'openai_rag'
        else:
            comp_name = 'openai_vanilla'
        comp_display = PROVIDER_DISPLAY_NAMES.get(comp_name, comp_key)

        status_class = "table-success" if comp_data['status'] == 'PASSED' else "table-danger"
        status_icon = "check" if comp_data['status'] == 'PASSED' else "times"
        status_color = "success" if comp_data['status'] == 'PASSED' else "danger"
        grade_color = "success" if comp_data['grade'] == 'A' else "danger"

        content += f"""
                                    <tr class="{status_class}">
                                        <td><strong>{comp_display}</strong></td>
                                        <td>{comp_data['answer']}</td>
                                        <td><span class="badge bg-{grade_color}">{comp_data['grade']}</span></td>
                                        <td><i class="fas fa-{status_icon} text-{status_color}"></i> {comp_data['status']}</td>
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
    """

    # Add provider-specific debugging section
    content += generate_provider_debugging_section(provider, provider_meta)

    # Judge Evaluation
    content += f"""
        <!-- Judge Evaluation with Confidence Visualization -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-light">
                        <h3><i class="fas fa-gavel"></i> LLM-As-A-Judge Evaluation</h3>
                        <small class="text-muted">Automated evaluation by GPT-5</small>
                    </div>
                    <div class="card-body">
                        <!-- Confidence Score -->
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
                                <p class="mb-0" style="white-space: pre-wrap;">{question_data.get('judge_reasoning', 'N/A')}</p>
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
        <p>&copy; 2025 {provider_display} Forensic Analysis - Question {question_id} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>

    <!-- No DataTables on individual pages - tables have varying structures -->
</body>
</html>
    """

    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)

    return str(output_file)


def generate_individual_question_json(
    question_data: Dict[str, Any],
    provider: str,
    output_file: Path,
    audit_log_path: Optional[Path] = None
) -> str:
    """
    Generate individual JSON forensic report for programmatic analysis
    """
    question_id = question_data['question_id']
    provider_key = PROVIDER_KEYS[provider]

    # Extract provider metadata
    provider_meta = None
    if audit_log_path:
        provider_meta = extract_provider_metadata(audit_log_path, question_id, provider_key)

    # Extract judge confidence
    judge_confidence = question_data.get('judge_confidence', 0.0)
    conf_props = get_confidence_display_properties(judge_confidence)

    # Build JSON structure
    json_data = {
        "question_id": question_id,
        "question": question_data.get('question', 'N/A'),
        "target_answer": question_data.get('target_answer', 'N/A'),

        # Provider metadata
        "provider_metadata": provider_meta if provider_meta else {},

        # Provider response
        "provider": {
            "name": PROVIDER_DISPLAY_NAMES[provider],
            "answer": question_data.get(f'{provider}_answer', 'N/A'),
            "grade": question_data.get(f'{provider}_grade', 'N/A'),
            "confidence": question_data.get(f'{provider}_confidence', 0.0),
            "status": "FAILED"
        },

        # Competitors
        "competitors": question_data.get('competitor_results', {}),

        # Judge evaluation
        "judge": {
            "reasoning": question_data.get('judge_reasoning', 'N/A'),
            "confidence": judge_confidence,
            "confidence_level": conf_props['level'],
            "model": "gpt-5"
        },

        # Metadata
        "metadata": {
            "domain": question_data.get('domain', 'unknown'),
            "complexity": question_data.get('complexity', 0.0),
            "penalty_points": question_data.get('penalty_points', 0.0),
            "penalty_type": question_data.get('penalty_type', 'unknown')
        },

        "schema_version": "2.0",  # v2.0: Added enriched citation content from CustomGPT API
        "generated_at": datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    return str(output_file)


def generate_dashboard(
    penalty_data: Dict[str, Any],
    provider: str,
    output_file: Path
) -> str:
    """Generate forensic dashboard HTML"""

    provider_display = PROVIDER_DISPLAY_NAMES[provider]
    penalty_cases = penalty_data.get('penalty_cases', [])
    analysis = penalty_data.get('analysis', {})

    total_failures = analysis.get('total_penalties', 0)
    total_penalty_points = analysis.get('total_penalty_points', 0.0)

    content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{provider_display} Forensic Dashboard</title>

    <!-- External Dependencies -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <style>
        .question-card {{ border-left: 4px solid #dc3545; }}
        .metric-card {{ border-left: 4px solid #0d6efd; }}
        .forensic-nav {{ background-color: #212529; }}
    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg forensic-nav">
        <div class="container-fluid">
            <a class="navbar-brand text-white" href="#">
                <i class="fas fa-microscope"></i> {provider_display} Forensic Analysis
            </a>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">
                    <i class="fas fa-exclamation-triangle text-danger"></i>
                    {provider_display} Penalty Case Forensic Analysis
                </h1>
                <p class="lead">Complete forensic investigation of all {total_failures} penalty cases</p>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-danger">{total_failures}</h2>
                        <p class="card-text">Total Failures</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-danger">{total_penalty_points:.1f}</h2>
                        <p class="card-text">Penalty Points</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h2 class="text-info">{len(penalty_cases)}</h2>
                        <p class="card-text">Cases Analyzed</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Overview Table -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3><i class="fas fa-table"></i> All Penalty Cases</h3>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped table-hover" id="penaltyTable">
                            <thead class="table-dark">
                                <tr>
                                    <th>Question ID</th>
                                    <th>Question</th>
                                    <th>Grade</th>
                                    <th>Penalty Points</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
    """

    for case in penalty_cases:
        question_id = case['question_id']
        question = case['question'][:80] + "..." if len(case['question']) > 80 else case['question']
        grade = case.get(f'{provider}_grade', 'N/A')
        penalty_points = case.get('penalty_points', 0.0)

        content += f"""
                                <tr>
                                    <td><code>{question_id}</code></td>
                                    <td>{question}</td>
                                    <td><span class="badge bg-danger">{grade}</span></td>
                                    <td><span class="text-danger fw-bold">{penalty_points}</span></td>
                                    <td>
                                        <a href="forensic_question_{question_id}.html" class="btn btn-sm btn-outline-danger">
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
        <p>&copy; 2025 """ + provider_display + """ Forensic Analysis - Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </footer>

    <script>
        $(document).ready(function() {
            $('#penaltyTable').DataTable({
                pageLength: 25,
                order: [[0, 'asc']],
                responsive: true
            });
        });
    </script>
</body>
</html>
    """

    with open(output_file, 'w') as f:
        f.write(content)

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate universal forensic reports for any provider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate forensics for CustomGPT
  python generate_universal_forensics.py --run-dir results/run_XXX --provider customgpt

  # Generate forensics for OpenAI RAG
  python generate_universal_forensics.py --run-dir results/run_XXX --provider openai_rag

  # Generate forensics for OpenAI Vanilla
  python generate_universal_forensics.py --run-dir results/run_XXX --provider openai_vanilla
        """
    )
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument(
        '--provider',
        required=True,
        choices=['customgpt', 'openai_rag', 'openai_vanilla'],
        help='Provider to generate forensics for'
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    provider = args.provider
    provider_display = PROVIDER_DISPLAY_NAMES[provider]

    print(f"\n{'='*60}")
    print(f"Generating Forensic Reports for {provider_display}")
    print(f"{'='*60}\n")

    # Load penalty data
    print("Loading penalty analysis data...")
    penalty_data = load_penalty_data(run_dir, provider)

    penalty_cases = penalty_data.get('penalty_cases', [])

    if not penalty_cases:
        print(f"✓ No penalty cases found for {provider_display}! 🎉")
        return

    print(f"Found {len(penalty_cases)} penalty cases")

    # Create output directory
    forensics_dir = run_dir / f"{provider}_forensics"
    forensics_dir.mkdir(exist_ok=True, parents=True)

    # Audit log path for metadata extraction
    audit_log_path = run_dir / "provider_requests.jsonl"
    if not audit_log_path.exists():
        print(f"⚠️  Warning: Audit log not found at {audit_log_path}")
        print(f"   Provider-specific debugging info will not be available")
        audit_log_path = None

    # Generate individual reports
    print("\nGenerating individual forensic reports...")
    for case in penalty_cases:
        question_id = case['question_id']

        # HTML report
        html_file = forensics_dir / f"forensic_question_{question_id}.html"
        generate_individual_question_html(case, provider, html_file, audit_log_path)

        # JSON report
        json_file = forensics_dir / f"forensic_question_{question_id}.json"
        generate_individual_question_json(case, provider, json_file, audit_log_path)

        print(f"  ✓ {question_id}")

    # Generate dashboard
    print("\nGenerating forensic dashboard...")
    dashboard_file = forensics_dir / "forensic_dashboard.html"
    generate_dashboard(penalty_data, provider, dashboard_file)

    print(f"\n{'='*60}")
    print(f"Forensic Reports Generated Successfully")
    print(f"{'='*60}")
    print(f"Dashboard: {dashboard_file}")
    print(f"Individual Reports: {forensics_dir}/forensic_question_*.html")
    print(f"JSON Reports: {forensics_dir}/forensic_question_*.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
