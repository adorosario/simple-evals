#!/usr/bin/env python3
"""
Enhanced Multi-Provider RAG Benchmark with Confidence Threshold Framework
Implements theoretical insights from OpenAI's "Why Language Models Hallucinate" paper
Compares Volume Strategy vs Quality Strategy across multiple confidence thresholds
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sampler.audited_customgpt_sampler import AuditedCustomGPTSampler
from sampler.audited_openai_vanilla_sampler import AuditedOpenAIVanillaSampler
from sampler.audited_openai_rag_sampler import AuditedOpenAIRAGSampler
from sampler.audited_gemini_rag_sampler import AuditedGeminiRAGSampler
from confidence_threshold_simpleqa_eval import ConfidenceThresholdSimpleQAEval, CONFIDENCE_THRESHOLDS

# Standardized system prompts for fair comparison
STANDARD_RAG_SYSTEM_PROMPT = """You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.
If the answer is not in the knowledge base, say: "I don't know based on the available documentation."
Be concise and factual."""

STANDARD_VANILLA_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on your training knowledge.
If you're not confident in your answer, say: "I don't know."
Be concise and factual."""
from audit_logger import AuditLogger, create_run_id
from leaderboard_generator import LeaderboardGenerator

# Brand kit report generators
from scripts.report_generators import (
    generate_quality_benchmark_report_v2,
    generate_statistical_analysis_report_v2
)
from scripts.generate_main_dashboard import generate_main_dashboard


def setup_samplers(audit_logger=None):
    """Initialize all providers for comparison with audit logging"""
    samplers = {}
    errors = []

    # 1. CustomGPT RAG Sampler (with GPT-5.1)
    print("🔧 Setting up CustomGPT (RAG) sampler...")
    try:
        os.environ["CUSTOMGPT_MODEL_NAME"] = "gpt-5.1"
        sampler = AuditedCustomGPTSampler.from_env(audit_logger=audit_logger)
        samplers["CustomGPT_RAG"] = sampler

        if audit_logger:
            audit_logger.add_provider("CustomGPT_RAG", sampler._get_request_data([]))

        print("   ✅ CustomGPT sampler ready (GPT-5.1)")
    except Exception as e:
        error_msg = f"CustomGPT setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 2. OpenAI Vanilla Sampler (no RAG, using GPT-5.1)
    print("🔧 Setting up OpenAI Vanilla (no RAG) sampler...")
    try:
        sampler = AuditedOpenAIVanillaSampler(
            model="gpt-5.1",
            system_message=STANDARD_VANILLA_SYSTEM_PROMPT,
            temperature=0,
            audit_logger=audit_logger
        )
        samplers["OpenAI_Vanilla"] = sampler

        if audit_logger:
            audit_logger.add_provider("OpenAI_Vanilla", sampler._get_request_data([]))

        print("   ✅ OpenAI Vanilla sampler ready (GPT-5.1)")
    except Exception as e:
        error_msg = f"OpenAI Vanilla setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 3. OpenAI RAG Sampler (with vector store, using GPT-5.1)
    print("🔧 Setting up OpenAI RAG (vector store) sampler...")
    try:
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        if vector_store_id:
            sampler = AuditedOpenAIRAGSampler(
                model="gpt-5.1",
                system_message=STANDARD_RAG_SYSTEM_PROMPT,
                temperature=0,
                audit_logger=audit_logger
            )
            samplers["OpenAI_RAG"] = sampler

            if audit_logger:
                audit_logger.add_provider("OpenAI_RAG", sampler._get_request_data([]))

            print("   ✅ OpenAI RAG sampler ready (GPT-5.1)")
        else:
            error_msg = "OpenAI RAG setup failed: OPENAI_VECTOR_STORE_ID not set"
            print(f"   ❌ {error_msg}")
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"OpenAI RAG setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 4. Google Gemini RAG Sampler (with File Search, using gemini-3-pro-preview)
    print("🔧 Setting up Google Gemini RAG (File Search) sampler...")
    try:
        google_store_name = os.environ.get("GOOGLE_FILE_SEARCH_STORE_NAME")
        if google_store_name:
            sampler = AuditedGeminiRAGSampler(
                store_name=google_store_name,
                model="gemini-3-pro-preview",
                system_message=STANDARD_RAG_SYSTEM_PROMPT,
                temperature=0.0,
                audit_logger=audit_logger
            )
            samplers["Google_Gemini_RAG"] = sampler

            if audit_logger:
                audit_logger.add_provider("Google_Gemini_RAG", sampler._get_request_data([]))

            print("   ✅ Google Gemini RAG sampler ready (gemini-3-pro-preview)")
        else:
            error_msg = "Google Gemini RAG setup skipped: GOOGLE_FILE_SEARCH_STORE_NAME not set"
            print(f"   ⚠️ {error_msg}")
            # Note: Not adding to errors since this is optional
    except Exception as e:
        error_msg = f"Google Gemini RAG setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    return samplers, errors


def run_quality_evaluation(sampler_name, sampler, n_samples=10, audit_logger=None, use_flex_tier=False):
    """Run quality-based evaluation for a single sampler"""
    print(f"\n🎯 Running quality-based evaluation for {sampler_name}...")
    print(f"   Samples: {n_samples}")
    print(f"   Methodology: 80% confidence threshold")

    try:
        # Create quality evaluation
        eval_instance = ConfidenceThresholdSimpleQAEval(
            grader_model=None,  # Uses GPT-5 (with optional flex tier)
            num_examples=n_samples,
            audit_logger=audit_logger,
            use_flex_tier=use_flex_tier
        )

        start_time = time.time()

        # Run single-threshold evaluation
        eval_result = eval_instance(sampler, provider_name=sampler_name)

        end_time = time.time()
        duration = end_time - start_time

        print(f"   ✅ Completed in {duration:.1f}s")

        # Extract metrics
        metrics = eval_result.metrics
        aggregated_result = {
            "sampler_name": sampler_name,
            "duration": duration,
            "samples_evaluated": n_samples,
            "success": True,
            "error": None,
            "metrics": {
                "volume_score": metrics.get("volume_score_mean", 0),
                "quality_score": metrics.get("quality_score_mean", 0),
                "attempted_rate": metrics.get("attempted_rate", 0),
                "accuracy_given_attempted": metrics.get("accuracy_given_attempted", 0),
                "abstention_rate": metrics.get("abstention_rate", 0),
                "overconfidence_penalty": metrics.get("overconfidence_penalty", 0),
                "threshold_value": metrics.get("threshold_value", 0),
                "penalty_ratio": metrics.get("penalty_ratio", 0),
                "n_correct": metrics.get("n_correct", 0),
                "n_incorrect": metrics.get("n_incorrect", 0),
                "n_not_attempted": metrics.get("n_not_attempted", 0),
                "conversations": len(eval_result.convos),
                # Provider performance metrics (latency, tokens, cost)
                "provider_latency_avg_ms": metrics.get("provider_latency_avg_ms"),
                "provider_latency_median_ms": metrics.get("provider_latency_median_ms"),
                "provider_latency_p95_ms": metrics.get("provider_latency_p95_ms"),
                "provider_latency_min_ms": metrics.get("provider_latency_min_ms"),
                "provider_latency_max_ms": metrics.get("provider_latency_max_ms"),
                "total_cost_usd": metrics.get("total_cost_usd"),
                "avg_cost_per_request_usd": metrics.get("avg_cost_per_request_usd"),
                "total_tokens": metrics.get("total_tokens"),
                "avg_tokens_per_request": metrics.get("avg_tokens_per_request"),
                "avg_prompt_tokens": metrics.get("avg_prompt_tokens"),
                "avg_completion_tokens": metrics.get("avg_completion_tokens")
            },
            "eval_result": eval_result  # Store full result for further analysis
        }

        # Print detailed summary with new metrics
        metrics = aggregated_result['metrics']
        print(f"   📊 Performance:")
        print(f"      Volume Score (traditional): {metrics['volume_score']:.3f}")
        print(f"      Quality Score (penalty-aware): {metrics['quality_score']:.3f}")
        print(f"      Total Examples: {metrics.get('n_total', 'N/A')}")
        print(f"      API Errors: {metrics.get('n_api_errors', 0)} ({metrics.get('api_error_rate', 0):.1%})")
        print(f"      Attempted Rate: {metrics['attempted_rate']:.1%}")
        print(f"      Abstention Rate: {metrics['abstention_rate']:.1%}")
        print(f"      Success on Attempted: {metrics['accuracy_given_attempted']:.1%}")

        # Display judge consistency results if available
        if 'judge_consistency' in metrics:
            consistency = metrics['judge_consistency']
            if consistency.get('consistency_rate') is not None:
                consistency_rate = consistency['consistency_rate']
                if consistency_rate >= 1.0:
                    print(f"      Judge Consistency: {consistency_rate:.1%} ✅")
                else:
                    print(f"      Judge Consistency: {consistency_rate:.1%} ⚠️")
                    print(f"      Inconsistent Responses: {consistency.get('inconsistent_responses', 0)}/{consistency.get('total_tested', 0)}")
            else:
                print(f"      Judge Consistency: Error ({consistency.get('error', 'Unknown')})")

        return aggregated_result

    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")

        if audit_logger:
            audit_logger.log_error(
                component=f"{sampler_name}_quality_evaluation",
                error=str(e),
                context={"samples": n_samples}
            )

        return {
            "sampler_name": sampler_name,
            "duration": None,
            "samples_evaluated": n_samples,
            "success": False,
            "error": str(e),
            "metrics": {}
        }


def run_parallel_quality_evaluations(samplers: Dict[str, Any], n_samples: int, audit_logger: AuditLogger, max_workers: int = 8, use_flex_tier: bool = False) -> List[Dict[str, Any]]:
    """Run quality-based evaluations in parallel across providers"""
    print(f"\n🏃 Running parallel quality evaluations with {max_workers} workers...")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_sampler = {
            executor.submit(
                run_quality_evaluation,
                sampler_name,
                sampler,
                n_samples,
                audit_logger,
                use_flex_tier
            ): sampler_name
            for sampler_name, sampler in samplers.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_sampler):
            sampler_name = future_to_sampler[future]
            try:
                result = future.result()
                results.append(result)

                if result['success']:
                    # Show performance summary with API error info
                    metrics = result['metrics']
                    api_errors = metrics.get('n_api_errors', 0)
                    error_info = f", API_Errors={api_errors}" if api_errors > 0 else ""
                    print(f"   ✅ {sampler_name}: Volume={metrics['volume_score']:.3f}, Quality={metrics['quality_score']:.3f}, Attempted={metrics['attempted_rate']:.1%}{error_info}")
                else:
                    print(f"   ❌ {sampler_name}: Failed")

            except Exception as e:
                print(f"   ❌ {sampler_name}: Exception - {e}")
                results.append({
                    "sampler_name": sampler_name,
                    "success": False,
                    "error": str(e),
                    "metrics": {}
                })

    return results


def generate_quality_benchmark_report(results, output_dir, run_metadata):
    """Generate provider-focused quality benchmark report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"quality_benchmark_report_{timestamp}.html")

    # Build detailed analysis
    successful_results = [r for r in results if r["success"]]

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Provider Quality Benchmark - OpenAI's Penalty-Aware Scoring</title>

    <!-- External Dependencies -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <!-- JavaScript Dependencies -->
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <!-- Open Graph / Social Media Meta Tags -->
    <meta property="og:title" content="RAG Provider Quality Benchmark">
    <meta property="og:description" content="Quality vs Volume strategy comparison using OpenAI's penalty-aware scoring methodology">
    <meta property="og:type" content="website">

    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --quality-color: #8b5cf6;
            --volume-color: #06b6d4;
            --light-bg: #f8fafc;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            color: #1e293b;
        }}

        .main-container {{
            background: white;
            margin: 20px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary-color), #3b82f6);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0 0 10px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin: 10px 0;
        }}

        .header .research-link {{
            display: inline-block;
            margin-top: 15px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            color: white;
            text-decoration: none;
            transition: all 0.3s ease;
        }}

        .header .research-link:hover {{
            background: rgba(255,255,255,0.3);
            color: white;
            text-decoration: none;
        }}

        .content-section {{
            padding: 30px;
        }}

        .about-section {{
            background: var(--light-bg);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid var(--primary-color);
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            border-left: 4px solid var(--primary-color);
        }}

        .summary-card h3 {{
            color: var(--primary-color);
            font-size: 1.1rem;
            margin: 0 0 10px 0;
            font-weight: 600;
        }}

        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin: 5px 0;
        }}

        .summary-card .description {{
            color: var(--secondary-color);
            font-size: 0.9rem;
        }}

        .table-section {{
            margin: 40px 0;
        }}

        .table-section h2 {{
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}

        /* DataTables Styling */
        .dataTables_wrapper {{
            margin-top: 20px;
        }}

        .table {{
            font-size: 0.9rem;
        }}

        .table th {{
            background: var(--light-bg);
            border-top: none;
            font-weight: 600;
            color: #374151;
        }}

        .score-quality {{ color: var(--quality-color); font-weight: 600; }}
        .score-volume {{ color: var(--volume-color); font-weight: 600; }}
        .score-high {{ color: var(--success-color); font-weight: 600; }}
        .score-medium {{ color: var(--warning-color); font-weight: 600; }}
        .score-low {{ color: var(--danger-color); font-weight: 600; }}

        .provider-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
        }}

        .provider-customgpt {{ background: #dcfce7; color: #166534; }}
        .provider-openai-rag {{ background: #dbeafe; color: #1e40af; }}
        .provider-openai-vanilla {{ background: #fef3c7; color: #92400e; }}

        .strategy-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .strategy-quality {{ background: #f3e8ff; color: #7c3aed; }}
        .strategy-volume {{ background: #ecfdf5; color: #059669; }}
        .strategy-balanced {{ background: #fef3c7; color: #d97706; }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .main-container {{ margin: 10px; }}
            .content-section {{ padding: 20px; }}
            .header {{ padding: 30px 20px; }}
            .header h1 {{ font-size: 2rem; }}
            .summary-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <!-- Header Section -->
        <div class="header">
            <h1><i class="fas fa-trophy me-3"></i>RAG Provider Quality Benchmark</h1>
            <div class="subtitle">Quality vs Volume Strategy Comparison</div>
            <div class="subtitle">Using OpenAI's Penalty-Aware Scoring (80% Confidence Threshold)</div>
            <a href="https://openai.com/index/why-language-models-hallucinate/" target="_blank" class="research-link">
                <i class="fas fa-external-link-alt me-2"></i>Read the OpenAI Research Paper
            </a>
            <div style="margin-top: 15px; opacity: 0.8;">
                <small>Generated: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</small>
            </div>
        </div>

        <!-- Content Section -->
        <div class="content-section">
            <!-- About This Benchmark -->
            <div class="about-section">
                <h2><i class="fas fa-info-circle me-2"></i>About This Benchmark</h2>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Research Foundation:</strong> Implements OpenAI's penalty-aware scoring from <a href="https://openai.com/index/why-language-models-hallucinate/" target="_blank">"Why Language Models Hallucinate"</a> (arXiv:2509.04664v1). Uses the recommended 80% confidence threshold to evaluate quality vs volume strategies.</p>
                        <p><strong>Dataset:</strong> <a href="https://github.com/openai/simple-evals" target="_blank">SimpleQA benchmark</a> - factual knowledge and reasoning evaluation.</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Methodology:</strong></p>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-robot me-2"></i><strong>Providers:</strong> GPT-5.1 / Gemini 3 Pro (SOTA December 2025)</li>
                            <li><i class="fas fa-gavel me-2"></i><strong>Judge:</strong> GPT-5.1 {"with Flex Tier (50% cost savings, slower)" if run_metadata.get("use_flex_tier", False) else "Standard Tier (faster responses)"}</li>
                            <li><i class="fas fa-shield-alt me-2"></i><strong>Threshold:</strong> 80% confidence (recommended in paper)</li>
                            <li><i class="fas fa-calculator me-2"></i><strong>Penalty Ratio:</strong> 4.0 (wrong answers = -4 points)</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Executive Summary -->
            <div class="summary-grid">
                <div class="summary-card">
                    <h3><i class="fas fa-users me-2"></i>Providers Tested</h3>
                    <div class="value">{len(successful_results)}</div>
                    <div class="description">RAG providers evaluated</div>
                </div>
                <div class="summary-card">
                    <h3><i class="fas fa-questions me-2"></i>Questions Per Provider</h3>
                    <div class="value">{run_metadata.get('samples_per_provider', 'N/A')}</div>
                    <div class="description">SimpleQA evaluation questions</div>
                </div>
                <div class="summary-card">
                    <h3><i class="fas fa-tasks me-2"></i>Total Evaluations</h3>
                    <div class="value">{run_metadata.get('actual_total_evaluations', 'N/A')}</div>
                    <div class="description">Questions × Providers</div>
                </div>
                <div class="summary-card">
                    <h3><i class="fas fa-check-circle me-2"></i>Coverage Status</h3>
                    <div class="value" style="color: {'var(--success-color)' if run_metadata.get('evaluation_coverage_complete', False) else 'var(--warning-color)'};">
                        {'✅ Complete' if run_metadata.get('evaluation_coverage_complete', False) else '⚠️ Partial'}
                    </div>
                    <div class="description">{run_metadata.get('actual_total_evaluations', 0)}/{run_metadata.get('expected_total_evaluations', 0)} evaluations</div>
                </div>
            </div>

            <!-- Key Research Insights -->
            <div class="alert alert-info">
                <h4><i class="fas fa-lightbulb me-2"></i>Quality vs Volume Strategy Comparison</h4>
                <div class="row">
                    <div class="col-md-6">
                        <h6><strong><i class="fas fa-chart-line me-1" style="color: var(--volume-color);"></i>Volume Strategy (Traditional)</strong></h6>
                        <ul>
                            <li><strong>Scoring:</strong> Correct=+1, Wrong=0, IDK=0</li>
                            <li><strong>Philosophy:</strong> Rewards guessing and penalizes uncertainty</li>
                            <li><strong>Problem:</strong> Creates "I Don't Know" tax for conservative systems</li>
                            <li><strong>Best for:</strong> High-volume, low-stakes applications</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6><strong><i class="fas fa-shield-alt me-1" style="color: var(--quality-color);"></i>Quality Strategy (Penalty-Aware)</strong></h6>
                        <ul>
                            <li><strong>Scoring:</strong> Correct=+1, Wrong=-4, IDK=0</li>
                            <li><strong>Philosophy:</strong> Rewards appropriate uncertainty and calibration</li>
                            <li><strong>Advantage:</strong> Penalizes overconfident incorrect responses</li>
                            <li><strong>Best for:</strong> High-stakes applications where accuracy matters</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Provider Leaderboard -->
            <div class="table-section">
                <h2><i class="fas fa-trophy me-2"></i>Provider Performance Leaderboard</h2>
                <div class="table-responsive">
                    <table id="providerTable" class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Provider</th>
                                <th>Total Examples</th>
                                <th>API Errors</th>
                                <th><i class="fas fa-shield-alt me-1" style="color: var(--quality-color);"></i>Quality Score</th>
                                <th><i class="fas fa-chart-line me-1" style="color: var(--volume-color);"></i>Volume Score</th>
                                <th>Attempted Rate</th>
                                <th>Success Rate</th>
                                <th>Abstention Rate</th>
                                <th>Strategy Assessment</th>
                            </tr>
                        </thead>
                        <tbody>"""

    # Sort providers by quality score (descending)
    sorted_results = sorted(successful_results, key=lambda x: x['metrics']['quality_score'], reverse=True)

    # Add provider data to table
    for rank, result in enumerate(sorted_results, 1):
        provider_name = result["sampler_name"]
        metrics = result["metrics"]

        # Determine strategy assessment
        quality_score = metrics['quality_score']
        volume_score = metrics['volume_score']
        attempted_rate = metrics['attempted_rate']

        if quality_score > volume_score:
            strategy_assessment = "Quality-Focused"
            strategy_class = "strategy-quality"
        elif volume_score > quality_score:
            strategy_assessment = "Volume-Focused"
            strategy_class = "strategy-volume"
        else:
            strategy_assessment = "Balanced"
            strategy_class = "strategy-balanced"

        # Provider badge styling
        provider_class = f"provider-{provider_name.lower().replace('_', '-')}"

        # Score styling
        quality_class = "score-high" if quality_score > 0.6 else "score-medium" if quality_score > 0.3 else "score-low"
        volume_class = "score-high" if volume_score > 0.6 else "score-medium" if volume_score > 0.3 else "score-low"

        # Get the new metrics with fallback for older data
        n_total = metrics.get('n_total', metrics.get('conversations', 0))
        n_api_errors = metrics.get('n_api_errors', 0)
        api_error_rate = metrics.get('api_error_rate', 0)

        # Style API errors based on severity
        api_error_class = "score-high" if api_error_rate == 0 else "score-medium" if api_error_rate < 0.1 else "score-low"

        html_content += f"""
                            <tr>
                                <td><strong>#{rank}</strong></td>
                                <td><span class="provider-badge {provider_class}">{provider_name}</span></td>
                                <td>{n_total}</td>
                                <td><span class="{api_error_class}">{n_api_errors} ({api_error_rate:.1%})</span></td>
                                <td><span class="score-quality {quality_class}">{quality_score:.3f}</span></td>
                                <td><span class="score-volume {volume_class}">{volume_score:.3f}</span></td>
                                <td>{attempted_rate:.1%}</td>
                                <td>{metrics['accuracy_given_attempted']:.1%}</td>
                                <td>{metrics['abstention_rate']:.1%}</td>
                                <td><span class="strategy-badge {strategy_class}">{strategy_assessment}</span></td>
                            </tr>"""

    html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Methodology Notes -->
            <div class="alert alert-secondary">
                <h6><i class="fas fa-info-circle me-2"></i>Methodology Notes</h6>
                <p><strong>80% Confidence Threshold:</strong> Based on OpenAI's research recommendation, judges only accept responses where they are >80% confident in the evaluation. This conservative threshold filters out borderline decisions that could distort results.</p>
                <p><strong>Penalty Ratio 4.0:</strong> Wrong answers receive -4 points, making incorrect responses costly relative to abstaining (0 points). This encourages appropriate uncertainty rather than overconfident guessing.</p>
                <p><strong>API Error Handling:</strong> Technical failures (timeouts, rate limits, network errors) are tracked separately and excluded from scoring calculations. This prevents technical issues from being confused with intentional abstentions.</p>
                <p><strong>Intelligent Abstention Detection:</strong> Uses GPT-5-nano classifier with few-shot examples to distinguish intentional abstentions ("I don't know") from uncertain attempts ("I think it might be X"). Replaces brittle string matching with robust intent classification.</p>
                <p><strong>Post-hoc Evaluation:</strong> Providers give natural responses without threshold contamination. The confidence framework is applied during evaluation, not during response generation.</p>
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
                    {{ targets: [2, 3], type: 'num' }},
                    {{ targets: [4, 5, 6], render: $.fn.dataTable.render.percentBar('round', '#dbeafe', '#1e40af', '#f1f5f9', '#374151', 0, 'right') }}
                ],
                language: {{
                    search: "Search providers:",
                    lengthMenu: "Show _MENU_ entries"
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    # Write report
    with open(report_file, 'w') as f:
        f.write(html_content)

    return report_file



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAG Provider Quality Benchmark using OpenAI's Penalty-Aware Scoring"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples to test (default: 10 for normal, 5 for debug)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited examples"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running evaluations"
    )
    parser.add_argument(
        "--flex-tier",
        action="store_true",
        help="Enable GPT-5 Flex tier for judge (50%% cost savings but much slower)"
    )
    return parser.parse_args()


def main():
    """Run RAG provider quality benchmark"""
    args = parse_args()

    # Extract arguments early
    debug_mode = args.debug
    max_workers = args.max_workers
    output_dir = args.output_dir
    dry_run = args.dry_run
    use_flex_tier = args.flex_tier

    print("🏆 RAG PROVIDER QUALITY BENCHMARK")
    print("=" * 70)
    print("Implementing OpenAI's Penalty-Aware Scoring Framework")
    print("Quality vs Volume Strategy Comparison using 80% Confidence Threshold")
    print("=" * 70)
    print("Comparing:")
    print("  1. OpenAI Vanilla (no RAG - baseline) [gpt-5.1]")
    print("  2. OpenAI RAG (vector store file search) [gpt-5.1]")
    print("  3. CustomGPT (RAG with existing knowledge base) [gpt-5.1]")
    print("  4. Google Gemini RAG (File Search) [gemini-3-pro]")
    if use_flex_tier:
        print("LLM-As-A-Judge: GPT-5.1 with Flex Tier (improved reliability, 50% cost savings, MUCH slower)")
    else:
        print("LLM-As-A-Judge: GPT-5.1 Standard Tier (improved reliability, faster responses)")
    print("Quality Methodology: 80% confidence threshold, penalty ratio 4.0")
    print("=" * 70)

    # Configuration
    if args.examples is not None:
        n_samples = args.examples
    elif args.debug:
        n_samples = 5
    else:
        n_samples = 10

    if debug_mode:
        print("🐛 DEBUG MODE ENABLED")

    print(f"📊 Configuration:")
    print(f"   Samples per provider: {n_samples}")
    print(f"   Confidence threshold: 80% (Conservative)")
    print(f"   Max parallel workers: {max_workers}")
    print(f"   Output directory: {output_dir}")
    print(f"   Debug mode: {debug_mode}")
    print(f"   Flex tier enabled: {use_flex_tier}")

    # Show methodology details
    print(f"\n🎯 Quality Evaluation Methodology:")
    threshold = CONFIDENCE_THRESHOLDS[0]  # Single threshold
    print(f"   {threshold.name}: t={threshold.threshold}, penalty={threshold.penalty_ratio}")
    print(f"   Scoring: Correct=+1, Wrong=-{threshold.penalty_ratio}, IDK=0")
    print(f"   Focus: Penalty-aware scoring to discourage overconfident mistakes")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize audit logging
    run_id = create_run_id()
    audit_logger = AuditLogger(run_id, output_dir)

    # Reset blind evaluation system for new benchmark run
    ConfidenceThresholdSimpleQAEval.reset_blind_evaluation()

    # Display judge configuration for full transparency
    temp_eval = ConfidenceThresholdSimpleQAEval(num_examples=1, audit_logger=audit_logger, use_flex_tier=use_flex_tier)
    temp_eval.print_judge_configuration_summary()

    print(f"\n📋 Audit Logging:")
    print(f"   Run ID: {run_id}")
    print(f"   Logs directory: {audit_logger.run_dir}")

    # Setup samplers
    print(f"\n🔧 Setting up samplers...")
    samplers, setup_errors = setup_samplers(audit_logger)

    if not samplers:
        print("❌ No samplers were successfully initialized!")
        for error in setup_errors:
            print(f"   - {error}")
        return 1

    print(f"\n✅ Successfully initialized {len(samplers)} samplers:")
    for name in samplers.keys():
        print(f"   - {name}")

    if setup_errors:
        print(f"\n⚠️  Setup warnings:")
        for error in setup_errors:
            print(f"   - {error}")

    # Dry run mode - just validate configuration
    if dry_run:
        print(f"\n🔍 DRY RUN MODE: Configuration validated successfully")
        print(f"   Ready to run {len(samplers)} providers")
        print(f"   Total evaluations: {len(samplers)}")
        print(f"   Questions per evaluation: {n_samples}")
        print(f"   Total questions to process: {len(samplers) * n_samples}")
        print(f"   Estimated runtime: ~{len(samplers) * n_samples * 4 / max_workers / 60:.1f} minutes")
        print(f"   Audit logs will be saved to: {audit_logger.run_dir}")
        return 0

    # Run evaluations (parallel across providers)
    results = run_parallel_quality_evaluations(samplers, n_samples, audit_logger, max_workers, use_flex_tier)

    # CRITICAL: Validate complete evaluation coverage
    expected_total_evaluations = len(samplers) * n_samples
    actual_total_evaluations = 0

    print(f"\n🔍 EVALUATION COVERAGE VALIDATION:")
    print("=" * 70)

    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    # Count actual evaluations completed
    for result in successful_results:
        conversations_completed = result["metrics"].get("conversations", 0)
        actual_total_evaluations += conversations_completed
        print(f"   ✅ {result['sampler_name']}: {conversations_completed} evaluations")

    # Report any failures that reduce evaluation count
    for result in failed_results:
        missed_evaluations = n_samples
        print(f"   ❌ {result['sampler_name']}: FAILED - missed {missed_evaluations} evaluations")

    print(f"\n📊 COVERAGE SUMMARY:")
    print(f"   Expected total evaluations: {expected_total_evaluations}")
    print(f"   Actual completed evaluations: {actual_total_evaluations}")

    if actual_total_evaluations < expected_total_evaluations:
        missing_evaluations = expected_total_evaluations - actual_total_evaluations
        print(f"   ⚠️  INCOMPLETE DATASET: Missing {missing_evaluations} evaluations ({missing_evaluations/expected_total_evaluations*100:.1f}%)")
        print(f"   ⚠️  Statistical analysis will be invalid with incomplete data!")
        if not debug_mode:
            print(f"   💡 Consider re-running failed providers or increasing retry logic")
    else:
        print(f"   ✅ COMPLETE DATASET: All {actual_total_evaluations} evaluations completed")

    # Create results summary
    print(f"\n📊 QUALITY BENCHMARK RESULTS SUMMARY:")
    print("=" * 70)

    if successful_results:
        print(f"🏆 PROVIDER PERFORMANCE LEADERBOARD:")

        # Sort by quality score for leaderboard display
        sorted_results = sorted(successful_results, key=lambda x: x['metrics']['quality_score'], reverse=True)

        for rank, result in enumerate(sorted_results, 1):
            provider_name = result["sampler_name"]
            metrics = result["metrics"]

            volume_score = metrics['volume_score']
            quality_score = metrics['quality_score']
            attempted_rate = metrics['attempted_rate']
            abstention_rate = metrics.get('abstention_rate', 1.0 - attempted_rate)

            # Determine strategy focus
            if quality_score > volume_score:
                strategy = "Quality-Focused"
            elif volume_score > quality_score:
                strategy = "Volume-Focused"
            else:
                strategy = "Balanced"

            print(f"\n   #{rank} {provider_name} ({strategy}):")
            print(f"      🏆 Quality Score: {quality_score:.3f}")
            print(f"      📊 Volume Score: {volume_score:.3f}")
            print(f"      📈 Attempted Rate: {attempted_rate:.1%}")
            print(f"      🛡️  Abstention Rate: {abstention_rate:.1%}")
            print(f"      ✅ Success Rate: {metrics['accuracy_given_attempted']:.1%}")
            print(f"      ⚠️  Overconfidence Penalty: {metrics['overconfidence_penalty']}")
            # Latency metrics
            latency_avg = metrics.get('provider_latency_avg_ms')
            latency_p95 = metrics.get('provider_latency_p95_ms')
            if latency_avg is not None:
                print(f"      ⏱️  Avg Latency: {latency_avg:,.0f}ms (p95: {latency_p95:,.0f}ms)" if latency_p95 else f"      ⏱️  Avg Latency: {latency_avg:,.0f}ms")
            else:
                print(f"      ⏱️  Avg Latency: N/A")
            # Cost metrics
            total_cost = metrics.get('total_cost_usd')
            avg_tokens = metrics.get('avg_tokens_per_request')
            if total_cost is not None:
                print(f"      💰 Total Cost: ${total_cost:.6f}")
            else:
                print(f"      💰 Total Cost: N/A (subscription model)")
            if avg_tokens is not None:
                print(f"      📦 Avg Tokens: {avg_tokens:,.0f}/request")

    if failed_results:
        print(f"\n❌ FAILED EVALUATIONS:")
        for result in failed_results:
            print(f"   - {result['sampler_name']}: {result['error']}")

    # Create JSON-serializable results (remove EvalResult objects)
    json_serializable_results = []
    for result in results:
        serializable_result = result.copy()
        # Remove the eval_result object which isn't JSON serializable
        if "eval_result" in serializable_result:
            del serializable_result["eval_result"]
        json_serializable_results.append(serializable_result)

    # Finalize audit logging with results and validation
    audit_logger.finalize_run({
        "total_providers": len(samplers),
        "successful_evaluations": len(successful_results),
        "failed_evaluations": len(failed_results),
        "samples_per_provider": n_samples,
        "confidence_threshold": "80% (Conservative)",
        "expected_total_evaluations": expected_total_evaluations,
        "actual_total_evaluations": actual_total_evaluations,
        "evaluation_coverage_complete": actual_total_evaluations >= expected_total_evaluations,
        "results": json_serializable_results
    })

    # Perform statistical analysis across providers
    print(f"\n📊 Performing statistical analysis...")
    statistical_analysis = None

    # Check if we have sufficient data for valid statistical analysis
    if actual_total_evaluations < expected_total_evaluations:
        print(f"   ⚠️  SKIPPING STATISTICAL ANALYSIS: Incomplete dataset ({actual_total_evaluations}/{expected_total_evaluations} evaluations)")
        print(f"   ⚠️  Statistical analysis requires complete data to avoid biased results")
    elif len(successful_results) >= 2:
        # Reconstruct provider results for statistical analysis
        provider_results = {}
        for result in successful_results:
            provider_name = result["sampler_name"]
            # Use the stored eval_result from the simplified structure
            provider_results[provider_name] = result["eval_result"]

        # Create evaluator instance for statistical analysis
        eval_instance = ConfidenceThresholdSimpleQAEval(num_examples=n_samples)
        statistical_analysis = eval_instance.analyze_statistical_significance(provider_results)

        # Reveal provider anonymization mapping after analysis is complete
        eval_instance.reveal_provider_mapping()

        # Print statistical summary
        summary = statistical_analysis["summary"]
        print(f"   📈 Statistical Summary:")
        print(f"      Total pairwise comparisons: {summary['total_comparisons']}")
        print(f"      Total statistical tests: {summary['total_statistical_tests']}")
        print(f"      Bonferroni corrected α: {summary['bonferroni_corrected_alpha']:.4f}")
        print(f"      Significant volume differences (raw): {summary['significant_volume_comparisons_raw']}")
        print(f"      Significant volume differences (corrected): {summary['significant_volume_comparisons_corrected']}")
        print(f"      Significant quality differences (raw): {summary['significant_quality_comparisons_raw']}")
        print(f"      Significant quality differences (corrected): {summary['significant_quality_comparisons_corrected']}")
        print(f"      Significant distribution differences (raw): {summary['significant_distribution_comparisons_raw']}")
        print(f"      Significant distribution differences (corrected): {summary['significant_distribution_comparisons_corrected']}")
        print(f"      Large effect sizes detected: {summary['large_effect_sizes']}")
        print(f"      Medium effect sizes detected: {summary['medium_effect_sizes']}")
    else:
        print(f"   ⚠️ Statistical analysis requires at least 2 successful providers (found {len(successful_results)})")

    # Generate comprehensive quality benchmark report
    print(f"\n📄 Generating quality benchmark report...")
    run_metadata = {
        "run_id": run_id,
        "samples_per_provider": n_samples,
        "max_workers": max_workers,
        "debug_mode": debug_mode,
        "confidence_threshold": "80% (Conservative)",
        "expected_total_evaluations": expected_total_evaluations,
        "actual_total_evaluations": actual_total_evaluations,
        "evaluation_coverage_complete": actual_total_evaluations >= expected_total_evaluations,
        "statistical_analysis": statistical_analysis,
        "use_flex_tier": use_flex_tier
    }

    report_file = generate_quality_benchmark_report_v2(
        results,
        str(audit_logger.run_dir),
        run_metadata
    )
    print(f"   Quality benchmark report saved: {report_file}")

    # Generate statistical analysis HTML report (if we have statistical data)
    if statistical_analysis and statistical_analysis.get("summary", {}).get("total_statistical_tests", 0) > 0:
        try:
            stat_report_file = generate_statistical_analysis_report_v2(
                str(audit_logger.run_dir),
                run_id,
                statistical_analysis
            )
            print(f"   Statistical analysis report saved: {stat_report_file}")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not generate statistical analysis HTML: {e}")

    # Save JSON results
    json_file = audit_logger.run_dir / "quality_benchmark_results.json"
    with open(json_file, 'w') as f:
        ct = CONFIDENCE_THRESHOLDS[0]  # Single threshold
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "framework": "quality_benchmark",
            "configuration": {
                "samples_per_provider": n_samples,
                "max_workers": max_workers,
                "debug_mode": debug_mode,
                "output_directory": output_dir,
                "confidence_threshold": {
                    "name": ct.name,
                    "threshold": ct.threshold,
                    "penalty_ratio": ct.penalty_ratio,
                    "description": ct.description
                }
            },
            "audit_summary": audit_logger.get_run_summary(),
            "results": json_serializable_results,
            "evaluation_validation": {
                "expected_total_evaluations": expected_total_evaluations,
                "actual_total_evaluations": actual_total_evaluations,
                "coverage_complete": actual_total_evaluations >= expected_total_evaluations,
                "missing_evaluations": max(0, expected_total_evaluations - actual_total_evaluations),
                "coverage_percentage": (actual_total_evaluations / expected_total_evaluations * 100) if expected_total_evaluations > 0 else 0
            },
            "summary": {
                "total_providers": len(results),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results)
            }
        }, f, indent=2)

    print(f"   JSON results saved: {json_file}")
    print(f"\n📋 Complete audit trail available in: {audit_logger.run_dir}")

    # Auto-generate forensic reports for providers with penalties
    print(f"\n🔬 Generating forensic reports...")
    forensic_count = 0

    # Correct provider name mapping for penalty analyzer
    provider_mapping = {
        "CustomGPT_RAG": "customgpt",
        "OpenAI_RAG": "openai_rag",
        "OpenAI_Vanilla": "openai_vanilla"
    }

    for result in successful_results:
        provider_name = result["sampler_name"]
        wrong_count = result["metrics"].get("n_incorrect", 0)

        if wrong_count > 0:
            print(f"\n   Analyzing {provider_name} ({wrong_count} penalty cases)...")
            provider_key = provider_mapping.get(provider_name, provider_name.lower())

            try:
                # Step 1: Run penalty analyzer
                import subprocess
                analyzer_cmd = [
                    "python", "scripts/universal_penalty_analyzer.py",
                    "--run-dir", str(audit_logger.run_dir),
                    "--provider", provider_key
                ]
                analyzer_result = subprocess.run(analyzer_cmd, capture_output=True, text=True, check=True)
                print(f"      ✓ Penalty analysis complete")

                # Step 2: Generate forensic reports
                forensic_cmd = [
                    "python", "scripts/generate_universal_forensics.py",
                    "--run-dir", str(audit_logger.run_dir),
                    "--provider", provider_key
                ]
                forensic_result = subprocess.run(forensic_cmd, capture_output=True, text=True, check=True)
                print(f"      ✓ Forensic reports generated")
                forensic_count += 1

            except subprocess.CalledProcessError as e:
                print(f"      ⚠️  Warning: Forensic generation failed for {provider_name}")
                print(f"      Error: {e.stderr if e.stderr else str(e)}")
            except Exception as e:
                print(f"      ⚠️  Warning: Unexpected error for {provider_name}: {e}")
        else:
            print(f"   ✓ {provider_name}: No penalties, skipping forensics")

    if forensic_count > 0:
        print(f"\n   ✅ Generated forensics for {forensic_count} provider(s)")
    else:
        print(f"\n   ℹ️  No forensic reports needed (no penalties)")

    # Generate main dashboard hub
    print(f"\n📊 Generating main dashboard hub...")
    try:
        dashboard_file = generate_main_dashboard(
            results_dir=str(audit_logger.run_dir),
            run_metadata={
                "run_id": run_metadata.get("run_id", ""),
                "timestamp": datetime.now(),
                "providers": [r["sampler_name"] for r in results if r["success"]],
                "total_questions": run_metadata.get("samples_per_provider", 5),
                "confidence_threshold": 0.8  # Fixed threshold for display
            }
        )
        print(f"   Main dashboard saved: {dashboard_file}")
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to generate main dashboard: {e}")
        print(f"   This is non-critical - other reports are still available")
        import traceback
        traceback.print_exc()  # Debug: show full traceback

    print(f"\n🎉 QUALITY BENCHMARK COMPLETE!")
    print(f"📁 Results directory: {audit_logger.run_dir}")
    print(f"🏠 Main dashboard: {audit_logger.run_dir}/index.html")
    print(f"📊 Quality benchmark: {report_file}")
    print(f"📄 JSON results: {json_file}")
    if forensic_count > 0:
        print(f"🔬 Forensic reports: {forensic_count} provider(s) analyzed")
    print(f"\n💡 Open the main dashboard (index.html) in your browser to navigate all reports")

    return 0


if __name__ == "__main__":
    sys.exit(main())