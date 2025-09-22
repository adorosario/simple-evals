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
from confidence_threshold_simpleqa_eval import ConfidenceThresholdSimpleQAEval, CONFIDENCE_THRESHOLDS
from audit_logger import AuditLogger, create_run_id
from leaderboard_generator import LeaderboardGenerator


def setup_samplers(audit_logger=None):
    """Initialize all providers for comparison with audit logging"""
    samplers = {}
    errors = []

    # 1. CustomGPT RAG Sampler (with GPT-4.1)
    print("🔧 Setting up CustomGPT (RAG) sampler...")
    try:
        os.environ["CUSTOMGPT_MODEL_NAME"] = "gpt-4.1"
        sampler = AuditedCustomGPTSampler.from_env(audit_logger=audit_logger)
        samplers["CustomGPT_RAG"] = sampler

        if audit_logger:
            audit_logger.add_provider("CustomGPT_RAG", sampler._get_request_data([]))

        print("   ✅ CustomGPT sampler ready (GPT-4.1)")
    except Exception as e:
        error_msg = f"CustomGPT setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 2. OpenAI Vanilla Sampler (no RAG, using GPT-4.1)
    print("🔧 Setting up OpenAI Vanilla (no RAG) sampler...")
    try:
        sampler = AuditedOpenAIVanillaSampler(
            model="gpt-4.1",
            system_message="You are a helpful assistant. Answer questions based on your training knowledge.",
            temperature=0.3,
            audit_logger=audit_logger
        )
        samplers["OpenAI_Vanilla"] = sampler

        if audit_logger:
            audit_logger.add_provider("OpenAI_Vanilla", sampler._get_request_data([]))

        print("   ✅ OpenAI Vanilla sampler ready (GPT-4.1)")
    except Exception as e:
        error_msg = f"OpenAI Vanilla setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 3. OpenAI RAG Sampler (with vector store, using GPT-4.1)
    print("🔧 Setting up OpenAI RAG (vector store) sampler...")
    try:
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        if vector_store_id:
            sampler = AuditedOpenAIRAGSampler(
                model="gpt-4.1",
                system_message="You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.",
                temperature=0.3,
                audit_logger=audit_logger
            )
            samplers["OpenAI_RAG"] = sampler

            if audit_logger:
                audit_logger.add_provider("OpenAI_RAG", sampler._get_request_data([]))

            print("   ✅ OpenAI RAG sampler ready (GPT-4.1)")
        else:
            error_msg = "OpenAI RAG setup failed: OPENAI_VECTOR_STORE_ID not set"
            print(f"   ❌ {error_msg}")
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"OpenAI RAG setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    return samplers, errors


def run_confidence_threshold_evaluation(sampler_name, sampler, n_samples=10, audit_logger=None):
    """Run confidence threshold evaluation for a single sampler"""
    print(f"\n🎯 Running confidence threshold evaluation for {sampler_name}...")
    print(f"   Samples: {n_samples}")
    print(f"   Thresholds: {len(CONFIDENCE_THRESHOLDS)}")

    try:
        # Create confidence threshold evaluation
        eval_instance = ConfidenceThresholdSimpleQAEval(
            grader_model=None,  # Uses GPT-5-mini by default
            num_examples=n_samples,
            audit_logger=audit_logger
        )

        start_time = time.time()

        # Run multi-threshold evaluation
        threshold_results = eval_instance(sampler, provider_name=sampler_name)

        end_time = time.time()
        duration = end_time - start_time

        print(f"   ✅ Completed in {duration:.1f}s")

        # Aggregate results across thresholds
        aggregated_result = {
            "sampler_name": sampler_name,
            "duration": duration,
            "samples_evaluated": n_samples,
            "success": True,
            "error": None,
            "threshold_results": {}
        }

        # Process each threshold result
        for threshold_name, eval_result in threshold_results.items():
            threshold_metrics = eval_result.metrics
            aggregated_result["threshold_results"][threshold_name] = {
                "volume_score": threshold_metrics.get("volume_score_mean", 0),
                "quality_score": threshold_metrics.get("quality_score_mean", 0),
                "attempted_rate": threshold_metrics.get("attempted_rate", 0),
                "accuracy_given_attempted": threshold_metrics.get("accuracy_given_attempted", 0),
                "abstention_rate": threshold_metrics.get("abstention_rate", 0),
                "overconfidence_penalty": threshold_metrics.get("overconfidence_penalty", 0),
                "threshold_value": threshold_metrics.get("threshold_value", 0),
                "penalty_ratio": threshold_metrics.get("penalty_ratio", 0),
                "n_correct": threshold_metrics.get("n_correct", 0),
                "n_incorrect": threshold_metrics.get("n_incorrect", 0),
                "n_not_attempted": threshold_metrics.get("n_not_attempted", 0),
                "conversations": len(eval_result.convos)
            }

        # Print detailed summary
        print(f"   📊 Threshold Performance:")
        for threshold_name, metrics in aggregated_result["threshold_results"].items():
            print(f"      {threshold_name}: Volume={metrics['volume_score']:.3f}, Quality={metrics['quality_score']:.3f}, Attempted={metrics['attempted_rate']:.3f}")

        return aggregated_result

    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")

        if audit_logger:
            audit_logger.log_error(
                component=f"{sampler_name}_confidence_evaluation",
                error=str(e),
                context={"samples": n_samples}
            )

        return {
            "sampler_name": sampler_name,
            "duration": None,
            "samples_evaluated": n_samples,
            "success": False,
            "error": str(e),
            "threshold_results": {}
        }


def run_parallel_confidence_evaluations(samplers: Dict[str, Any], n_samples: int, audit_logger: AuditLogger, max_workers: int = 3) -> List[Dict[str, Any]]:
    """Run confidence threshold evaluations in parallel across providers"""
    print(f"\n🏃 Running parallel confidence threshold evaluations with {max_workers} workers...")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_sampler = {
            executor.submit(
                run_confidence_threshold_evaluation,
                sampler_name,
                sampler,
                n_samples,
                audit_logger
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
                    # Show best threshold performance
                    best_volume = max(result['threshold_results'].values(), key=lambda x: x['volume_score'])
                    best_quality = max(result['threshold_results'].values(), key=lambda x: x['quality_score'])
                    print(f"   ✅ {sampler_name}: Best Volume={best_volume['volume_score']:.3f}, Best Quality={best_quality['quality_score']:.3f}")
                else:
                    print(f"   ❌ {sampler_name}: Failed")

            except Exception as e:
                print(f"   ❌ {sampler_name}: Exception - {e}")
                results.append({
                    "sampler_name": sampler_name,
                    "success": False,
                    "error": str(e),
                    "threshold_results": {}
                })

    return results


def generate_confidence_threshold_report(results, output_dir, run_metadata):
    """Generate comprehensive confidence threshold report with modern UI"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"confidence_threshold_report_{timestamp}.html")

    # Build detailed analysis
    successful_results = [r for r in results if r["success"]]


    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Benchmark Results: OpenAI's new Quality Metric using SimpleQA benchmark</title>

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
    <meta property="og:title" content="RAG Benchmark Results: OpenAI's new Quality Metric">
    <meta property="og:description" content="Comprehensive evaluation of RAG providers using OpenAI's confidence threshold framework">
    <meta property="og:type" content="website">

    <style>
        :root {{
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
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

        .grade-correct {{ color: var(--success-color); font-weight: 600; }}
        .grade-incorrect {{ color: var(--danger-color); font-weight: 600; }}
        .grade-not-attempted {{ color: var(--secondary-color); font-weight: 600; }}

        .threshold-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        .threshold-balanced {{ background: #dbeafe; color: #1e40af; }}
        .threshold-conservative {{ background: #fef3c7; color: #92400e; }}
        .threshold-cautious {{ background: #fecaca; color: #991b1b; }}

        .provider-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
        }}

        .provider-customgpt {{ background: #dcfce7; color: #166534; }}
        .provider-openai {{ background: #dbeafe; color: #1e40af; }}

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
            <h1><i class="fas fa-chart-line me-3"></i>RAG Benchmark Results</h1>
            <div class="subtitle">OpenAI's new Quality Metric using SimpleQA benchmark</div>
            <div class="subtitle">Volume Strategy vs Quality Strategy with Confidence Thresholds</div>
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
                        <p><strong>Research Foundation:</strong> This evaluation implements the confidence threshold framework from OpenAI's paper <a href="https://openai.com/index/why-language-models-hallucinate/" target="_blank">"Why Language Models Hallucinate"</a>, which identifies that traditional evaluation systems penalize uncertainty and reward guessing.</p>
                        <p><strong>Dataset:</strong> We use the <a href="https://github.com/openai/simple-evals" target="_blank">SimpleQA benchmark</a>, a comprehensive question-answering dataset designed to test factual knowledge and reasoning capabilities.</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>LLM Models:</strong></p>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-robot me-2"></i><strong>Providers:</strong> GPT-4.1 (standardized across all RAG systems)</li>
                            <li><i class="fas fa-gavel me-2"></i><strong>Judge:</strong> GPT-5-mini (87.5% cost reduction vs GPT-4.1)</li>
                        </ul>
                        <p><strong>Methodology:</strong> Post-hoc evaluation where providers give natural responses that are then assessed against different confidence threshold criteria.</p>
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
                    <h3><i class="fas fa-layer-group me-2"></i>Confidence Thresholds</h3>
                    <div class="value">3</div>
                    <div class="description">Balanced (50%), Conservative (75%), Cautious (90%)</div>
                </div>
                <div class="summary-card">
                    <h3><i class="fas fa-tasks me-2"></i>Total Evaluations</h3>
                    <div class="value">{run_metadata.get('actual_total_evaluations', 'N/A')}</div>
                    <div class="description">Questions × Providers × Thresholds</div>
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
                <h4><i class="fas fa-lightbulb me-2"></i>Key Insights from OpenAI Research</h4>
                <div class="row">
                    <div class="col-md-6">
                        <h6><strong>Volume Strategy (Traditional)</strong></h6>
                        <ul>
                            <li>Binary scoring: Correct=1, Wrong=0, IDK=0</li>
                            <li>Rewards guessing and penalizes uncertainty</li>
                            <li>Creates "I Don't Know" tax for conservative systems</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6><strong>Quality Strategy (Penalty-Aware)</strong></h6>
                        <ul>
                            <li>Threshold scoring: Correct=+1, Wrong=-k, IDK=0</li>
                            <li>Rewards appropriate uncertainty and calibration</li>
                            <li>Penalizes overconfident incorrect responses</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Provider Comparison Table -->
            <div class="table-section">
                <h2><i class="fas fa-chart-bar me-2"></i>Provider Performance Comparison</h2>
                <div class="table-responsive">
                    <table id="providerTable" class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Provider</th>
                                <th>Strategy</th>
                                <th>Volume Score</th>
                                <th>Quality Score</th>
                                <th>Attempted Rate</th>
                                <th>Success Rate</th>
                                <th>Abstention Rate</th>
                                <th>Overconfidence Penalty</th>
                                <th>Assessment</th>
                            </tr>
                        </thead>
                        <tbody>"""

    # Add provider comparison data to table
    for result in successful_results:
        provider_name = result["sampler_name"]
        threshold_results = result["threshold_results"]

        for threshold_name, metrics in threshold_results.items():
            strategy_assessment = "Aggressive" if metrics['attempted_rate'] > 0.9 else "Conservative" if metrics['attempted_rate'] < 0.7 else "Balanced"
            provider_class = f"provider-{provider_name.lower().replace('_', '-')}"
            threshold_class = f"threshold-{threshold_name.lower()}"

            html_content += f"""
                            <tr>
                                <td><span class="provider-badge {provider_class}">{provider_name}</span></td>
                                <td><span class="threshold-badge {threshold_class}">{threshold_name}</span></td>
                                <td><span class="{'grade-correct' if metrics['volume_score'] > 0.5 else 'grade-incorrect' if metrics['volume_score'] < 0 else ''}">{metrics['volume_score']:.3f}</span></td>
                                <td><span class="{'grade-correct' if metrics['quality_score'] > 0.5 else 'grade-incorrect' if metrics['quality_score'] < 0 else ''}">{metrics['quality_score']:.3f}</span></td>
                                <td>{metrics['attempted_rate']:.1%}</td>
                                <td>{metrics['accuracy_given_attempted']:.1%}</td>
                                <td>{metrics['abstention_rate']:.1%}</td>
                                <td><span class="grade-incorrect">{metrics['overconfidence_penalty']}</span></td>
                                <td><span class="threshold-badge {threshold_class}">{strategy_assessment}</span></td>
                            </tr>"""

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    </div>

    <!-- JavaScript for Interactive Features -->
    <script>
        // Initialize DataTables for provider comparison
        $(document).ready(function() {
            $('#providerTable').DataTable({
                responsive: true,
                pageLength: 25,
                order: [[2, 'desc']], // Sort by Volume Score descending
                columnDefs: [
                    { targets: [2, 3], type: 'num' },
                    { targets: [4, 5, 6], render: $.fn.dataTable.render.percentBar('round', '#dbeafe', '#1e40af', '#f1f5f9', '#374151', 0, 'right') }
                ],
                language: {
                    search: "Search providers:",
                    lengthMenu: "Show _MENU_ entries"
                }
            });
        });
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
        description="Enhanced Multi-Provider RAG Benchmark with Confidence Thresholds"
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
        default=3,
        help="Maximum number of parallel workers (default: 3)"
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
    return parser.parse_args()


def main():
    """Run enhanced confidence threshold benchmark"""
    args = parse_args()

    print("🎯 ENHANCED MULTI-PROVIDER RAG BENCHMARK")
    print("=" * 70)
    print("Implementing OpenAI's 'Why Language Models Hallucinate' Framework")
    print("Volume Strategy vs Quality Strategy with Confidence Thresholds")
    print("=" * 70)
    print("Comparing:")
    print("  1. OpenAI Vanilla (no RAG - baseline) [gpt-4.1]")
    print("  2. OpenAI RAG (vector store file search) [gpt-4.1]")
    print("  3. CustomGPT (RAG with existing knowledge base) [gpt-4.1]")
    print("LLM-As-A-Judge: GPT-5-mini (87.5% cost reduction)")
    print("=" * 70)

    # Configuration
    if args.examples is not None:
        n_samples = args.examples
    elif args.debug:
        n_samples = 5
    else:
        n_samples = 10

    debug_mode = args.debug
    max_workers = args.max_workers
    output_dir = args.output_dir
    dry_run = args.dry_run

    if debug_mode:
        print("🐛 DEBUG MODE ENABLED")

    print(f"📊 Configuration:")
    print(f"   Samples per provider per threshold: {n_samples}")
    print(f"   Confidence thresholds: {len(CONFIDENCE_THRESHOLDS)}")
    print(f"   Max parallel workers: {max_workers}")
    print(f"   Output directory: {output_dir}")
    print(f"   Debug mode: {debug_mode}")

    # Show threshold details
    print(f"\n🎯 Confidence Threshold Framework:")
    for threshold in CONFIDENCE_THRESHOLDS:
        print(f"   {threshold.name}: t={threshold.threshold}, penalty={threshold.penalty_ratio}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize audit logging
    run_id = create_run_id()
    audit_logger = AuditLogger(run_id, output_dir)

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
        print(f"   Ready to run {len(samplers)} providers × {len(CONFIDENCE_THRESHOLDS)} thresholds")
        print(f"   Total evaluations: {len(samplers) * len(CONFIDENCE_THRESHOLDS)}")
        print(f"   Questions per evaluation: {n_samples}")
        print(f"   Total questions to process: {len(samplers) * len(CONFIDENCE_THRESHOLDS) * n_samples}")
        print(f"   Estimated runtime: ~{len(samplers) * len(CONFIDENCE_THRESHOLDS) * n_samples * 4 / max_workers / 60:.1f} minutes")
        print(f"   Audit logs will be saved to: {audit_logger.run_dir}")
        return 0

    # Run evaluations (parallel across providers, sequential across thresholds within each provider)
    results = run_parallel_confidence_evaluations(samplers, n_samples, audit_logger, max_workers)

    # CRITICAL: Validate complete evaluation coverage (900 total evaluations)
    expected_total_evaluations = len(samplers) * len(CONFIDENCE_THRESHOLDS) * n_samples
    actual_total_evaluations = 0

    print(f"\n🔍 EVALUATION COVERAGE VALIDATION:")
    print("=" * 70)

    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    # Count actual evaluations completed
    for result in successful_results:
        for threshold_name, metrics in result["threshold_results"].items():
            conversations_completed = metrics.get("conversations", 0)
            actual_total_evaluations += conversations_completed
            print(f"   ✅ {result['sampler_name']} - {threshold_name}: {conversations_completed} evaluations")

    # Report any failures that reduce evaluation count
    for result in failed_results:
        missed_evaluations = len(CONFIDENCE_THRESHOLDS) * n_samples
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
    print(f"\n📊 CONFIDENCE THRESHOLD RESULTS SUMMARY:")
    print("=" * 70)

    if successful_results:
        print(f"🏆 PROVIDER PERFORMANCE:")

        # Analyze each provider's performance across thresholds
        for result in successful_results:
            provider_name = result["sampler_name"]
            threshold_results = result["threshold_results"]

            print(f"\n   {provider_name}:")
            for threshold_name, metrics in threshold_results.items():
                volume_score = metrics['volume_score']
                quality_score = metrics['quality_score']
                attempted_rate = metrics['attempted_rate']
                print(f"      {threshold_name}: Volume={volume_score:.3f}, Quality={quality_score:.3f}, Attempted={attempted_rate:.1%}")

            # Identify best strategies for this provider
            if threshold_results:
                best_volume = max(threshold_results.items(), key=lambda x: x[1]['volume_score'])
                best_quality = max(threshold_results.items(), key=lambda x: x[1]['quality_score'])
                print(f"      → Best Volume: {best_volume[0]} ({best_volume[1]['volume_score']:.3f})")
                print(f"      → Best Quality: {best_quality[0]} ({best_quality[1]['quality_score']:.3f})")

    if failed_results:
        print(f"\n❌ FAILED EVALUATIONS:")
        for result in failed_results:
            print(f"   - {result['sampler_name']}: {result['error']}")

    # Finalize audit logging with results and validation
    audit_logger.finalize_run({
        "total_providers": len(samplers),
        "successful_evaluations": len(successful_results),
        "failed_evaluations": len(failed_results),
        "samples_per_provider": n_samples,
        "confidence_thresholds": len(CONFIDENCE_THRESHOLDS),
        "expected_total_evaluations": expected_total_evaluations,
        "actual_total_evaluations": actual_total_evaluations,
        "evaluation_coverage_complete": actual_total_evaluations >= expected_total_evaluations,
        "results": results
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
            # Convert threshold_results to the expected format
            provider_results[provider_name] = {}
            for threshold_name, metrics in result["threshold_results"].items():
                # Create mock EvalResult with metrics
                class MockEvalResult:
                    def __init__(self, metrics_dict):
                        self.metrics = metrics_dict

                provider_results[provider_name][threshold_name] = MockEvalResult(metrics)

        # Create evaluator instance for statistical analysis
        eval_instance = ConfidenceThresholdSimpleQAEval(num_examples=n_samples)
        statistical_analysis = eval_instance.analyze_statistical_significance(provider_results)

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

    # Generate comprehensive confidence threshold report
    print(f"\n📄 Generating confidence threshold report...")
    run_metadata = {
        "run_id": run_id,
        "samples_per_provider": n_samples,
        "max_workers": max_workers,
        "debug_mode": debug_mode,
        "confidence_thresholds": len(CONFIDENCE_THRESHOLDS),
        "expected_total_evaluations": expected_total_evaluations,
        "actual_total_evaluations": actual_total_evaluations,
        "evaluation_coverage_complete": actual_total_evaluations >= expected_total_evaluations,
        "statistical_analysis": statistical_analysis
    }

    report_file = generate_confidence_threshold_report(
        results,
        str(audit_logger.run_dir),
        run_metadata
    )
    print(f"   Confidence threshold report saved: {report_file}")

    # Save JSON results
    json_file = audit_logger.run_dir / "confidence_threshold_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "framework": "confidence_threshold_benchmark",
            "configuration": {
                "samples_per_provider": n_samples,
                "max_workers": max_workers,
                "debug_mode": debug_mode,
                "output_directory": output_dir,
                "confidence_thresholds": [
                    {
                        "name": ct.name,
                        "threshold": ct.threshold,
                        "penalty_ratio": ct.penalty_ratio,
                        "description": ct.description
                    } for ct in CONFIDENCE_THRESHOLDS
                ]
            },
            "audit_summary": audit_logger.get_run_summary(),
            "results": results,
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

    print(f"\n🎉 CONFIDENCE THRESHOLD BENCHMARK COMPLETE!")
    print(f"Report: {report_file}")
    print(f"Results: {json_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())