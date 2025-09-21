#!/usr/bin/env python3
"""
Multi-Provider RAG Benchmark Script
Compares multiple providers: OpenAI Vanilla, OpenAI RAG, CustomGPT RAG
Extensible architecture for adding new RAG providers (Ragie, Pinecone, etc.)
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
from audited_simpleqa_eval import AuditedSimpleQAEval
from audit_logger import AuditLogger, create_run_id
from leaderboard_generator import LeaderboardGenerator


def setup_samplers(audit_logger=None):
    """Initialize all providers for comparison with audit logging"""
    samplers = {}
    errors = []

    # 1. CustomGPT RAG Sampler (with GPT-4.1)
    print("🔧 Setting up CustomGPT (RAG) sampler...")
    try:
        # Set the required environment variable for CustomGPT
        os.environ["CUSTOMGPT_MODEL_NAME"] = "gpt-4.1"  # Updated to use GPT-4.1
        sampler = AuditedCustomGPTSampler.from_env(audit_logger=audit_logger)
        samplers["CustomGPT_RAG"] = sampler

        # Register with audit logger
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
            model="gpt-4.1",  # Updated to use GPT-4.1
            system_message="You are a helpful assistant. Answer questions based on your training knowledge.",
            temperature=0.3,
            audit_logger=audit_logger
        )
        samplers["OpenAI_Vanilla"] = sampler

        # Register with audit logger
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
                model="gpt-4.1",  # Updated to use GPT-4.1
                system_message="You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.",
                temperature=0.3,
                audit_logger=audit_logger
            )
            samplers["OpenAI_RAG"] = sampler

            # Register with audit logger
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


def run_single_sampler_evaluation(sampler_name, sampler, n_samples=10, audit_logger=None):
    """Run evaluation for a single sampler with audit logging"""
    print(f"\n🚀 Running evaluation for {sampler_name}...")
    print(f"   Samples: {n_samples}")

    try:
        # Create SimpleQA evaluation with audited grader (GPT-4.1)
        eval_instance = AuditedSimpleQAEval(
            grader_model=None,  # Uses GPT-4.1 by default
            num_examples=n_samples,
            audit_logger=audit_logger
        )

        start_time = time.time()

        # Run evaluation
        result = eval_instance(sampler, provider_name=sampler_name)

        end_time = time.time()
        duration = end_time - start_time

        print(f"   ✅ Completed in {duration:.1f}s")
        print(f"   Score: {result.score}")
        print(f"   Conversations: {len(result.convos)}")

        return {
            "sampler_name": sampler_name,
            "score": result.score,
            "metrics": result.metrics,
            "duration": duration,
            "samples_evaluated": n_samples,
            "conversations": len(result.convos),
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")

        # Log the error
        if audit_logger:
            audit_logger.log_error(
                component=f"{sampler_name}_evaluation",
                error=str(e),
                context={"samples": n_samples}
            )

        return {
            "sampler_name": sampler_name,
            "score": None,
            "metrics": None,
            "duration": None,
            "samples_evaluated": n_samples,
            "conversations": 0,
            "success": False,
            "error": str(e)
        }


def create_comparison_report(results, output_dir, audit_logger=None):
    """Legacy comparison report - replaced by LeaderboardGenerator"""
    # This function is now deprecated in favor of LeaderboardGenerator
    # Keeping for backward compatibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"legacy_comparison_{timestamp}.html")

    # Sort results by score (highest first)
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    successful_results.sort(key=lambda x: x["score"] or 0, reverse=True)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Provider RAG Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .results {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .sampler-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; }}
        .sampler-card.winner {{ border-color: #4CAF50; background: #f9fff9; }}
        .sampler-card.failed {{ border-color: #f44336; background: #fff9f9; }}
        .score {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric {{ margin: 10px 0; }}
        .error {{ color: #f44336; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 Multi-Provider RAG Benchmark Results</h1>
        <p>Comparison of OpenAI Vanilla vs OpenAI RAG vs CustomGPT RAG (All using GPT-4.1)</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <p><strong>Samplers tested:</strong> {len(results)}</p>
        <p><strong>Successful evaluations:</strong> {len(successful_results)}</p>
        <p><strong>Failed evaluations:</strong> {len(failed_results)}</p>
        {f'<p><strong>Winner:</strong> {successful_results[0]["sampler_name"]} (Score: {successful_results[0]["score"]:.3f})</p>' if successful_results else ''}
    </div>

    <div class="results">
"""

    # Add successful results
    for i, result in enumerate(successful_results):
        is_winner = i == 0
        card_class = "sampler-card winner" if is_winner else "sampler-card"

        html_content += f"""
        <div class="{card_class}">
            <h3>{result["sampler_name"]} {'🏆' if is_winner else ''}</h3>
            <div class="score">{result["score"]:.3f}</div>
            <div class="metric"><strong>Duration:</strong> {result["duration"]:.1f}s</div>
            <div class="metric"><strong>Samples:</strong> {result["samples_evaluated"]}</div>
            <div class="metric"><strong>Conversations:</strong> {result["conversations"]}</div>
            {f'<div class="metric"><strong>Metrics:</strong> {json.dumps(result["metrics"], indent=2)}</div>' if result["metrics"] else ''}
        </div>
"""

    # Add failed results
    for result in failed_results:
        html_content += f"""
        <div class="sampler-card failed">
            <h3>{result["sampler_name"]} ❌</h3>
            <div class="error">Evaluation Failed</div>
            <div class="metric"><strong>Error:</strong> {result["error"]}</div>
        </div>
"""

    # Add detailed comparison table
    if len(successful_results) > 1:
        html_content += """
    </div>

    <div class="comparison">
        <h2>📈 Detailed Comparison</h2>
        <table>
            <tr>
                <th>Sampler</th>
                <th>Score</th>
                <th>Duration (s)</th>
                <th>Samples</th>
                <th>Performance</th>
            </tr>
"""

        for result in successful_results:
            samples_per_sec = result["samples_evaluated"] / result["duration"] if result["duration"] > 0 else 0
            html_content += f"""
            <tr>
                <td>{result["sampler_name"]}</td>
                <td>{result["score"]:.3f}</td>
                <td>{result["duration"]:.1f}</td>
                <td>{result["samples_evaluated"]}</td>
                <td>{samples_per_sec:.2f} samples/sec</td>
            </tr>
"""

        html_content += """
        </table>
    </div>
"""

    html_content += """
</body>
</html>
"""

    # Write report
    with open(report_file, 'w') as f:
        f.write(html_content)

    return report_file


def run_parallel_evaluations(samplers: Dict[str, Any], n_samples: int, audit_logger: AuditLogger, max_workers: int = 3) -> List[Dict[str, Any]]:
    """Run evaluations in parallel across providers"""
    print(f"\n🏃 Running parallel evaluations with {max_workers} workers...")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_sampler = {
            executor.submit(
                run_single_sampler_evaluation,
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
                print(f"   ✅ {sampler_name}: {result['score']:.3f}" if result['success'] else f"   ❌ {sampler_name}: Failed")
            except Exception as e:
                print(f"   ❌ {sampler_name}: Exception - {e}")
                results.append({
                    "sampler_name": sampler_name,
                    "success": False,
                    "error": str(e)
                })

    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Provider RAG Benchmark - Compare OpenAI, CustomGPT, and other RAG providers"
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
    """Run multi-provider RAG benchmark"""
    args = parse_args()

    print("🚀 MULTI-PROVIDER RAG BENCHMARK")
    print("=" * 70)
    print("Comparing:")
    print("  1. OpenAI Vanilla (no RAG - baseline) [gpt-4.1]")
    print("  2. OpenAI RAG (vector store file search) [gpt-4.1]")
    print("  3. CustomGPT (RAG with existing knowledge base) [gpt-4.1]")
    print("  [Extensible for future providers: Ragie, Pinecone Assistant, etc.]")
    print("=" * 70)

    # Configuration
    if args.examples is not None:
        n_samples = args.examples
    elif args.debug:
        n_samples = 5  # Debug mode default
    else:
        n_samples = 10  # Normal mode default

    debug_mode = args.debug
    max_workers = args.max_workers
    output_dir = args.output_dir
    dry_run = args.dry_run

    if debug_mode:
        print("🐛 DEBUG MODE ENABLED")

    print(f"📊 Configuration:")
    print(f"   Samples per sampler: {n_samples}")
    print(f"   Max parallel workers: {max_workers}")
    print(f"   Output directory: {output_dir}")
    print(f"   Debug mode: {debug_mode}")

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
        print(f"   Ready to run {len(samplers)} providers with {n_samples} samples each")
        print(f"   Total questions to process: {len(samplers) * n_samples}")
        print(f"   Estimated runtime: ~{len(samplers) * n_samples * 3 / max_workers / 60:.1f} minutes")
        print(f"   Audit logs will be saved to: {audit_logger.run_dir}")
        return 0

    # Run evaluations (parallel across providers)
    results = run_parallel_evaluations(samplers, n_samples, audit_logger, max_workers)

    # Create results summary
    print(f"\n📊 RESULTS SUMMARY:")
    print("=" * 70)

    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]

    if successful_results:
        # Sort by score
        successful_results.sort(key=lambda x: x["score"], reverse=True)

        print(f"🏆 RANKINGS:")
        for i, result in enumerate(successful_results, 1):
            print(f"   {i}. {result['sampler_name']}: {result['score']:.3f} ({result['duration']:.1f}s)")

        # Winner analysis
        winner = successful_results[0]
        print(f"\n🥇 WINNER: {winner['sampler_name']}")
        print(f"   Score: {winner['score']:.3f}")
        print(f"   Duration: {winner['duration']:.1f}s")

        if len(successful_results) > 1:
            runner_up = successful_results[1]
            improvement = (winner['score'] - runner_up['score']) / runner_up['score'] * 100
            print(f"   Improvement over runner-up: {improvement:.1f}%")

    if failed_results:
        print(f"\n❌ FAILED EVALUATIONS:")
        for result in failed_results:
            print(f"   - {result['sampler_name']}: {result['error']}")

    # Finalize audit logging with results
    audit_logger.finalize_run({
        "total_providers": len(samplers),
        "successful_evaluations": len([r for r in results if r.get("success", False)]),
        "failed_evaluations": len([r for r in results if not r.get("success", True)]),
        "samples_per_provider": n_samples,
        "results": results
    })

    # Generate comprehensive leaderboard report
    print(f"\n📄 Generating leaderboard report...")
    leaderboard_gen = LeaderboardGenerator(audit_logger)

    run_metadata = {
        "run_id": run_id,
        "samples_per_provider": n_samples,
        "max_workers": max_workers,
        "debug_mode": debug_mode
    }

    report_file = leaderboard_gen.generate_comprehensive_report(
        results,
        str(audit_logger.run_dir),
        run_metadata
    )
    print(f"   Leaderboard report saved: {report_file}")

    # Generate JSON leaderboard
    json_leaderboard_file = audit_logger.run_dir / "leaderboard.json"
    leaderboard_gen.generate_json_leaderboard(results, str(json_leaderboard_file))
    print(f"   JSON leaderboard saved: {json_leaderboard_file}")

    # Save JSON results in the audit run directory
    json_file = audit_logger.run_dir / "benchmark_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "samples_per_sampler": n_samples,
                "max_workers": max_workers,
                "debug_mode": debug_mode,
                "output_directory": output_dir
            },
            "audit_summary": audit_logger.get_run_summary(),
            "results": results,
            "summary": {
                "total_samplers": len(results),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "winner": successful_results[0]["sampler_name"] if successful_results else None
            }
        }, f, indent=2)

    print(f"   JSON results saved: {json_file}")
    print(f"\n📋 Complete audit trail available in: {audit_logger.run_dir}")

    # Generate Medium blog post
    print(f"\n📝 Generating Medium blog post...")
    try:
        from generate_medium_blog_post import MediumBlogPostGenerator
        blog_generator = MediumBlogPostGenerator()
        blog_file = blog_generator.generate_blog_post(str(audit_logger.run_dir))
        if blog_file:
            print(f"   Blog post saved: {blog_file}")
        else:
            print(f"   ⚠️ Blog post generation failed")
    except Exception as e:
        print(f"   ⚠️ Blog post generation error: {e}")

    print(f"\n🎉 BENCHMARK COMPLETE!")
    if successful_results:
        print(f"Winner: {successful_results[0]['sampler_name']} with score {successful_results[0]['score']:.3f}")
    print(f"Report: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())