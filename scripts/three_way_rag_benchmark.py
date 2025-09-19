#!/usr/bin/env python3
"""
Three-Way RAG Benchmark Script
Compares CustomGPT (RAG), OpenAI Vanilla (no RAG), and OpenAI RAG (vector store)
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sampler.customgpt_sampler import CustomGPTSampler
from sampler.openai_vanilla_sampler import OpenAIVanillaSampler
from sampler.openai_rag_sampler import OpenAIRAGSampler
from simpleqa_eval import SimpleQAEval


def setup_samplers():
    """Initialize all three samplers for comparison"""
    samplers = {}
    errors = []

    # 1. CustomGPT RAG Sampler
    print("🔧 Setting up CustomGPT (RAG) sampler...")
    try:
        # Set the required environment variable for CustomGPT
        os.environ["CUSTOMGPT_MODEL_NAME"] = "customgpt-rag"
        samplers["CustomGPT_RAG"] = CustomGPTSampler.from_env()
        print("   ✅ CustomGPT sampler ready")
    except Exception as e:
        error_msg = f"CustomGPT setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 2. OpenAI Vanilla Sampler (no RAG)
    print("🔧 Setting up OpenAI Vanilla (no RAG) sampler...")
    try:
        samplers["OpenAI_Vanilla"] = OpenAIVanillaSampler(
            model="gpt-4o-mini",
            system_message="You are a helpful assistant. Answer questions based on your training knowledge.",
            temperature=0.3
        )
        print("   ✅ OpenAI Vanilla sampler ready")
    except Exception as e:
        error_msg = f"OpenAI Vanilla setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    # 3. OpenAI RAG Sampler (with vector store)
    print("🔧 Setting up OpenAI RAG (vector store) sampler...")
    try:
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        if vector_store_id:
            samplers["OpenAI_RAG"] = OpenAIRAGSampler(
                model="gpt-4o-mini",
                system_message="You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.",
                temperature=0.3
            )
            print("   ✅ OpenAI RAG sampler ready")
        else:
            error_msg = "OpenAI RAG setup failed: OPENAI_VECTOR_STORE_ID not set"
            print(f"   ❌ {error_msg}")
            errors.append(error_msg)
    except Exception as e:
        error_msg = f"OpenAI RAG setup failed: {e}"
        print(f"   ❌ {error_msg}")
        errors.append(error_msg)

    return samplers, errors


def run_single_sampler_evaluation(sampler_name, sampler, n_samples=10):
    """Run evaluation for a single sampler"""
    print(f"\n🚀 Running evaluation for {sampler_name}...")
    print(f"   Samples: {n_samples}")

    try:
        # Create a grader model for evaluation (using same model for consistency)
        from sampler.chat_completion_sampler import ChatCompletionSampler
        grader = ChatCompletionSampler(model="gpt-4o-mini", temperature=0.0)

        # Create SimpleQA evaluation with grader
        eval_instance = SimpleQAEval(grader_model=grader, num_examples=n_samples)

        start_time = time.time()

        # Run evaluation
        result = eval_instance(sampler)

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


def create_comparison_report(results, output_dir):
    """Create a comprehensive comparison report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"three_way_rag_comparison_{timestamp}.html")

    # Sort results by score (highest first)
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    successful_results.sort(key=lambda x: x["score"] or 0, reverse=True)

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Three-Way RAG Benchmark Results</title>
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
        <h1>🏆 Three-Way RAG Benchmark Results</h1>
        <p>Comparison of CustomGPT (RAG) vs OpenAI Vanilla vs OpenAI RAG</p>
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


def main():
    """Run three-way RAG benchmark"""
    print("🚀 THREE-WAY RAG BENCHMARK")
    print("=" * 70)
    print("Comparing:")
    print("  1. CustomGPT (RAG with existing knowledge base)")
    print("  2. OpenAI Vanilla (no RAG - baseline)")
    print("  3. OpenAI RAG (vector store file search)")
    print("=" * 70)

    # Configuration
    n_samples = int(os.environ.get("BENCHMARK_SAMPLES", "10"))  # Default to 10 samples
    output_dir = "results"

    print(f"📊 Configuration:")
    print(f"   Samples per sampler: {n_samples}")
    print(f"   Output directory: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Setup samplers
    print(f"\n🔧 Setting up samplers...")
    samplers, setup_errors = setup_samplers()

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

    # Run evaluations
    print(f"\n🏃 Running evaluations...")
    results = []

    for sampler_name, sampler in samplers.items():
        result = run_single_sampler_evaluation(sampler_name, sampler, n_samples)
        results.append(result)

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

    # Generate report
    print(f"\n📄 Generating comparison report...")
    report_file = create_comparison_report(results, output_dir)
    print(f"   Report saved: {report_file}")

    # Save JSON results
    json_file = os.path.join(output_dir, f"three_way_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "samples_per_sampler": n_samples,
                "output_directory": output_dir
            },
            "results": results,
            "summary": {
                "total_samplers": len(results),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "winner": successful_results[0]["sampler_name"] if successful_results else None
            }
        }, f, indent=2)

    print(f"   JSON results saved: {json_file}")

    print(f"\n🎉 BENCHMARK COMPLETE!")
    if successful_results:
        print(f"Winner: {successful_results[0]['sampler_name']} with score {successful_results[0]['score']:.3f}")
    print(f"Report: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())