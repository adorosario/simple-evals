#!/usr/bin/env python3
"""
RAG Benchmark: CustomGPT vs OpenAI GPT-4o
Compares RAG-enabled CustomGPT against standard OpenAI GPT-4o using SimpleQA evaluation.
"""

import argparse
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any
import common
from simpleqa_eval import SimpleQAEval
from sampler.chat_completion_sampler import ChatCompletionSampler, OPENAI_SYSTEM_MESSAGE_API
from sampler.customgpt_sampler import CustomGPTSampler


def create_samplers() -> Dict[str, Any]:
    """Create the two samplers for comparison."""
    samplers = {
        "openai_gpt4o": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=0.0  # Deterministic for fair comparison
        ),
        "customgpt_rag": CustomGPTSampler(
            model_name="gpt-3.5-turbo",  # Model backing CustomGPT
            temperature=0.0  # Deterministic for fair comparison
        )
    }
    return samplers


def run_side_by_side_evaluation(samplers: Dict[str, Any], num_examples: int = None, debug: bool = False) -> Dict[str, Any]:
    """Run evaluation on both models with the same questions."""

    # Create judge model (GPT-4o for grading)
    grader_sampler = ChatCompletionSampler(model="gpt-4o", temperature=0.0)

    # Create evaluation instance
    eval_instance = SimpleQAEval(
        grader_model=grader_sampler,
        num_examples=num_examples if num_examples else (5 if debug else None),
        n_repeats=1
    )

    print(f"Running evaluation on {len(eval_instance.examples)} questions...")
    print("Models being compared:")
    print("  - OpenAI GPT-4o (no RAG)")
    print("  - CustomGPT (with RAG)")
    print("  - Judge: GPT-4o")
    print("-" * 50)

    results = {}
    detailed_results = []

    for model_name, sampler in samplers.items():
        print(f"\nEvaluating {model_name}...")
        result = eval_instance(sampler)
        results[model_name] = {
            "score": result.score,
            "metrics": result.metrics,
            "eval_result": result
        }

    # Collect detailed per-question results for comparison
    openai_results = results["openai_gpt4o"]["eval_result"]
    customgpt_results = results["customgpt_rag"]["eval_result"]

    # Extract per-question details for side-by-side comparison
    for i, example in enumerate(eval_instance.examples):
        if i < len(openai_results.convos) and i < len(customgpt_results.convos):
            detailed_results.append({
                "question": example.get("problem", ""),
                "correct_answer": example.get("answer", ""),
                "openai_response": openai_results.convos[i][-1]["content"] if openai_results.convos[i] else "",
                "customgpt_response": customgpt_results.convos[i][-1]["content"] if customgpt_results.convos[i] else "",
                "openai_score": openai_results.convos[i] if i < len(openai_results.convos) else 0,
                "customgpt_score": customgpt_results.convos[i] if i < len(customgpt_results.convos) else 0,
            })

    return results, detailed_results


def calculate_comparative_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comparative metrics between the two models."""
    openai_metrics = results["openai_gpt4o"]["metrics"]
    customgpt_metrics = results["customgpt_rag"]["metrics"]

    comparison = {
        "openai_accuracy": openai_metrics.get("is_correct", 0),
        "customgpt_accuracy": customgpt_metrics.get("is_correct", 0),
        "accuracy_difference": customgpt_metrics.get("is_correct", 0) - openai_metrics.get("is_correct", 0),
        "openai_attempt_rate": openai_metrics.get("is_given_attempted", 0),
        "customgpt_attempt_rate": customgpt_metrics.get("is_given_attempted", 0),
        "attempt_rate_difference": customgpt_metrics.get("is_given_attempted", 0) - openai_metrics.get("is_given_attempted", 0),
        "openai_score": results["openai_gpt4o"]["score"],
        "customgpt_score": results["customgpt_rag"]["score"],
        "score_difference": results["customgpt_rag"]["score"] - results["openai_gpt4o"]["score"],
    }

    return comparison


def generate_comparative_report(results: Dict[str, Any], detailed_results: List[Dict], comparison_metrics: Dict[str, Any], output_dir: str):
    """Generate HTML report comparing both models."""

    # Create comparative HTML report
    comparative_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Benchmark: CustomGPT vs OpenAI GPT-4o</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metric-table th {{ background-color: #f2f2f2; }}
            .comparison-row {{ margin: 20px 0; padding: 10px; border: 1px solid #ccc; }}
            .better {{ color: green; font-weight: bold; }}
            .worse {{ color: red; font-weight: bold; }}
            .same {{ color: orange; font-weight: bold; }}
            .question-comparison {{ margin: 20px 0; padding: 15px; border: 1px solid #eee; }}
            .response-box {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>RAG Benchmark Results</h1>
        <h2>CustomGPT (RAG) vs OpenAI GPT-4o (No RAG)</h2>

        <h3>Summary Metrics</h3>
        <table class="metric-table">
            <tr><th>Metric</th><th>OpenAI GPT-4o</th><th>CustomGPT (RAG)</th><th>Difference</th></tr>
            <tr>
                <td>Accuracy</td>
                <td>{comparison_metrics['openai_accuracy']:.3f}</td>
                <td>{comparison_metrics['customgpt_accuracy']:.3f}</td>
                <td class="{'better' if comparison_metrics['accuracy_difference'] > 0 else 'worse' if comparison_metrics['accuracy_difference'] < 0 else 'same'}">
                    {comparison_metrics['accuracy_difference']:+.3f}
                </td>
            </tr>
            <tr>
                <td>Attempt Rate</td>
                <td>{comparison_metrics['openai_attempt_rate']:.3f}</td>
                <td>{comparison_metrics['customgpt_attempt_rate']:.3f}</td>
                <td class="{'better' if comparison_metrics['attempt_rate_difference'] > 0 else 'worse' if comparison_metrics['attempt_rate_difference'] < 0 else 'same'}">
                    {comparison_metrics['attempt_rate_difference']:+.3f}
                </td>
            </tr>
            <tr>
                <td>Overall Score</td>
                <td>{comparison_metrics['openai_score']:.3f}</td>
                <td>{comparison_metrics['customgpt_score']:.3f}</td>
                <td class="{'better' if comparison_metrics['score_difference'] > 0 else 'worse' if comparison_metrics['score_difference'] < 0 else 'same'}">
                    {comparison_metrics['score_difference']:+.3f}
                </td>
            </tr>
        </table>

        <h3>Analysis</h3>
        <div class="comparison-row">
            <strong>RAG Impact:</strong>
            {f"CustomGPT with RAG performs {'better' if comparison_metrics['accuracy_difference'] > 0.01 else 'worse' if comparison_metrics['accuracy_difference'] < -0.01 else 'similarly'} than OpenAI GPT-4o"}
            ({comparison_metrics['accuracy_difference']:+.1%} accuracy difference)
        </div>

        <h3>Detailed Individual Model Reports</h3>
        <h4>OpenAI GPT-4o Results</h4>
        {common.make_report(results['openai_gpt4o']['eval_result'])}

        <h4>CustomGPT RAG Results</h4>
        {common.make_report(results['customgpt_rag']['eval_result'])}

    </body>
    </html>
    """

    # Save comparative report
    report_path = os.path.join(output_dir, "rag_comparison_report.html")
    with open(report_path, "w") as f:
        f.write(comparative_html)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark CustomGPT (RAG) vs OpenAI GPT-4o on SimpleQA")
    parser.add_argument("--examples", type=int, help="Number of examples to evaluate (default: all)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with 5 examples")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir or f"rag_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RAG BENCHMARK: CustomGPT vs OpenAI GPT-4o")
    print("=" * 60)

    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "CUSTOMGPT_API_KEY", "CUSTOMGPT_PROJECT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        return

    try:
        # Create samplers
        samplers = create_samplers()

        # Run evaluations
        results, detailed_results = run_side_by_side_evaluation(
            samplers,
            num_examples=args.examples,
            debug=args.debug
        )

        # Calculate comparative metrics
        comparison_metrics = calculate_comparative_metrics(results)

        # Print results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"OpenAI GPT-4o Accuracy: {comparison_metrics['openai_accuracy']:.3f}")
        print(f"CustomGPT RAG Accuracy: {comparison_metrics['customgpt_accuracy']:.3f}")
        print(f"RAG Accuracy Gain: {comparison_metrics['accuracy_difference']:+.3f}")
        print(f"RAG Score Difference: {comparison_metrics['score_difference']:+.3f}")

        # Save detailed results
        results_file = os.path.join(output_dir, "rag_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "comparison_metrics": comparison_metrics,
                "openai_metrics": results["openai_gpt4o"]["metrics"],
                "customgpt_metrics": results["customgpt_rag"]["metrics"],
                "detailed_results": detailed_results[:10]  # Save first 10 for space
            }, f, indent=2)

        # Generate comparative report
        report_path = generate_comparative_report(results, detailed_results, comparison_metrics, output_dir)

        print(f"\nResults saved to: {output_dir}")
        print(f"Comparative report: {report_path}")
        print(f"Detailed metrics: {results_file}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()