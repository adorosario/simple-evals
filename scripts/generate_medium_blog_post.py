#!/usr/bin/env python3
"""
Medium Blog Post Generator for Multi-Provider RAG Benchmark Results
Generates professional, unbiased data science articles based on benchmark runs
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI


class MediumBlogPostGenerator:
    """
    Generates professional Medium-style blog posts from benchmark results
    """

    def __init__(self, model: str = "o1-2024-12-17"):  # GPT-5 (o1) for highest quality
        self.client = OpenAI()
        self.model = model

    def load_benchmark_data(self, run_dir: str) -> Dict[str, Any]:
        """Load all benchmark data from a run directory"""
        run_path = Path(run_dir)

        # Load main results
        with open(run_path / "benchmark_results.json", 'r') as f:
            benchmark_data = json.load(f)

        # Load leaderboard data
        with open(run_path / "leaderboard.json", 'r') as f:
            leaderboard_data = json.load(f)

        # Load provider requests for timing analysis
        provider_requests = []
        provider_requests_file = run_path / "provider_requests.jsonl"
        if provider_requests_file.exists():
            with open(provider_requests_file, 'r') as f:
                provider_requests = [json.loads(line) for line in f]

        # Load judge evaluations for quality analysis
        judge_evaluations = []
        judge_evaluations_file = run_path / "judge_evaluations.jsonl"
        if judge_evaluations_file.exists():
            with open(judge_evaluations_file, 'r') as f:
                judge_evaluations = [json.loads(line) for line in f]

        # Load run metadata
        with open(run_path / "run_metadata.json", 'r') as f:
            metadata = json.load(f)

        return {
            "benchmark_results": benchmark_data,
            "leaderboard": leaderboard_data,
            "provider_requests": provider_requests,
            "judge_evaluations": judge_evaluations,
            "metadata": metadata,
            "run_dir": str(run_path)
        }

    def analyze_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights and statistics from the benchmark data"""

        # Performance analysis
        leaderboard = data["leaderboard"]["leaderboard"]

        performance_stats = {
            "total_providers": len(leaderboard),
            "total_questions": sum(p["samples"] for p in leaderboard),
            "best_performer": leaderboard[0] if leaderboard else None,
            "worst_performer": leaderboard[-1] if leaderboard else None,
            "score_range": {
                "highest": max(p["score"] for p in leaderboard) if leaderboard else 0,
                "lowest": min(p["score"] for p in leaderboard) if leaderboard else 0,
                "average": statistics.mean(p["score"] for p in leaderboard) if leaderboard else 0
            }
        }

        # Timing analysis by provider
        timing_stats = {}
        for provider_name in [p["provider"] for p in leaderboard]:
            provider_requests = [r for r in data["provider_requests"] if r["provider"] == provider_name]
            if provider_requests:
                latencies = [r["response"]["latency_ms"] for r in provider_requests]
                timing_stats[provider_name] = {
                    "avg_latency_ms": statistics.mean(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                    "total_requests": len(latencies),
                    "throughput_qps": 1000 / statistics.mean(latencies)
                }

        # Judge evaluation analysis
        judge_stats = {
            "total_evaluations": len(data["judge_evaluations"]),
            "avg_judge_latency": 0,
            "grade_distribution": {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
        }

        if data["judge_evaluations"]:
            judge_latencies = [j["judge"]["latency_ms"] for j in data["judge_evaluations"]]
            judge_stats["avg_judge_latency"] = statistics.mean(judge_latencies)

            # Count grades
            for eval_data in data["judge_evaluations"]:
                for grade in eval_data["grades"].values():
                    judge_stats["grade_distribution"][grade] = judge_stats["grade_distribution"].get(grade, 0) + 1

        # Cost analysis (rough estimates)
        cost_analysis = self._estimate_costs(timing_stats, performance_stats["total_questions"])

        return {
            "performance": performance_stats,
            "timing": timing_stats,
            "judge": judge_stats,
            "costs": cost_analysis,
            "metadata": data["metadata"]
        }

    def _estimate_costs(self, timing_stats: Dict, total_questions: int) -> Dict[str, Any]:
        """Estimate rough costs based on timing and provider type"""
        # Rough cost estimates (these would need to be updated with actual pricing)
        cost_estimates = {
            "OpenAI_Vanilla": {"cost_per_1k_tokens": 0.03, "avg_tokens_per_request": 100},
            "OpenAI_RAG": {"cost_per_1k_tokens": 0.06, "avg_tokens_per_request": 150},  # Higher due to RAG
            "CustomGPT_RAG": {"cost_per_1k_tokens": 0.05, "avg_tokens_per_request": 120}
        }

        estimated_costs = {}
        for provider, stats in timing_stats.items():
            if provider in cost_estimates:
                cost_info = cost_estimates[provider]
                total_tokens = stats["total_requests"] * cost_info["avg_tokens_per_request"]
                estimated_cost = (total_tokens / 1000) * cost_info["cost_per_1k_tokens"]
                estimated_costs[provider] = {
                    "estimated_total_cost": estimated_cost,
                    "cost_per_question": estimated_cost / stats["total_requests"] if stats["total_requests"] > 0 else 0
                }

        return estimated_costs

    def generate_blog_post(self, run_dir: str) -> str:
        """Generate a professional Medium blog post from benchmark results"""

        print("📊 Loading benchmark data...")
        data = self.load_benchmark_data(run_dir)

        print("🔍 Analyzing performance data...")
        analysis = self.analyze_performance_data(data)

        print("✍️ Generating blog post with GPT-5...")

        # Create comprehensive prompt for GPT-5
        prompt = self._create_blog_post_prompt(data, analysis)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a rigorous technical writer with academic standards writing for Medium and Data Science Central.

CRITICAL REQUIREMENTS:
- NEVER hallucinate or invent data points not provided in the user's data
- Cite ONLY the exact statistics provided in the prompt
- Use precise, factual language that would withstand peer review
- When you don't have specific data, explicitly state "data not provided" rather than estimate
- Focus on methodology transparency and reproducibility
- Write for technical practitioners while remaining accessible
- Include appropriate caveats and limitations
- Use evidence-based conclusions only

FORBIDDEN:
- Making claims not supported by the provided data
- Inventing benchmark comparisons with other studies
- Speculating beyond what the data shows
- Adding external claims or citations not in the data"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency and accuracy
                max_tokens=4000
            )

            blog_post = response.choices[0].message.content

            # Save the blog post
            output_file = Path(run_dir) / "medium_blog_post.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(blog_post)

            print(f"📝 Blog post generated: {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"❌ Error generating blog post: {e}")
            return ""

    def _create_blog_post_prompt(self, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for GPT-5 to generate the blog post"""

        # Extract key data points
        leaderboard = data["leaderboard"]["leaderboard"]
        best_performer = analysis["performance"]["best_performer"]
        timing_stats = analysis["timing"]
        judge_stats = analysis["judge"]

        prompt = f"""
Write a rigorous technical analysis of a RAG (Retrieval-Augmented Generation) benchmark study for Medium publication.

## EXACT STUDY PARAMETERS (cite these precisely)
- **Questions Evaluated**: {analysis['performance']['total_questions']} questions from SimpleQA dataset
- **Providers Under Test**: {analysis['performance']['total_providers']} systems
  - OpenAI gpt-4.1 (vanilla, no retrieval)
  - OpenAI gpt-4.1 + vector store RAG
  - CustomGPT RAG (gpt-4.1 backend)
- **Evaluation Method**: gpt-4.1 LLM-As-A-Judge with detailed reasoning
- **Execution**: Parallel evaluation with comprehensive audit logging
- **Reproducibility**: Complete request/response logs and judge explanations available

## Performance Results
{self._format_leaderboard_for_prompt(leaderboard)}

## Timing Analysis
{self._format_timing_for_prompt(timing_stats)}

## Judge Evaluation Statistics
- Total Evaluations: {judge_stats['total_evaluations']}
- Average Judge Latency: {judge_stats['avg_judge_latency']:.1f}ms
- Grade Distribution: {judge_stats['grade_distribution']}

## Key Insights to Address:
1. **Performance Gap Analysis**: Why do RAG systems outperform vanilla LLMs?
2. **Latency vs Accuracy Trade-offs**: How much slower are RAG systems and is it worth it?
3. **Provider Differentiation**: What explains the differences between OpenAI RAG and CustomGPT RAG?
4. **Cost-Benefit Analysis**: Performance improvements vs computational overhead
5. **Practical Implications**: When to choose each approach

## Blog Post Requirements:
- **Title**: Compelling, SEO-friendly title about RAG benchmark findings
- **Abstract/Introduction**: Hook readers with key findings
- **Methodology Section**: Transparent explanation of approach
- **Results Analysis**: Data-driven discussion of findings
- **Performance Deep Dive**: Statistical analysis with confidence intervals
- **Latency Analysis**: Speed vs accuracy trade-offs
- **Business Implications**: Practical guidance for practitioners
- **Limitations**: Honest discussion of study constraints
- **Conclusion**: Key takeaways and future directions

Write in markdown format suitable for Medium publication. Include suggested charts/visualizations where appropriate. Be analytical, unbiased, and focus on actionable insights for data science practitioners.

Target length: 2000-2500 words.
"""

        return prompt

    def _format_leaderboard_for_prompt(self, leaderboard: List[Dict]) -> str:
        """Format leaderboard data for the prompt"""
        formatted = "**Provider Performance Rankings:**\n"
        for i, provider in enumerate(leaderboard, 1):
            formatted += f"{i}. **{provider['provider']}**: {provider['score']:.1%} accuracy"
            formatted += f" (Grade: {provider['grade']}, Duration: {provider['duration_s']:.1f}s)\n"
        return formatted

    def _format_timing_for_prompt(self, timing_stats: Dict) -> str:
        """Format timing data for the prompt"""
        formatted = "**Provider Latency Analysis:**\n"
        for provider, stats in timing_stats.items():
            formatted += f"- **{provider}**: {stats['avg_latency_ms']:.0f}ms avg"
            formatted += f" ({stats['throughput_qps']:.2f} req/sec)\n"
        return formatted


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Generate Medium blog post from RAG benchmark results"
    )
    parser.add_argument(
        "run_dir",
        help="Path to the benchmark run directory containing results"
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="OpenAI model to use for generation (default: o1-2024-12-17 / GPT-5)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.run_dir):
        print(f"❌ Run directory not found: {args.run_dir}")
        return 1

    generator = MediumBlogPostGenerator(model=args.model)
    output_file = generator.generate_blog_post(args.run_dir)

    if output_file:
        print(f"✅ Blog post successfully generated!")
        print(f"📄 File: {output_file}")
        return 0
    else:
        print("❌ Failed to generate blog post")
        return 1


if __name__ == "__main__":
    sys.exit(main())