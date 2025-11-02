#!/usr/bin/env python3
"""
OpenAI Flex Service Tier Comparison Test

This script compares GPT-5 performance between standard and flex service tiers,
measuring latency, success rates, and response quality using sample queries.

Uses unified brand kit for consistent, Apple-inspired HTML reports.
"""

import csv
import json
import os
import sys
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for brand kit import
sys.path.insert(0, str(Path(__file__).parent.parent))
from brand_kit import (
    get_html_head,
    get_navigation_bar,
    get_page_header,
    format_timestamp
)


@dataclass
class TestResult:
    """Result of a single API call test"""
    service_tier: str
    query_id: int
    query_text: str
    success: bool
    latency_ms: Optional[float]
    response_length: Optional[int]
    error_type: Optional[str]
    error_message: Optional[str]
    response_text: Optional[str]
    timestamp: str
    timeout_occurred: bool = False
    retry_count: int = 0


@dataclass
class ComparisonStats:
    """Statistical comparison between service tiers"""
    service_tier: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    avg_response_length: float
    timeout_count: int
    resource_unavailable_count: int
    total_retry_count: int


class FlexTierComparison:
    def __init__(self,
                 model: str = "gpt-5",
                 timeout_seconds: int = 900,  # 15 minutes as recommended
                 max_retries: int = 3):
        self.client = OpenAI(timeout=timeout_seconds)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.results: List[TestResult] = []

        # Verify API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def load_sample_queries(self, csv_path: str, limit: Optional[int] = None) -> List[Tuple[int, str]]:
        """Load sample queries from CSV file"""
        queries = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if limit and i >= limit:
                        break
                    # Use the 'problem' field as the query
                    query_text = row.get('problem', '').strip()
                    if query_text:
                        queries.append((i, query_text))
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            # Fallback to default queries
            queries = [
                (0, "What is the capital of France and what is its population?"),
                (1, "Explain the theory of relativity in simple terms."),
                (2, "What are the main causes of climate change?"),
                (3, "How does photosynthesis work in plants?"),
                (4, "What is the history of the Internet?")
            ]

        return queries

    def make_api_call(self, query_text: str, service_tier: str, query_id: int) -> TestResult:
        """Make a single API call and measure performance"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                # Prepare the API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": query_text}
                    ],
                    "max_completion_tokens": 1000,
                }

                # Add service_tier parameter for flex tier
                if service_tier == "flex":
                    api_params["service_tier"] = "flex"

                # Make the API call
                response = self.client.chat.completions.create(**api_params)

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                response_text = response.choices[0].message.content or ""
                response_length = len(response_text)

                return TestResult(
                    service_tier=service_tier,
                    query_id=query_id,
                    query_text=query_text,
                    success=True,
                    latency_ms=latency_ms,
                    response_length=response_length,
                    error_type=None,
                    error_message=None,
                    response_text=response_text[:500] + "..." if len(response_text) > 500 else response_text,
                    timestamp=timestamp,
                    retry_count=retry_count
                )

            except openai.APITimeoutError as e:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                return TestResult(
                    service_tier=service_tier,
                    query_id=query_id,
                    query_text=query_text,
                    success=False,
                    latency_ms=latency_ms,
                    response_length=None,
                    error_type="TimeoutError",
                    error_message=str(e),
                    response_text=None,
                    timestamp=timestamp,
                    timeout_occurred=True,
                    retry_count=retry_count
                )

            except openai.RateLimitError as e:
                if "resource_unavailable" in str(e).lower() or "429" in str(e):
                    # Resource unavailable error for flex tier - retry with exponential backoff
                    if attempt < self.max_retries:
                        retry_count += 1
                        wait_time = 2 ** attempt
                        print(f"Resource unavailable for {service_tier}, retrying in {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        continue

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                return TestResult(
                    service_tier=service_tier,
                    query_id=query_id,
                    query_text=query_text,
                    success=False,
                    latency_ms=latency_ms,
                    response_length=None,
                    error_type="ResourceUnavailable" if "resource_unavailable" in str(e).lower() else "RateLimitError",
                    error_message=str(e),
                    response_text=None,
                    timestamp=timestamp,
                    retry_count=retry_count
                )

            except Exception as e:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                return TestResult(
                    service_tier=service_tier,
                    query_id=query_id,
                    query_text=query_text,
                    success=False,
                    latency_ms=latency_ms,
                    response_length=None,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    response_text=None,
                    timestamp=timestamp,
                    retry_count=retry_count
                )

    def run_comparison(self,
                      queries: List[Tuple[int, str]],
                      max_workers: int = 4,
                      randomize_order: bool = True) -> List[TestResult]:
        """Run comparison tests for both service tiers"""
        print(f"Running comparison with {len(queries)} queries per tier...")
        print(f"Model: {self.model}, Timeout: {self.timeout_seconds}s, Max workers: {max_workers}")

        # Create test tasks for both tiers
        tasks = []
        for query_id, query_text in queries:
            tasks.append(("standard", query_id, query_text))
            tasks.append(("flex", query_id, query_text))

        # Randomize order to avoid bias
        if randomize_order:
            import random
            random.shuffle(tasks)

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for service_tier, query_id, query_text in tasks:
                future = executor.submit(self.make_api_call, query_text, service_tier, query_id)
                future_to_task[future] = (service_tier, query_id, query_text)

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                service_tier, query_id, query_text = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    status = "✓" if result.success else "✗"
                    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
                    print(f"{status} [{completed}/{len(tasks)}] {service_tier.upper()} Q{query_id}: {latency}")

                except Exception as e:
                    print(f"✗ [{completed + 1}/{len(tasks)}] {service_tier.upper()} Q{query_id}: Error - {e}")
                    completed += 1

        return results

    def calculate_statistics(self, results: List[TestResult]) -> Dict[str, ComparisonStats]:
        """Calculate comparison statistics for each service tier"""
        stats_by_tier = {}

        for service_tier in ["standard", "flex"]:
            tier_results = [r for r in results if r.service_tier == service_tier]

            if not tier_results:
                continue

            successful_results = [r for r in tier_results if r.success and r.latency_ms is not None]

            # Calculate latency statistics
            if successful_results:
                latencies = [r.latency_ms for r in successful_results]
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

                response_lengths = [r.response_length for r in successful_results if r.response_length]
                avg_response_length = statistics.mean(response_lengths) if response_lengths else 0.0
            else:
                avg_latency = median_latency = min_latency = max_latency = std_latency = 0.0
                avg_response_length = 0.0

            # Count error types
            timeout_count = len([r for r in tier_results if r.timeout_occurred])
            resource_unavailable_count = len([r for r in tier_results if r.error_type == "ResourceUnavailable"])
            total_retry_count = sum(r.retry_count for r in tier_results)

            stats_by_tier[service_tier] = ComparisonStats(
                service_tier=service_tier,
                total_requests=len(tier_results),
                successful_requests=len(successful_results),
                failed_requests=len(tier_results) - len(successful_results),
                success_rate=len(successful_results) / len(tier_results) * 100,
                avg_latency_ms=avg_latency,
                median_latency_ms=median_latency,
                min_latency_ms=min_latency,
                max_latency_ms=max_latency,
                std_latency_ms=std_latency,
                avg_response_length=avg_response_length,
                timeout_count=timeout_count,
                resource_unavailable_count=resource_unavailable_count,
                total_retry_count=total_retry_count
            )

        return stats_by_tier

    def generate_report(self,
                       results: List[TestResult],
                       stats: Dict[str, ComparisonStats],
                       output_dir: str = "results") -> str:
        """Generate comprehensive comparison report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"flex_tier_comparison_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate HTML report
        html_content = self._generate_html_report(results, stats)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Also save raw data as JSON
        json_filename = f"flex_tier_comparison_data_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            data = {
                "metadata": {
                    "timestamp": timestamp,
                    "model": self.model,
                    "timeout_seconds": self.timeout_seconds,
                    "total_queries": len(results) // 2  # Divided by 2 since we test both tiers
                },
                "statistics": {tier: asdict(stat) for tier, stat in stats.items()},
                "raw_results": [asdict(result) for result in results]
            }
            json.dump(data, f, indent=2, default=str)

        return report_path

    def _generate_html_report(self, results: List[TestResult], stats: Dict[str, ComparisonStats]) -> str:
        """Generate HTML report content using brand kit"""

        # Calculate comparison metrics
        if "standard" in stats and "flex" in stats:
            standard_stats = stats["standard"]
            flex_stats = stats["flex"]

            latency_improvement = ((standard_stats.avg_latency_ms - flex_stats.avg_latency_ms)
                                 / standard_stats.avg_latency_ms * 100) if standard_stats.avg_latency_ms > 0 else 0
            success_rate_diff = flex_stats.success_rate - standard_stats.success_rate

            # Cost estimation (50% cheaper for flex according to docs)
            cost_savings = 50.0
        else:
            latency_improvement = success_rate_diff = cost_savings = 0
            standard_stats = flex_stats = None

        # Start HTML with brand kit
        html = get_html_head(
            title="OpenAI Flex vs Standard Service Tier Comparison",
            description="Performance comparison between GPT-5 Flex and Standard service tiers"
        )

        html += f"""
<body>
    {get_navigation_bar(active_page='quality', run_id='unknown')}

    <div class="main-container">
        {get_page_header(
            title="OpenAI Flex vs Standard Tier Comparison",
            subtitle=f"Model: {self.model} | Total Queries: {len(results) // 2}",
            meta_info=f"Generated: {format_timestamp()}"
        )}

        <div class="content-section">
            <!-- Summary Metrics -->
            <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{latency_improvement:+.1f}%</div>
            <div class="metric-label">Latency Change (Flex vs Standard)</div>
            <small>{"Flex is slower" if latency_improvement > 0 else "Flex is faster"}</small>
        </div>

        <div class="metric-card">
            <div class="metric-value">{success_rate_diff:+.1f}%</div>
            <div class="metric-label">Success Rate Difference</div>
            <small>{"Flex is more reliable" if success_rate_diff > 0 else "Standard is more reliable"}</small>
        </div>

        <div class="metric-card">
            <div class="metric-value">-{cost_savings:.0f}%</div>
            <div class="metric-label">Cost Savings (Flex)</div>
            <small>According to OpenAI pricing</small>
        </div>
    </div>

    <h2>Performance Statistics</h2>
    <table class="stats-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th class="tier-standard">Standard Tier</th>
                <th class="tier-flex">Flex Tier</th>
                <th>Difference</th>
            </tr>
        </thead>
        <tbody>
        """

        if standard_stats and flex_stats:
            html += f"""
            <tr>
                <td>Success Rate</td>
                <td>{standard_stats.success_rate:.1f}%</td>
                <td>{flex_stats.success_rate:.1f}%</td>
                <td>{success_rate_diff:+.1f}%</td>
            </tr>
            <tr>
                <td>Average Latency</td>
                <td>{standard_stats.avg_latency_ms:.0f}ms</td>
                <td>{flex_stats.avg_latency_ms:.0f}ms</td>
                <td>{flex_stats.avg_latency_ms - standard_stats.avg_latency_ms:+.0f}ms</td>
            </tr>
            <tr>
                <td>Median Latency</td>
                <td>{standard_stats.median_latency_ms:.0f}ms</td>
                <td>{flex_stats.median_latency_ms:.0f}ms</td>
                <td>{flex_stats.median_latency_ms - standard_stats.median_latency_ms:+.0f}ms</td>
            </tr>
            <tr>
                <td>Min Latency</td>
                <td>{standard_stats.min_latency_ms:.0f}ms</td>
                <td>{flex_stats.min_latency_ms:.0f}ms</td>
                <td>{flex_stats.min_latency_ms - standard_stats.min_latency_ms:+.0f}ms</td>
            </tr>
            <tr>
                <td>Max Latency</td>
                <td>{standard_stats.max_latency_ms:.0f}ms</td>
                <td>{flex_stats.max_latency_ms:.0f}ms</td>
                <td>{flex_stats.max_latency_ms - standard_stats.max_latency_ms:+.0f}ms</td>
            </tr>
            <tr>
                <td>Std Deviation</td>
                <td>{standard_stats.std_latency_ms:.0f}ms</td>
                <td>{flex_stats.std_latency_ms:.0f}ms</td>
                <td>{flex_stats.std_latency_ms - standard_stats.std_latency_ms:+.0f}ms</td>
            </tr>
            <tr>
                <td>Timeouts</td>
                <td>{standard_stats.timeout_count}</td>
                <td>{flex_stats.timeout_count}</td>
                <td>{flex_stats.timeout_count - standard_stats.timeout_count:+d}</td>
            </tr>
            <tr>
                <td>Resource Unavailable</td>
                <td>{standard_stats.resource_unavailable_count}</td>
                <td>{flex_stats.resource_unavailable_count}</td>
                <td>{flex_stats.resource_unavailable_count - standard_stats.resource_unavailable_count:+d}</td>
            </tr>
            <tr>
                <td>Total Retries</td>
                <td>{standard_stats.total_retry_count}</td>
                <td>{flex_stats.total_retry_count}</td>
                <td>{flex_stats.total_retry_count - standard_stats.total_retry_count:+d}</td>
            </tr>
            """

        html += """
        </tbody>
    </table>

    <div class="details">
        <h2>Key Findings</h2>
        <ul>
        """

        if standard_stats and flex_stats:
            if flex_stats.avg_latency_ms > standard_stats.avg_latency_ms:
                html += f"<li>Flex tier is <strong>{((flex_stats.avg_latency_ms - standard_stats.avg_latency_ms) / standard_stats.avg_latency_ms * 100):.1f}% slower</strong> on average</li>"
            else:
                html += f"<li>Flex tier is <strong>{((standard_stats.avg_latency_ms - flex_stats.avg_latency_ms) / standard_stats.avg_latency_ms * 100):.1f}% faster</strong> on average</li>"

            if flex_stats.success_rate > standard_stats.success_rate:
                html += f"<li>Flex tier has a <strong>{success_rate_diff:.1f}% higher success rate</strong></li>"
            elif flex_stats.success_rate < standard_stats.success_rate:
                html += f"<li>Standard tier has a <strong>{-success_rate_diff:.1f}% higher success rate</strong></li>"
            else:
                html += "<li>Both tiers have similar success rates</li>"

            if flex_stats.resource_unavailable_count > 0:
                html += f"<li>Flex tier experienced <strong>{flex_stats.resource_unavailable_count} resource unavailable errors</strong></li>"

            html += f"<li>Flex tier offers <strong>50% cost savings</strong> according to OpenAI pricing</li>"

        html += """
        </ul>

        <h2>Recommendations</h2>
        <ul>
        """

        if standard_stats and flex_stats:
            if flex_stats.resource_unavailable_count > 0:
                html += "<li>Implement exponential backoff retry logic when using Flex tier</li>"

            if flex_stats.avg_latency_ms > standard_stats.avg_latency_ms:
                html += "<li>Use Flex tier for non-time-sensitive batch processing to save costs</li>"
                html += "<li>Use Standard tier for real-time or interactive applications</li>"

            html += "<li>Monitor success rates and implement fallback to Standard tier for critical workloads</li>"
            html += "<li>Consider Flex tier for model evaluations, data enrichment, and offline analysis</li>"

        html += """
        </ul>
    </div>

    <div class="details">
        <h2>Individual Query Results</h2>
        <div class="query-results">
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Query ID</th>
                        <th>Service Tier</th>
                        <th>Status</th>
                        <th>Latency</th>
                        <th>Response Length</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
        """

        for result in sorted(results, key=lambda r: (r.query_id, r.service_tier)):
            status_class = "success" if result.success else "error"
            status_text = "Success" if result.success else "Failed"
            latency_text = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
            response_len = result.response_length if result.response_length else "N/A"
            error_text = result.error_type if result.error_type else ""

            html += f"""
                    <tr>
                        <td>Q{result.query_id}</td>
                        <td class="tier-{result.service_tier}">{result.service_tier.title()}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{latency_text}</td>
                        <td>{response_len}</td>
                        <td>{error_text}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>

            <!-- Footer -->
            <hr class="mt-5">
            <div class="text-center text-muted mb-4">
                <p>
                    <strong>Flex vs Standard Tier Comparison</strong> |
                    Generated: """ + format_timestamp() + """
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """

        return html


def main():
    parser = argparse.ArgumentParser(description="Compare OpenAI Flex vs Standard service tiers")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model to test (default: gpt-5)")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to test per tier (default: 20)")
    parser.add_argument("--csv-file", default="build-rag/simple_qa_test_set.csv",
                       help="CSV file with sample queries (default: build-rag/simple_qa_test_set.csv)")
    parser.add_argument("--timeout", type=int, default=900, help="API timeout in seconds (default: 900)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max concurrent requests (default: 4)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results (default: results)")
    parser.add_argument("--no-randomize", action="store_true", help="Don't randomize query order")

    args = parser.parse_args()

    # Initialize comparison tool
    comparison = FlexTierComparison(
        model=args.model,
        timeout_seconds=args.timeout
    )

    # Load sample queries
    print(f"Loading sample queries from {args.csv_file}...")
    queries = comparison.load_sample_queries(args.csv_file, limit=args.queries)
    print(f"Loaded {len(queries)} queries")

    # Run comparison
    results = comparison.run_comparison(
        queries=queries,
        max_workers=args.max_workers,
        randomize_order=not args.no_randomize
    )

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = comparison.calculate_statistics(results)

    # Generate report
    print("Generating report...")
    report_path = comparison.generate_report(results, stats, args.output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("FLEX VS STANDARD TIER COMPARISON SUMMARY")
    print(f"{'='*60}")

    if "standard" in stats and "flex" in stats:
        standard_stats = stats["standard"]
        flex_stats = stats["flex"]

        print(f"Model: {args.model}")
        print(f"Queries per tier: {len(queries)}")
        print(f"Total requests: {standard_stats.total_requests + flex_stats.total_requests}")
        print()

        print("SUCCESS RATES:")
        print(f"  Standard: {standard_stats.success_rate:.1f}% ({standard_stats.successful_requests}/{standard_stats.total_requests})")
        print(f"  Flex:     {flex_stats.success_rate:.1f}% ({flex_stats.successful_requests}/{flex_stats.total_requests})")
        print()

        print("AVERAGE LATENCY:")
        print(f"  Standard: {standard_stats.avg_latency_ms:.0f}ms")
        print(f"  Flex:     {flex_stats.avg_latency_ms:.0f}ms")
        print(f"  Difference: {flex_stats.avg_latency_ms - standard_stats.avg_latency_ms:+.0f}ms")
        print()

        print("ERROR ANALYSIS:")
        print(f"  Timeouts (Standard): {standard_stats.timeout_count}")
        print(f"  Timeouts (Flex):     {flex_stats.timeout_count}")
        print(f"  Resource Unavailable (Flex): {flex_stats.resource_unavailable_count}")
        print(f"  Total Retries (Flex):        {flex_stats.total_retry_count}")
        print()

        print("COST IMPLICATIONS:")
        print("  Flex tier offers 50% cost savings (per OpenAI pricing)")
        print(f"  Trade-off: {((flex_stats.avg_latency_ms - standard_stats.avg_latency_ms) / standard_stats.avg_latency_ms * 100):+.1f}% latency change")

    print(f"\nDetailed report saved to: {report_path}")
    print(f"Raw data saved to: {report_path.replace('.html', '_data.json')}")


if __name__ == "__main__":
    main()