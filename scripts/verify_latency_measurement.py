#!/usr/bin/env python3
"""
Latency Measurement Verification Script

This script verifies that latency measurements are consistent and accurate
across all RAG providers by analyzing audit logs and benchmark results.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics


def load_audit_log(run_dir: Path) -> List[Dict]:
    """Load provider requests from audit log."""
    requests_file = run_dir / "provider_requests.jsonl"
    if not requests_file.exists():
        return []

    requests = []
    with open(requests_file, 'r') as f:
        for line in f:
            try:
                requests.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return requests


def load_benchmark_results(run_dir: Path) -> Optional[Dict]:
    """Load benchmark results JSON."""
    results_file = run_dir / "quality_benchmark_results.json"
    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def extract_latencies_from_audit(requests: List[Dict]) -> Dict[str, List[float]]:
    """Extract per-provider latencies from audit log."""
    latencies = {}
    for req in requests:
        provider = req.get("provider", "Unknown")
        latency = req.get("response", {}).get("latency_ms")
        if latency is not None:
            if provider not in latencies:
                latencies[provider] = []
            latencies[provider].append(latency)
    return latencies


def extract_latencies_from_results(results: Dict) -> Dict[str, Dict]:
    """Extract latency stats from benchmark results."""
    stats = {}
    for result in results.get("results", []):
        provider = result.get("sampler_name", "Unknown")
        metrics = result.get("metrics", {})
        stats[provider] = {
            "avg_ms": metrics.get("provider_latency_avg_ms"),
            "median_ms": metrics.get("provider_latency_median_ms"),
            "p95_ms": metrics.get("provider_latency_p95_ms"),
            "min_ms": metrics.get("provider_latency_min_ms"),
            "max_ms": metrics.get("provider_latency_max_ms"),
        }
    return stats


def verify_consistency(audit_latencies: Dict[str, List[float]],
                       reported_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    """Verify that audit log latencies match reported statistics."""
    results = {}

    for provider, latencies in audit_latencies.items():
        if not latencies:
            results[provider] = {"status": "NO_DATA", "message": "No latencies in audit log"}
            continue

        # Calculate stats from audit log
        calculated = {
            "avg_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        }
        if len(latencies) >= 20:
            sorted_lats = sorted(latencies)
            p95_idx = int(len(sorted_lats) * 0.95)
            calculated["p95_ms"] = sorted_lats[p95_idx]
        else:
            calculated["p95_ms"] = max(latencies)

        # Compare with reported stats
        reported = reported_stats.get(provider, {})

        discrepancies = []
        for metric, calc_val in calculated.items():
            rep_val = reported.get(metric)
            if rep_val is None:
                discrepancies.append(f"{metric}: no reported value")
            elif abs(calc_val - rep_val) > 1.0:  # Allow 1ms tolerance
                discrepancies.append(f"{metric}: calculated={calc_val:.2f}, reported={rep_val:.2f}")

        if discrepancies:
            results[provider] = {
                "status": "DISCREPANCY",
                "calculated": calculated,
                "reported": reported,
                "discrepancies": discrepancies
            }
        else:
            results[provider] = {
                "status": "PASS",
                "calculated": calculated,
                "reported": reported
            }

    return results


def analyze_outliers(latencies: Dict[str, List[float]], threshold_ms: float = 10000) -> Dict[str, Dict]:
    """Analyze outliers (requests > threshold)."""
    results = {}
    for provider, lats in latencies.items():
        outliers = [l for l in lats if l > threshold_ms]
        results[provider] = {
            "total_requests": len(lats),
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(lats) * 100 if lats else 0,
            "outlier_latencies": sorted(outliers, reverse=True)[:5] if outliers else []
        }
    return results


def analyze_token_correlation(requests: List[Dict]) -> Dict[str, Dict]:
    """Analyze correlation between token count and latency."""
    import math

    results = {}

    # Group by provider
    by_provider = {}
    for req in requests:
        provider = req.get("provider", "Unknown")
        latency = req.get("response", {}).get("latency_ms")
        token_usage = req.get("metadata", {}).get("token_usage", {})

        if latency and token_usage:
            if provider not in by_provider:
                by_provider[provider] = []

            total_tokens = token_usage.get("total_tokens", 0)
            if total_tokens > 0:
                by_provider[provider].append({
                    "latency": latency,
                    "tokens": total_tokens
                })

    # Calculate correlation for each provider
    for provider, data in by_provider.items():
        if len(data) < 5:
            results[provider] = {"correlation": None, "message": "Insufficient data"}
            continue

        latencies = [d["latency"] for d in data]
        tokens = [d["tokens"] for d in data]

        # Calculate Pearson correlation coefficient
        n = len(data)
        sum_x = sum(latencies)
        sum_y = sum(tokens)
        sum_xy = sum(l * t for l, t in zip(latencies, tokens))
        sum_x2 = sum(l * l for l in latencies)
        sum_y2 = sum(t * t for t in tokens)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

        if denominator > 0:
            correlation = numerator / denominator
        else:
            correlation = 0

        results[provider] = {
            "correlation": round(correlation, 3),
            "sample_size": n,
            "avg_latency": round(statistics.mean(latencies), 2),
            "avg_tokens": round(statistics.mean(tokens), 2)
        }

    return results


def print_report(run_dir: Path, consistency: Dict, outliers: Dict, correlation: Dict):
    """Print verification report."""
    print("=" * 80)
    print(f"LATENCY MEASUREMENT VERIFICATION REPORT")
    print(f"Run Directory: {run_dir.name}")
    print("=" * 80)
    print()

    # Consistency check
    print("1. CONSISTENCY CHECK (Audit Log vs Reported Stats)")
    print("-" * 60)
    all_pass = True
    for provider, result in consistency.items():
        status = result["status"]
        if status == "PASS":
            print(f"  [{provider}] PASS")
        elif status == "DISCREPANCY":
            all_pass = False
            print(f"  [{provider}] DISCREPANCY")
            for disc in result.get("discrepancies", []):
                print(f"    - {disc}")
        else:
            print(f"  [{provider}] {status}: {result.get('message', '')}")

    print()
    overall = "PASS" if all_pass else "FAIL"
    print(f"  Overall: {overall}")
    print()

    # Outlier analysis
    print("2. OUTLIER ANALYSIS (>10 seconds)")
    print("-" * 60)
    for provider, result in outliers.items():
        total = result["total_requests"]
        count = result["outlier_count"]
        pct = result["outlier_percentage"]
        print(f"  [{provider}] {count}/{total} requests ({pct:.1f}%)")
        if result["outlier_latencies"]:
            top3 = result["outlier_latencies"][:3]
            top3_str = ", ".join(f"{l/1000:.1f}s" for l in top3)
            print(f"    Top outliers: {top3_str}")
    print()

    # Token correlation
    print("3. TOKEN-LATENCY CORRELATION")
    print("-" * 60)
    for provider, result in correlation.items():
        corr = result.get("correlation")
        if corr is not None:
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
            print(f"  [{provider}] r={corr:.3f} ({strength})")
            print(f"    Avg latency: {result['avg_latency']:.0f}ms, Avg tokens: {result['avg_tokens']:.0f}")
        else:
            print(f"  [{provider}] {result.get('message', 'No correlation data')}")
    print()

    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


def main():
    # Find results directory
    results_dir = Path(__file__).parent.parent / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find most recent run
    runs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                  reverse=True)

    if not runs:
        print("Error: No benchmark runs found")
        sys.exit(1)

    # Allow specifying run as argument
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
        run_dir = results_dir / run_name
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    else:
        run_dir = runs[0]
        print(f"Using most recent run: {run_dir.name}")
        print()

    # Load data
    audit_log = load_audit_log(run_dir)
    benchmark_results = load_benchmark_results(run_dir)

    if not audit_log:
        print("Error: No audit log data found")
        sys.exit(1)

    if not benchmark_results:
        print("Warning: No benchmark results found, using audit log only")

    # Extract latencies
    audit_latencies = extract_latencies_from_audit(audit_log)
    reported_stats = extract_latencies_from_results(benchmark_results) if benchmark_results else {}

    # Verify consistency
    consistency = verify_consistency(audit_latencies, reported_stats)

    # Analyze outliers
    outliers = analyze_outliers(audit_latencies)

    # Analyze token correlation
    correlation = analyze_token_correlation(audit_log)

    # Print report
    print_report(run_dir, consistency, outliers, correlation)


if __name__ == "__main__":
    main()
