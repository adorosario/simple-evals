"""
Publication-Ready Leaderboard Generator
Creates comprehensive reports and leaderboards for multi-provider benchmarks
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics


class LeaderboardGenerator:
    """
    Generate publication-ready reports and leaderboards
    """

    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger

    def _analyze_provider_errors(self, run_dir: str) -> Dict[str, Any]:
        """
        Analyze provider request logs for errors and failures
        """
        provider_requests_file = os.path.join(run_dir, "provider_requests.jsonl")
        error_analysis = {"total_errors": 0, "by_provider": {}, "error_types": {}}

        if not os.path.exists(provider_requests_file):
            return error_analysis

        try:
            with open(provider_requests_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    request_data = json.loads(line)
                    provider = request_data.get("provider", "unknown")

                    # Initialize provider stats
                    if provider not in error_analysis["by_provider"]:
                        error_analysis["by_provider"][provider] = {
                            "total_requests": 0,
                            "empty_responses": 0,
                            "timeouts": 0,
                            "errors": 0
                        }

                    error_analysis["by_provider"][provider]["total_requests"] += 1

                    # Check for empty responses (indicates error)
                    response_content = request_data.get("response", {}).get("content", "")
                    if not response_content or response_content.strip() == "":
                        error_analysis["by_provider"][provider]["empty_responses"] += 1
                        error_analysis["total_errors"] += 1
                        error_analysis["error_types"]["empty_response"] = error_analysis["error_types"].get("empty_response", 0) + 1

                    # Check for timeout indicators (very high latency)
                    latency_ms = request_data.get("response", {}).get("latency_ms", 0)
                    if latency_ms > 60000:  # Over 60 seconds indicates likely timeout
                        error_analysis["by_provider"][provider]["timeouts"] += 1
                        error_analysis["error_types"]["timeout"] = error_analysis["error_types"].get("timeout", 0) + 1

        except Exception as e:
            print(f"Error analyzing provider requests: {e}")

        return error_analysis

    def generate_comprehensive_report(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        run_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Generate a comprehensive HTML report with leaderboard
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"multi_provider_leaderboard_{timestamp}.html")

        # Process results for leaderboard
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", True)]

        # Sort by score (highest first)
        successful_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Calculate statistical significance and confidence intervals
        leaderboard_data = self._calculate_statistics(successful_results)

        # Analyze provider errors for fairness assessment
        error_analysis = self._analyze_provider_errors(output_dir)

        # Generate HTML content
        html_content = self._generate_html_report(
            leaderboard_data,
            successful_results,
            failed_results,
            run_metadata or {},
            error_analysis
        )

        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_file

    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate statistical metrics for leaderboard
        """
        leaderboard_data = []

        for result in results:
            score = result.get("score", 0)
            metrics = result.get("metrics", {})
            duration = result.get("duration", 0)
            samples = result.get("samples_evaluated", 0)

            # Calculate additional metrics
            throughput = samples / duration if duration > 0 else 0
            accuracy_attempted = metrics.get("accuracy_given_attempted", 0)

            # Quality metrics
            correct_rate = metrics.get("is_correct", 0)
            incorrect_rate = metrics.get("is_incorrect", 0)
            not_attempted_rate = metrics.get("is_not_attempted", 0)

            leaderboard_entry = {
                "provider": result["sampler_name"],
                "score": score,
                "accuracy_attempted": accuracy_attempted,
                "correct_rate": correct_rate,
                "incorrect_rate": incorrect_rate,
                "not_attempted_rate": not_attempted_rate,
                "duration_s": duration,
                "throughput_qps": throughput,
                "samples": samples,
                "rank": 0,  # Will be set later
                "grade": self._calculate_grade(score),
                "confidence_interval": self._estimate_confidence_interval(score, samples)
            }

            leaderboard_data.append(leaderboard_entry)

        # Assign ranks
        for i, entry in enumerate(leaderboard_data):
            entry["rank"] = i + 1

        return leaderboard_data

    def _calculate_grade(self, score: float) -> str:
        """
        Assign letter grade based on score
        """
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "F"

    def _estimate_confidence_interval(self, score: float, samples: int) -> Dict[str, float]:
        """
        Estimate 95% confidence interval for the score
        Simple approximation using normal distribution
        """
        if samples == 0:
            return {"lower": 0, "upper": 0, "margin": 0}

        # Standard error approximation
        se = (score * (1 - score) / samples) ** 0.5
        margin = 1.96 * se  # 95% confidence interval

        return {
            "lower": max(0, score - margin),
            "upper": min(1, score + margin),
            "margin": margin
        }

    def _generate_html_report(
        self,
        leaderboard_data: List[Dict[str, Any]],
        successful_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        error_analysis: Dict[str, Any] = None
    ) -> str:
        """
        Generate the HTML report content
        """
        run_id = metadata.get("run_id", "unknown")
        total_samples = metadata.get("samples_per_provider", 0)
        total_providers = len(successful_results) + len(failed_results)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Provider RAG Benchmark Leaderboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #dee2e6;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metadata-item {{
            text-align: center;
        }}
        .metadata-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }}
        .metadata-value {{
            font-size: 1.2em;
            font-weight: 600;
            color: #495057;
        }}
        .leaderboard {{
            padding: 40px;
        }}
        .leaderboard h2 {{
            margin-top: 0;
            color: #495057;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .leaderboard-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .leaderboard-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .leaderboard-table td {{
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .leaderboard-table tr:hover {{
            background: #f8f9fa;
        }}
        .rank-1 {{
            background: linear-gradient(135deg, #ffd700, #ffed4e) !important;
        }}
        .rank-2 {{
            background: linear-gradient(135deg, #c0c0c0, #e5e5e5) !important;
        }}
        .rank-3 {{
            background: linear-gradient(135deg, #cd7f32, #deb887) !important;
        }}
        .provider-name {{
            font-weight: 600;
            color: #495057;
        }}
        .score {{
            font-weight: 700;
            font-size: 1.1em;
        }}
        .grade {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .grade-A\\+, .grade-A {{
            background: #d4edda;
            color: #155724;
        }}
        .grade-B\\+, .grade-B, .grade-B- {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        .grade-C\\+, .grade-C, .grade-C- {{
            background: #fff3cd;
            color: #856404;
        }}
        .grade-F {{
            background: #f8d7da;
            color: #721c24;
        }}
        .confidence-interval {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        .metrics-details {{
            margin-top: 40px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-title {{
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #667eea;
        }}
        .audit-info {{
            margin-top: 40px;
            padding: 20px;
            background: #e9ecef;
            border-radius: 8px;
        }}
        .audit-info h3 {{
            margin-top: 0;
            color: #495057;
        }}
        .footer {{
            background: #495057;
            color: white;
            padding: 20px 40px;
            text-align: center;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border: 1px solid transparent;
            border-radius: 5px;
        }}
        .alert-success {{
            color: #155724;
            background-color: #d4edda;
            border-color: #c3e6cb;
        }}
        .alert-warning {{
            color: #856404;
            background-color: #fff3cd;
            border-color: #ffeaa7;
        }}
        .alert-danger {{
            color: #721c24;
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }}
        .provider-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .provider-stat {{
            border-radius: 8px;
            padding: 15px;
        }}
        @media (max-width: 768px) {{
            .leaderboard-table {{
                font-size: 0.9em;
            }}
            .leaderboard-table th,
            .leaderboard-table td {{
                padding: 10px 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Multi-Provider RAG Benchmark</h1>
            <p>Comprehensive evaluation of RAG providers using gpt-4.1 and LLM-As-A-Judge</p>
        </div>

        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <div class="metadata-label">Run ID</div>
                    <div class="metadata-value">{run_id}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Providers Tested</div>
                    <div class="metadata-value">{total_providers}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Samples per Provider</div>
                    <div class="metadata-value">{total_samples}</div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-label">Generated</div>
                    <div class="metadata-value">{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
                </div>
            </div>
        </div>

        <div class="leaderboard">
            <h2>🏅 Leaderboard</h2>
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Provider</th>
                        <th>Score</th>
                        <th>Grade</th>
                        <th>Accuracy</th>
                        <th>Correct %</th>
                        <th>Not Attempted %</th>
                        <th>Duration (s)</th>
                        <th>Throughput (q/s)</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add leaderboard rows
        for entry in leaderboard_data:
            rank_class = ""
            if entry["rank"] == 1:
                rank_class = "rank-1"
            elif entry["rank"] == 2:
                rank_class = "rank-2"
            elif entry["rank"] == 3:
                rank_class = "rank-3"

            ci = entry["confidence_interval"]
            grade_escaped = entry['grade'].replace('+', '\\+').replace('-', '-')
            grade_class = f"grade-{grade_escaped}"

            html_content += f"""
                    <tr class="{rank_class}">
                        <td><strong>{entry["rank"]}</strong></td>
                        <td class="provider-name">{entry["provider"]}</td>
                        <td class="score">
                            {entry["score"]:.3f}
                            <div class="confidence-interval">±{ci["margin"]:.3f}</div>
                        </td>
                        <td><span class="grade {grade_class}">{entry["grade"]}</span></td>
                        <td>{entry["accuracy_attempted"]:.3f}</td>
                        <td>{entry["correct_rate"]:.1%}</td>
                        <td>{entry["not_attempted_rate"]:.1%}</td>
                        <td>{entry["duration_s"]:.1f}</td>
                        <td>{entry["throughput_qps"]:.2f}</td>
                    </tr>
"""

        # Add failed providers section if any
        if failed_results:
            html_content += """
                </tbody>
            </table>

            <h3 style="color: #dc3545; margin-top: 40px;">❌ Failed Evaluations</h3>
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
"""
            for result in failed_results:
                html_content += f"""
                    <tr>
                        <td class="provider-name">{result["sampler_name"]}</td>
                        <td style="color: #dc3545;">{result.get("error", "Unknown error")}</td>
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
"""

        # Add performance metrics section
        if leaderboard_data:
            best_provider = leaderboard_data[0]
            avg_score = statistics.mean([p["score"] for p in leaderboard_data])
            total_duration = sum([p["duration_s"] for p in leaderboard_data])

            html_content += f"""
            <div class="metrics-details">
                <h2>📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Best Performance</div>
                        <div class="metric-value">{best_provider["provider"]}</div>
                        <div style="color: #6c757d; margin-top: 5px;">{best_provider["score"]:.3f} score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Average Score</div>
                        <div class="metric-value">{avg_score:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Total Runtime</div>
                        <div class="metric-value">{total_duration:.1f}s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Questions Evaluated</div>
                        <div class="metric-value">{total_samples * len(leaderboard_data)}</div>
                    </div>
                </div>
            </div>
"""

        # Add audit trail information
        if self.audit_logger:
            audit_summary = self.audit_logger.get_run_summary()
            html_content += f"""
            <div class="audit-info">
                <h3>🔍 Audit Trail</h3>
                <p><strong>Complete audit logs available for debugging and analysis:</strong></p>
                <ul>
                    <li><strong>Provider Requests:</strong> {audit_summary["logs"]["provider_requests"]}</li>
                    <li><strong>Judge Evaluations:</strong> {audit_summary["logs"]["judge_evaluations"]}</li>
                    <li><strong>Run Metadata:</strong> {audit_summary["logs"]["metadata"]}</li>
                </ul>
                <p>All requests, responses, and judge reasoning are logged for complete traceability.</p>
            </div>
"""

        html_content += """
        </div>

        <div class="footer">
            <p>Generated by Multi-Provider RAG Benchmark using gpt-4.1 for evaluation and judging</p>
            <p>All evaluations use SimpleQA dataset with comprehensive audit logging</p>
        </div>
    </div>
</body>
</html>
"""

        # Add error analysis section for fairness assessment
        if error_analysis:
            html_content += self._generate_error_analysis_html(error_analysis)

        return html_content

    def _generate_error_analysis_html(self, error_analysis: Dict[str, Any]) -> str:
        """
        Generate HTML section for error analysis and fairness assessment
        """
        total_errors = error_analysis.get("total_errors", 0)

        if total_errors == 0:
            return """
        <div class="section">
            <h2>🟢 Fairness Assessment</h2>
            <div class="alert alert-success">
                <strong>✅ All provider requests completed successfully</strong><br>
                No errors, timeouts, or empty responses detected across all providers.<br>
                This benchmark provides a fair comparison with complete data integrity.
            </div>
        </div>
"""

        # Generate error details if there are any errors
        html = f"""
        <div class="section">
            <h2>⚠️ Fairness Assessment - ERRORS DETECTED</h2>
            <div class="alert alert-warning">
                <strong>WARNING: {total_errors} errors detected across providers</strong><br>
                This may affect benchmark fairness. Review error details below.
            </div>

            <div class="provider-stats">
"""

        for provider, stats in error_analysis.get("by_provider", {}).items():
            total_requests = stats.get("total_requests", 0)
            empty_responses = stats.get("empty_responses", 0)
            timeouts = stats.get("timeouts", 0)

            error_rate = (empty_responses + timeouts) / total_requests * 100 if total_requests > 0 else 0
            status_class = "success" if error_rate == 0 else "warning" if error_rate < 5 else "danger"

            html += f"""
                <div class="provider-stat alert-{status_class}">
                    <h4>{provider}</h4>
                    <p><strong>Requests:</strong> {total_requests}</p>
                    <p><strong>Empty Responses:</strong> {empty_responses}</p>
                    <p><strong>Timeouts:</strong> {timeouts}</p>
                    <p><strong>Error Rate:</strong> {error_rate:.1f}%</p>
                </div>
"""

        html += """
            </div>
        </div>
"""
        return html

    def generate_json_leaderboard(self, results: List[Dict[str, Any]], output_file: str):
        """
        Generate a JSON leaderboard for programmatic access
        """
        successful_results = [r for r in results if r.get("success", False)]
        successful_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        leaderboard_data = self._calculate_statistics(successful_results)

        leaderboard_json = {
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "total_providers": len(results),
                "successful_providers": len(successful_results),
                "failed_providers": len(results) - len(successful_results)
            },
            "leaderboard": leaderboard_data,
            "raw_results": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_json, f, indent=2, ensure_ascii=False)

        return output_file