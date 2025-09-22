#!/usr/bin/env python3
"""
Adaptive Blog Post Generator
Analyzes any benchmark run and generates compelling narrative based on actual findings
"""

import json
import sys
from pathlib import Path
from openai import OpenAI

class AdaptiveBlogGenerator:
    def __init__(self):
        self.client = OpenAI()

    def analyze_benchmark_data(self, run_dir: str):
        """Analyze benchmark data to extract insights"""
        run_path = Path(run_dir)

        # Load benchmark results
        with open(run_path / "benchmark_results.json", 'r') as f:
            data = json.load(f)

        # Extract provider performance
        providers = {}
        for result in data["results"]:
            name = result["sampler_name"]
            providers[name] = {
                "accuracy": result["score"] * 100,  # Convert to percentage
                "latency_seconds": result["duration"],
                "total_questions": result["samples_evaluated"],
                "correct_answers": int(result["score"] * result["samples_evaluated"])
            }

        # Calculate insights
        insights = self.extract_insights(providers)

        # Load sample judge evaluations for specific examples
        examples = self.load_examples(run_path)

        return {
            "providers": providers,
            "insights": insights,
            "examples": examples,
            "run_metadata": data.get("configuration", {})
        }

    def extract_insights(self, providers):
        """Extract key insights from provider data"""
        insights = {}

        # Sort by accuracy
        sorted_providers = sorted(providers.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        best_performer = sorted_providers[0]
        worst_performer = sorted_providers[-1]

        insights["performance_gap"] = best_performer[1]["accuracy"] - worst_performer[1]["accuracy"]
        insights["best_provider"] = best_performer[0]
        insights["worst_provider"] = worst_performer[0]
        insights["speed_champion"] = min(providers.items(), key=lambda x: x[1]["latency_seconds"])

        # Find RAG vs Vanilla patterns
        rag_providers = [p for p in providers.keys() if "RAG" in p]
        vanilla_providers = [p for p in providers.keys() if "Vanilla" in p or ("RAG" not in p and "gpt" in p.lower())]

        if rag_providers and vanilla_providers:
            rag_accuracy = max([providers[p]["accuracy"] for p in rag_providers])
            vanilla_accuracy = max([providers[p]["accuracy"] for p in vanilla_providers])
            insights["rag_vs_vanilla"] = {
                "rag_accuracy": rag_accuracy,
                "vanilla_accuracy": vanilla_accuracy,
                "improvement": rag_accuracy - vanilla_accuracy
            }

        return insights

    def load_examples(self, run_path, max_examples=3):
        """Load specific examples of right/wrong answers"""
        examples = {"failures": [], "successes": []}

        try:
            with open(run_path / "judge_evaluations.jsonl", 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_examples * 2:  # Limit examples
                        break

                    eval_data = json.loads(line)

                    # Look for failures (INCORRECT grades)
                    for provider, grade in eval_data.get("grades", {}).items():
                        if grade == "INCORRECT" and len(examples["failures"]) < max_examples:
                            examples["failures"].append({
                                "question": eval_data["question"],
                                "correct_answer": eval_data["target_answer"],
                                "wrong_answer": eval_data["provider_responses"][provider],
                                "provider": provider
                            })
                        elif grade == "CORRECT" and len(examples["successes"]) < max_examples:
                            examples["successes"].append({
                                "question": eval_data["question"],
                                "correct_answer": eval_data["target_answer"],
                                "provider": provider
                            })
        except Exception as e:
            print(f"Warning: Could not load examples: {e}")

        return examples

    def generate_blog_post(self, analysis, output_path):
        """Generate adaptive blog post based on analysis"""

        # Create dynamic prompt based on actual findings
        prompt = self.create_adaptive_prompt(analysis)

        try:
            print("🎨 Generating adaptive blog post based on actual findings...")

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=6000  # Reduced to avoid timeout
            )

            # If GPT-5 returns nothing, create fallback
            html_content = response.choices[0].message.content
            if not html_content or len(html_content.strip()) < 100:
                html_content = self.create_fallback_html(analysis)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ Adaptive blog post created: {output_path}")
            return output_path

        except Exception as e:
            print(f"⚠️ GPT-5 failed, creating fallback HTML: {e}")
            html_content = self.create_fallback_html(analysis)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path

    def create_adaptive_prompt(self, analysis):
        """Create prompt based on actual data insights"""
        providers = analysis["providers"]
        insights = analysis["insights"]
        examples = analysis["examples"]

        # Dynamically generate title based on findings
        if insights.get("performance_gap", 0) > 50:
            title_theme = "Shocking Performance Gap"
        elif "rag_vs_vanilla" in insights:
            title_theme = "RAG vs Vanilla Showdown"
        else:
            title_theme = "AI Performance Analysis"

        prompt = f"""Create a compelling OpenAI-style blog post about AI performance findings.

DYNAMIC DATA ANALYSIS:
Providers tested: {list(providers.keys())}
Performance summary: {json.dumps(insights, indent=2)}

ACTUAL EXAMPLES FROM TESTING:
{json.dumps(examples, indent=2)}

ADAPTIVE REQUIREMENTS:
1. Generate a compelling title based on the actual performance gap found
2. Tell the story of what THIS specific data reveals
3. Use the ACTUAL provider names and numbers from the data
4. Highlight the most surprising finding from this run
5. Include real examples of failures/successes if available

STRUCTURE:
- HTML with embedded Chart.js showing the actual provider data
- OpenAI-style narrative focusing on insights
- Responsive design with modern styling
- Interactive charts with real performance numbers

KEY INSIGHTS TO EXPLORE (adapt based on data):
{self.generate_insight_prompts(insights)}

Create a complete, beautiful HTML blog post that tells the story of these specific findings."""

        return prompt

    def generate_insight_prompts(self, insights):
        """Generate insight prompts based on actual findings"""
        prompts = []

        if insights.get("performance_gap", 0) > 30:
            prompts.append(f"- Massive {insights['performance_gap']:.1f}% performance gap between best and worst")

        if "rag_vs_vanilla" in insights:
            rag_data = insights["rag_vs_vanilla"]
            prompts.append(f"- RAG achieved {rag_data['rag_accuracy']:.1f}% vs Vanilla's {rag_data['vanilla_accuracy']:.1f}%")

        if insights.get("speed_champion"):
            prompts.append(f"- {insights['speed_champion'][0]} was fastest at {insights['speed_champion'][1]['latency_seconds']:.1f}s")

        return "\\n".join(prompts) if prompts else "- Analyze comparative performance patterns"

    def create_fallback_html(self, analysis):
        """Create fallback HTML if GPT-5 fails"""
        providers = analysis["providers"]
        insights = analysis["insights"]

        # Generate chart data
        provider_names = list(providers.keys())
        accuracies = [providers[p]["accuracy"] for p in provider_names]
        latencies = [providers[p]["latency_seconds"] for p in provider_names]

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Performance Analysis: {insights.get('performance_gap', 0):.1f}% Gap Revealed</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; line-height: 1.6; margin: 0; background: #f8fafc; }}
        .container {{ max-width: 900px; margin: 2rem auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 2rem; border-radius: 12px 12px 0 0; }}
        .header h1 {{ font-size: 2.2rem; margin: 0 0 1rem 0; }}
        .content {{ padding: 2rem; }}
        .chart-container {{ background: #f8fafc; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
        .insight {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 1rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Performance Analysis: Key Findings</h1>
            <p>Benchmark results showing {insights.get('performance_gap', 0):.1f}% performance difference across providers</p>
        </div>
        <div class="content">
            <div class="insight">
                <h3>🎯 Key Finding</h3>
                <p><strong>{insights.get('best_provider', 'Top performer')}</strong> achieved {max(accuracies):.1f}% accuracy, while <strong>{insights.get('worst_provider', 'lowest performer')}</strong> reached {min(accuracies):.1f}%.</p>
            </div>

            <div class="chart-container">
                <h3>Accuracy Comparison</h3>
                <canvas id="accuracyChart"></canvas>
            </div>

            <div class="chart-container">
                <h3>Response Time Analysis</h3>
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Accuracy Chart
        new Chart(document.getElementById('accuracyChart'), {{
            type: 'bar',
            data: {{
                labels: {provider_names},
                datasets: [{{
                    label: 'Accuracy (%)',
                    data: {accuracies},
                    backgroundColor: ['#667eea', '#f093fb', '#4facfe', '#43e97b'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});

        // Latency Chart
        new Chart(document.getElementById('latencyChart'), {{
            type: 'bar',
            data: {{
                labels: {provider_names},
                datasets: [{{
                    label: 'Response Time (seconds)',
                    data: {latencies},
                    backgroundColor: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
    </script>
</body>
</html>"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_adaptive_blog_post.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_path = f"{run_dir}/adaptive_blog_post.html"

    generator = AdaptiveBlogGenerator()

    print("🔍 Analyzing benchmark data...")
    analysis = generator.analyze_benchmark_data(run_dir)

    print("📝 Generating adaptive blog post...")
    result = generator.generate_blog_post(analysis, output_path)

    if result:
        print(f"✅ Adaptive blog post created: {result}")
        print("🎯 Content adapted to actual findings from this run")

if __name__ == "__main__":
    main()