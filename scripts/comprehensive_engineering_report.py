#!/usr/bin/env python3
"""
Comprehensive Engineering Report Generator
Creates a professional post-mortem report for provider penalty analysis.

This script combines penalty analysis, competitive comparison, and failure insights
into a comprehensive engineering document for improvement recommendations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

def load_penalty_analysis(run_dir: str, provider_name: str) -> Dict[str, Any]:
    """Load the detailed penalty analysis data"""
    run_id = Path(run_dir).name

    # Try multiple possible locations for penalty analysis
    possible_locations = [
        Path(run_dir) / f"{provider_name}_penalty_analysis" / f"{provider_name}_penalty_analysis_{run_id}.json",
        Path(f"{provider_name}_penalty_analysis") / f"{provider_name}_penalty_analysis_{run_id}.json",
        Path(".") / f"{provider_name}_penalty_analysis" / f"{provider_name}_penalty_analysis_{run_id}.json"
    ]

    for analysis_file in possible_locations:
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                return json.load(f)

    raise FileNotFoundError(f"Penalty analysis file not found. Tried: {[str(f) for f in possible_locations]}")

def load_run_metadata(run_dir: str) -> Dict[str, Any]:
    """Load run metadata to get overall metrics"""
    metadata_file = Path(run_dir) / "run_metadata.json"
    with open(metadata_file, 'r') as f:
        return json.load(f)

def get_provider_metrics(metadata: Dict[str, Any], provider_name: str) -> Dict[str, Any]:
    """Extract metrics for the specified provider"""
    for result in metadata['results']['results']:
        if provider_name.lower() in result['sampler_name'].lower():
            return result
    return None

def generate_executive_summary(provider_metrics: Dict[str, Any], penalty_cases: List[Dict]) -> str:
    """Generate executive summary section"""
    metrics = provider_metrics['metrics']
    sampler_name = provider_metrics['sampler_name']

    total_questions = metrics['n_correct'] + metrics['n_incorrect'] + metrics['n_not_attempted']
    accuracy = metrics['n_correct'] / (metrics['n_correct'] + metrics['n_incorrect']) * 100 if (metrics['n_correct'] + metrics['n_incorrect']) > 0 else 0

    return f"""## Executive Summary

**Provider**: {sampler_name}
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d')}
**Total Questions**: {total_questions}
**Overall Performance**:
- Volume Score: {metrics['volume_score']:.3f}
- Quality Score: {metrics['quality_score']:.3f}
- Accuracy: {accuracy:.1f}% ({metrics['n_correct']}/{metrics['n_correct'] + metrics['n_incorrect']})
- Abstention Rate: {metrics['n_not_attempted']}/{total_questions} ({metrics['n_not_attempted']/total_questions*100:.1f}%)

**Critical Issues**:
- **{len(penalty_cases)} incorrect answers** resulting in {len(penalty_cases) * 4.0:.1f} penalty points
- Quality score reduced by {metrics['volume_score'] - metrics['quality_score']:.3f} points due to penalties
- Penalty ratio of {metrics['penalty_ratio']} applied per incorrect answer

**Key Finding**: Each incorrect answer (B grade) receives a -{metrics['penalty_ratio']:.1f} quality score penalty, significantly impacting overall performance.
"""

def generate_domain_analysis(penalty_cases: List[Dict]) -> str:
    """Generate domain-specific analysis"""
    domain_breakdown = {}
    for case in penalty_cases:
        domain = case['domain']
        if domain not in domain_breakdown:
            domain_breakdown[domain] = []
        domain_breakdown[domain].append(case)

    analysis = "## Domain Analysis\n\n"

    for domain, cases in sorted(domain_breakdown.items(), key=lambda x: len(x[1]), reverse=True):
        analysis += f"### {domain.title()} Domain ({len(cases)} failures)\n\n"

        avg_complexity = sum(case['complexity'] for case in cases) / len(cases)
        avg_confidence = sum(case['customgpt_confidence'] for case in cases) / len(cases)

        analysis += f"- **Failure Count**: {len(cases)}\n"
        analysis += f"- **Average Complexity**: {avg_complexity:.3f}\n"
        analysis += f"- **Average Confidence**: {avg_confidence:.3f}\n"
        analysis += f"- **Impact**: {len(cases) * 4.0:.1f} penalty points\n\n"

        analysis += "**Failed Questions**:\n"
        for i, case in enumerate(cases, 1):
            analysis += f"{i}. `{case['question_id']}`: {case['question'][:80]}...\n"
        analysis += "\n"

    return analysis

def generate_competitive_analysis(penalty_cases: List[Dict]) -> str:
    """Generate competitive comparison analysis"""
    analysis = "## Competitive Analysis\n\n"

    openai_rag_wins = sum(1 for case in penalty_cases if case['openai_rag_grade'] == 'A')
    openai_vanilla_wins = sum(1 for case in penalty_cases if case['openai_vanilla_grade'] == 'A')
    both_failed = sum(1 for case in penalty_cases if case['openai_rag_grade'] == 'B' and case['openai_vanilla_grade'] == 'B')

    analysis += f"**Performance Comparison** (on {len(penalty_cases)} failed questions):\n\n"
    analysis += f"- **OpenAI RAG succeeded**: {openai_rag_wins}/{len(penalty_cases)} ({openai_rag_wins/len(penalty_cases)*100:.1f}%)\n"
    analysis += f"- **OpenAI Vanilla succeeded**: {openai_vanilla_wins}/{len(penalty_cases)} ({openai_vanilla_wins/len(penalty_cases)*100:.1f}%)\n"
    analysis += f"- **Both competitors failed**: {both_failed}/{len(penalty_cases)} ({both_failed/len(penalty_cases)*100:.1f}%)\n\n"

    if openai_rag_wins > openai_vanilla_wins:
        analysis += "**Key Insight**: OpenAI RAG outperformed vanilla on questions where CustomGPT failed, suggesting retrieval advantages.\n\n"
    elif openai_vanilla_wins > openai_rag_wins:
        analysis += "**Key Insight**: OpenAI Vanilla outperformed RAG on questions where CustomGPT failed, suggesting reasoning advantages.\n\n"
    else:
        analysis += "**Key Insight**: OpenAI RAG and Vanilla performed similarly on CustomGPT failure cases.\n\n"

    analysis += "### Detailed Case-by-Case Comparison\n\n"
    analysis += "| Question ID | CustomGPT | OpenAI RAG | OpenAI Vanilla | Pattern |\n"
    analysis += "|-------------|-----------|------------|----------------|----------|\n"

    for case in penalty_cases:
        pattern = ""
        if case['openai_rag_grade'] == 'A' and case['openai_vanilla_grade'] == 'A':
            pattern = "Both competitors succeeded"
        elif case['openai_rag_grade'] == 'A':
            pattern = "RAG advantage"
        elif case['openai_vanilla_grade'] == 'A':
            pattern = "Vanilla advantage"
        else:
            pattern = "All providers failed"

        analysis += f"| {case['question_id']} | B | {case['openai_rag_grade']} | {case['openai_vanilla_grade']} | {pattern} |\n"

    return analysis + "\n"

def generate_failure_details(penalty_cases: List[Dict]) -> str:
    """Generate detailed failure analysis"""
    analysis = "## Detailed Failure Analysis\n\n"

    for i, case in enumerate(penalty_cases, 1):
        analysis += f"### Failure {i}: {case['question_id']}\n\n"
        analysis += f"**Question**: {case['question']}\n\n"
        analysis += f"**Target Answer**: {case['target_answer']}\n\n"
        analysis += f"**CustomGPT Answer**: {case['customgpt_answer']}\n\n"

        analysis += f"**Metrics**:\n"
        analysis += f"- Domain: {case['domain']}\n"
        analysis += f"- Complexity: {case['complexity']:.3f}\n"
        analysis += f"- Confidence: {case['customgpt_confidence']}\n"
        analysis += f"- Penalty Points: {case['penalty_points']}\n\n"

        analysis += f"**Judge Reasoning**: {case['judge_reasoning']}\n\n"

        analysis += f"**Competitive Performance**:\n"
        analysis += f"- OpenAI RAG: {case['openai_rag_grade']}\n"
        analysis += f"- OpenAI Vanilla: {case['openai_vanilla_grade']}\n\n"

        if case['openai_rag_grade'] == 'A' or case['openai_vanilla_grade'] == 'A':
            analysis += "⚠️ **Critical**: Competitors succeeded where CustomGPT failed\n\n"
        else:
            analysis += "ℹ️ **Note**: All providers struggled with this question\n\n"

        analysis += "---\n\n"

    return analysis

def generate_recommendations(penalty_cases: List[Dict], provider_metrics: Dict[str, Any]) -> str:
    """Generate engineering recommendations"""
    metrics = provider_metrics['metrics']

    # Analyze patterns
    high_confidence_failures = [case for case in penalty_cases if case['customgpt_confidence'] > 0.95]
    competitor_success_cases = [case for case in penalty_cases if case['openai_rag_grade'] == 'A' or case['openai_vanilla_grade'] == 'A']

    recommendations = "## Engineering Recommendations\n\n"

    recommendations += "### Priority 1: Immediate Actions\n\n"

    if len(high_confidence_failures) > 0:
        recommendations += f"1. **Confidence Calibration Crisis**: {len(high_confidence_failures)} failures had >95% confidence\n"
        recommendations += "   - Review confidence scoring algorithm\n"
        recommendations += "   - Implement uncertainty estimation improvements\n"
        recommendations += "   - Add confidence validation against retrieval quality\n\n"

    if len(competitor_success_cases) > len(penalty_cases) * 0.5:
        recommendations += f"2. **Competitive Gap**: Competitors succeeded on {len(competitor_success_cases)}/{len(penalty_cases)} failed questions\n"
        recommendations += "   - Analyze competitor knowledge sources\n"
        recommendations += "   - Review retrieval algorithm effectiveness\n"
        recommendations += "   - Benchmark against OpenAI RAG capabilities\n\n"

    # Domain-specific recommendations
    domain_counts = {}
    for case in penalty_cases:
        domain_counts[case['domain']] = domain_counts.get(case['domain'], 0) + 1

    top_domain = max(domain_counts.items(), key=lambda x: x[1])
    if top_domain[1] > 1:
        recommendations += f"3. **Domain Focus**: {top_domain[1]} failures in {top_domain[0]} domain\n"
        recommendations += f"   - Expand {top_domain[0]} knowledge base coverage\n"
        recommendations += f"   - Review {top_domain[0]}-specific retrieval patterns\n"
        recommendations += f"   - Validate {top_domain[0]} fact accuracy\n\n"

    recommendations += "### Priority 2: System Improvements\n\n"
    recommendations += f"1. **Quality Score Impact**: Current penalty ratio of {metrics['penalty_ratio']} is severe\n"
    recommendations += "   - Each wrong answer costs 4.0 quality points\n"
    recommendations += "   - Consider implementing graduated penalties\n"
    recommendations += "   - Evaluate abstention vs incorrect answer trade-offs\n\n"

    recommendations += "2. **Knowledge Base Enhancement**:\n"
    recommendations += "   - Add fact verification layer\n"
    recommendations += "   - Implement source confidence scoring\n"
    recommendations += "   - Review document indexing quality\n\n"

    recommendations += "3. **Retrieval Algorithm**:\n"
    recommendations += "   - Analyze retrieval relevance for failed questions\n"
    recommendations += "   - Implement multi-hop reasoning capabilities\n"
    recommendations += "   - Add semantic similarity improvements\n\n"

    recommendations += "### Priority 3: Long-term Strategy\n\n"
    recommendations += "1. **Continuous Monitoring**:\n"
    recommendations += "   - Implement real-time confidence calibration tracking\n"
    recommendations += "   - Set up automated competitive benchmarking\n"
    recommendations += "   - Create domain-specific performance dashboards\n\n"

    recommendations += "2. **Model Training**:\n"
    recommendations += "   - Fine-tune on identified failure patterns\n"
    recommendations += "   - Implement active learning from failures\n"
    recommendations += "   - Add uncertainty quantification training\n\n"

    return recommendations

def generate_comprehensive_report(run_dir: str, provider_name: str, output_file: str = None) -> str:
    """Generate comprehensive engineering report"""

    # Load all data
    penalty_analysis = load_penalty_analysis(run_dir, provider_name)
    metadata = load_run_metadata(run_dir)
    provider_metrics = get_provider_metrics(metadata, provider_name)

    if not provider_metrics:
        raise ValueError(f"Provider {provider_name} not found in run results")

    penalty_cases = penalty_analysis['penalty_cases']

    # Generate report
    report = f"""# CustomGPT Engineering Post-Mortem Report
**Run ID**: {Path(run_dir).name}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Penalty Cases**: {len(penalty_cases)}

"""

    report += generate_executive_summary(provider_metrics, penalty_cases)
    report += "\n"
    report += generate_domain_analysis(penalty_cases)
    report += "\n"
    report += generate_competitive_analysis(penalty_cases)
    report += "\n"
    report += generate_failure_details(penalty_cases)
    report += "\n"
    report += generate_recommendations(penalty_cases, provider_metrics)

    report += f"""

---
*This report was automatically generated from evaluation run {Path(run_dir).name}*
*Report generated at: {datetime.now().isoformat()}*
"""

    # Save report if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"Comprehensive engineering report saved to: {output_path}")

    return report

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive engineering report')
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--provider', default='customgpt', help='Provider name to analyze')
    parser.add_argument('--output', help='Output file path (default: auto-generated in run directory to keep root clean)')

    args = parser.parse_args()

    # Generate output path if not specified
    if args.output:
        output_file = args.output
    else:
        output_file = f"{args.run_dir}/{args.provider}_engineering_report_{Path(args.run_dir).name}.md"

    # Generate report
    try:
        report = generate_comprehensive_report(args.run_dir, args.provider, output_file)
        print(f"\n✓ Successfully generated comprehensive engineering report")
        print(f"✓ Report saved to: {output_file}")
        print(f"✓ Analyzed {len(load_penalty_analysis(args.run_dir, args.provider)['penalty_cases'])} penalty cases")

    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())