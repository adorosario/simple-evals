#!/usr/bin/env python3
"""
Universal Penalty Deep-Dive Analysis
Comprehensive investigation of penalty cases for ANY provider (CustomGPT, OpenAI RAG, OpenAI Vanilla).

Academic-grade analysis for peer review:
- Every penalty question with detailed breakdown
- Provider judge analysis for penalty decisions
- Comparison with other providers on same questions
- Detailed error categorization and root cause analysis
- Fair, unbiased analysis suitable for academic publication
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Provider name mappings for consistency
PROVIDER_DISPLAY_NAMES = {
    'CustomGPT_RAG': 'CustomGPT',
    'OpenAI_RAG': 'OpenAI RAG',
    'OpenAI_Vanilla': 'OpenAI Vanilla'
}

@dataclass
class PenaltyCase:
    """Detailed penalty case analysis - provider-agnostic"""
    question_id: str
    question: str
    target_answer: str

    # Provider under analysis
    provider_name: str
    provider_answer: str
    provider_confidence: float
    provider_grade: str

    # Judge analysis
    judge_reasoning: str
    judge_confidence: float

    # Penalty information
    penalty_type: str  # 'overconfidence', 'abstention_failure', 'incorrect_confident'
    penalty_points: float

    # Question metadata
    question_domain: str
    question_complexity: float
    answer_length: int

    # Comparative analysis with other providers
    competitor_results: Dict[str, Dict[str, Any]] = None  # {provider: {answer, grade, status}}

    # Abstention analysis
    abstention_judge_recommendation: Optional[str] = None
    abstention_judge_confidence: Optional[float] = None

@dataclass
class PenaltyAnalysis:
    """Analysis results for penalty patterns"""
    provider_name: str
    total_penalties: int
    total_penalty_points: float
    penalty_categories: Dict[str, int]
    domain_penalties: Dict[str, int]
    complexity_penalties: Dict[str, List[float]]
    root_causes: Dict[str, int]
    improvement_opportunities: List[str]
    competitive_analysis: Dict[str, Any]  # How competitors performed on same questions

class UniversalPenaltyAnalyzer:
    """Deep-dive analyzer for any provider's penalty cases"""

    def __init__(self, run_dir: str, provider: str, output_dir: Optional[str] = None):
        """
        Args:
            run_dir: Path to evaluation run directory
            provider: Provider to analyze ('customgpt', 'openai_rag', 'openai_vanilla')
            output_dir: Output directory (default: {run_dir}/{provider}_penalty_analysis)
        """
        self.run_dir = Path(run_dir)
        self.provider = provider.lower()
        self.provider_key = self._get_provider_key(provider)
        self.provider_display = PROVIDER_DISPLAY_NAMES.get(self.provider_key, self.provider_key)

        # Set output directory
        if output_dir is None:
            self.output_dir = self.run_dir / f"{self.provider}_penalty_analysis"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load all evaluation data
        self.evaluations = self._load_evaluations()
        self.abstention_data = self._load_abstention_classifications()

        print(f"Loaded {len(self.evaluations)} evaluations for {self.provider_display} penalty analysis")

    def _get_provider_key(self, provider: str) -> str:
        """Convert provider argument to internal key"""
        provider = provider.lower()
        mapping = {
            'customgpt': 'CustomGPT_RAG',
            'openai_rag': 'OpenAI_RAG',
            'openai_vanilla': 'OpenAI_Vanilla'
        }
        if provider not in mapping:
            raise ValueError(f"Unknown provider: {provider}. Must be one of: {list(mapping.keys())}")
        return mapping[provider]

    def _load_evaluations(self) -> Dict[str, Dict]:
        """Load all judge evaluations organized by question and provider"""
        evaluations = {}

        judge_file = self.run_dir / "judge_evaluations.jsonl"
        if not judge_file.exists():
            raise FileNotFoundError(f"Judge evaluations not found: {judge_file}")

        with open(judge_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)

                    # Skip consistency test evaluations
                    provider = data['metadata']['real_provider_name']
                    if 'ConsistencyTest' in provider:
                        continue

                    question_id = data['question_id']
                    if question_id not in evaluations:
                        evaluations[question_id] = {}

                    evaluations[question_id][provider] = data

        return evaluations

    def _load_abstention_classifications(self) -> Dict[str, Dict]:
        """Load abstention recommendations"""
        abstention_data = {}

        abs_file = self.run_dir / "abstention_classifications.jsonl"
        if not abs_file.exists():
            print(f"Warning: Abstention classifications not found: {abs_file}")
            return abstention_data

        with open(abs_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    question_id = data['question_id']
                    # Use real_provider_name from metadata
                    provider = data.get('metadata', {}).get('real_provider_name', data.get('provider_name'))

                    if question_id not in abstention_data:
                        abstention_data[question_id] = {}

                    abstention_data[question_id][provider] = data

        return abstention_data

    def identify_penalty_cases(self) -> List[PenaltyCase]:
        """
        Identify all questions where the target provider incurred penalties.

        A penalty is incurred when:
        - Grade is not 'A' (correct answer)
        - Confidence threshold was exceeded (answered when should abstain)
        """
        penalty_cases = []

        for question_id, provider_evals in self.evaluations.items():
            if self.provider_key not in provider_evals:
                continue

            provider_data = provider_evals[self.provider_key]

            # Extract provider answer
            provider_answer = list(provider_data['provider_responses'].values())[0]

            # Parse judge response (new format)
            judge_data = provider_data.get('judge', {})
            if 'response' in judge_data and isinstance(judge_data['response'], str):
                # Parse JSON string response
                try:
                    judge_response = json.loads(judge_data['response'])
                    grade = judge_response.get('grade', 'B')
                    confidence = judge_response.get('confidence', 1.0)
                    reasoning = judge_response.get('reasoning', '')
                except json.JSONDecodeError:
                    # Fallback to grades dict
                    blind_id = list(provider_data['provider_responses'].keys())[0]
                    grade_str = provider_data.get('grades', {}).get(blind_id, 'INCORRECT')
                    grade = 'A' if grade_str == 'CORRECT' else 'B'
                    confidence = 1.0
                    reasoning = judge_data.get('reasoning', '')
            else:
                # Fallback to grades dict
                blind_id = list(provider_data['provider_responses'].keys())[0]
                grade_str = provider_data.get('grades', {}).get(blind_id, 'INCORRECT')
                grade = 'A' if grade_str == 'CORRECT' else 'B'
                confidence = 1.0
                reasoning = judge_data.get('reasoning', '')

            # Calculate if this case incurred penalty
            penalty_points = 0.0
            penalty_type = "none"

            if grade != 'A':
                # Failed to answer correctly
                penalty_points = 4.0
                penalty_type = "incorrect_answer"

            if penalty_points == 0:
                continue  # No penalty, skip

            # Get competitor results
            competitor_results = {}
            all_providers = ['CustomGPT_RAG', 'OpenAI_RAG', 'OpenAI_Vanilla']
            for comp_provider in all_providers:
                if comp_provider == self.provider_key:
                    continue
                if comp_provider in provider_evals:
                    comp_data = provider_evals[comp_provider]
                    comp_answer = list(comp_data['provider_responses'].values())[0]

                    # Parse competitor grade
                    comp_judge_data = comp_data.get('judge', {})
                    if 'response' in comp_judge_data and isinstance(comp_judge_data['response'], str):
                        try:
                            comp_judge_response = json.loads(comp_judge_data['response'])
                            comp_grade = comp_judge_response.get('grade', 'B')
                        except json.JSONDecodeError:
                            blind_id = list(comp_data['provider_responses'].keys())[0]
                            grade_str = comp_data.get('grades', {}).get(blind_id, 'INCORRECT')
                            comp_grade = 'A' if grade_str == 'CORRECT' else 'B'
                    else:
                        blind_id = list(comp_data['provider_responses'].keys())[0]
                        grade_str = comp_data.get('grades', {}).get(blind_id, 'INCORRECT')
                        comp_grade = 'A' if grade_str == 'CORRECT' else 'B'

                    competitor_results[comp_provider] = {
                        'answer': comp_answer,
                        'grade': comp_grade,
                        'status': 'PASSED' if comp_grade == 'A' else 'FAILED'
                    }

            # Get abstention recommendation
            abstention_judge_rec = None
            abstention_judge_conf = None
            if question_id in self.abstention_data:
                if self.provider_key in self.abstention_data[question_id]:
                    abs_data = self.abstention_data[question_id][self.provider_key]
                    if 'classifier' in abs_data:
                        abstention_judge_rec = abs_data['classifier'].get('classification')
                        abstention_judge_conf = abs_data['classifier'].get('confidence', 0.0)

            penalty_case = PenaltyCase(
                question_id=question_id,
                question=provider_data['question'],
                target_answer=provider_data.get('target_answer', provider_data.get('target', 'N/A')),
                provider_name=self.provider_display,
                provider_answer=provider_answer,
                provider_confidence=provider_data.get('metadata', {}).get('confidence', 1.0),
                provider_grade=grade,
                judge_reasoning=reasoning,
                judge_confidence=confidence,
                penalty_type=penalty_type,
                penalty_points=penalty_points,
                question_domain=provider_data['metadata'].get('domain', 'unknown'),
                question_complexity=provider_data['metadata'].get('complexity', 0.0),
                answer_length=len(provider_answer.split()),
                competitor_results=competitor_results,
                abstention_judge_recommendation=abstention_judge_rec,
                abstention_judge_confidence=abstention_judge_conf
            )

            penalty_cases.append(penalty_case)

        print(f"\nIdentified {len(penalty_cases)} penalty cases for {self.provider_display}")
        return penalty_cases

    def analyze_patterns(self, penalty_cases: List[PenaltyCase]) -> PenaltyAnalysis:
        """Analyze patterns in penalty cases"""

        if not penalty_cases:
            return PenaltyAnalysis(
                provider_name=self.provider_display,
                total_penalties=0,
                total_penalty_points=0.0,
                penalty_categories={},
                domain_penalties={},
                complexity_penalties={},
                root_causes={},
                improvement_opportunities=[],
                competitive_analysis={}
            )

        # Basic statistics
        total_penalties = len(penalty_cases)
        total_penalty_points = sum(case.penalty_points for case in penalty_cases)

        # Categorize penalties
        penalty_categories = Counter(case.penalty_type for case in penalty_cases)
        domain_penalties = Counter(case.question_domain for case in penalty_cases)

        # Complexity analysis
        complexity_penalties = defaultdict(list)
        for case in penalty_cases:
            complexity_penalties[case.penalty_type].append(case.question_complexity)

        # Root cause analysis
        root_causes = self._identify_root_causes(penalty_cases)

        # Competitive analysis
        competitive_analysis = self._analyze_competitive_performance(penalty_cases)

        # Generate improvement opportunities
        improvement_opportunities = self._generate_recommendations(
            penalty_cases, root_causes, competitive_analysis
        )

        return PenaltyAnalysis(
            provider_name=self.provider_display,
            total_penalties=total_penalties,
            total_penalty_points=total_penalty_points,
            penalty_categories=dict(penalty_categories),
            domain_penalties=dict(domain_penalties),
            complexity_penalties=dict(complexity_penalties),
            root_causes=root_causes,
            improvement_opportunities=improvement_opportunities,
            competitive_analysis=competitive_analysis
        )

    def _identify_root_causes(self, penalty_cases: List[PenaltyCase]) -> Dict[str, int]:
        """Identify root causes of penalties"""
        causes = defaultdict(int)

        for case in penalty_cases:
            # Check if provider should have abstained
            if case.abstention_judge_recommendation == 'abstain':
                causes['should_have_abstained'] += 1
                if case.abstention_judge_confidence and case.abstention_judge_confidence > 0.8:
                    causes['high_confidence_abstention_recommendation'] += 1

            # Check if competitors succeeded
            if case.competitor_results:
                competitor_success_count = sum(
                    1 for comp_data in case.competitor_results.values()
                    if comp_data['status'] == 'PASSED'
                )
                if competitor_success_count == len(case.competitor_results):
                    causes['all_competitors_succeeded'] += 1
                elif competitor_success_count > 0:
                    causes['some_competitors_succeeded'] += 1
                else:
                    causes['all_providers_failed'] += 1

            # Complexity-based causes
            if case.question_complexity > 0.8:
                causes['high_complexity_question'] += 1

            # Answer length analysis
            if case.answer_length < 5:
                causes['very_short_answer'] += 1
            elif case.answer_length > 100:
                causes['very_long_answer'] += 1

        return dict(causes)

    def _analyze_competitive_performance(self, penalty_cases: List[PenaltyCase]) -> Dict[str, Any]:
        """Analyze how competitors performed on questions where target provider failed"""

        competitive_stats = {
            'provider_failed_competitor_passed': defaultdict(int),
            'all_failed_together': 0,
            'provider_only_failure': 0
        }

        for case in penalty_cases:
            if not case.competitor_results:
                continue

            competitor_pass_count = sum(
                1 for comp_data in case.competitor_results.values()
                if comp_data['status'] == 'PASSED'
            )

            if competitor_pass_count == 0:
                competitive_stats['all_failed_together'] += 1
            elif competitor_pass_count == len(case.competitor_results):
                competitive_stats['provider_only_failure'] += 1
                for comp_name in case.competitor_results.keys():
                    competitive_stats['provider_failed_competitor_passed'][comp_name] += 1

        return dict(competitive_stats)

    def _generate_recommendations(
        self,
        penalty_cases: List[PenaltyCase],
        root_causes: Dict[str, int],
        competitive_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        total_penalties = len(penalty_cases)

        # Abstention recommendations
        if root_causes.get('should_have_abstained', 0) > total_penalties * 0.3:
            recommendations.append(
                f"Improve abstention calibration: {root_causes['should_have_abstained']} cases "
                f"({100*root_causes['should_have_abstained']/total_penalties:.1f}%) should have abstained"
            )

        # Competitive gap analysis
        if competitive_analysis.get('provider_only_failure', 0) > 0:
            recommendations.append(
                f"Investigate {competitive_analysis['provider_only_failure']} cases where "
                f"{self.provider_display} failed but all competitors succeeded"
            )

        # Complexity-based recommendations
        if root_causes.get('high_complexity_question', 0) > total_penalties * 0.5:
            recommendations.append(
                f"Focus on high-complexity questions: {root_causes['high_complexity_question']} "
                f"failures ({100*root_causes['high_complexity_question']/total_penalties:.1f}%) "
                f"on complex questions"
            )

        return recommendations

    def generate_engineering_report(
        self,
        penalty_cases: List[PenaltyCase],
        analysis: PenaltyAnalysis
    ) -> str:
        """Generate detailed engineering post-mortem report"""

        run_id = self.run_dir.name
        report = []

        # Header
        report.append(f"# {self.provider_display} Engineering Post-Mortem Report")
        report.append(f"**Run ID:** `{run_id}`")
        report.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("---")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Penalty Cases:** {analysis.total_penalties}")
        report.append(f"- **Total Penalty Points:** {analysis.total_penalty_points}")
        report.append(f"- **Average Penalty per Case:** {analysis.total_penalty_points / max(analysis.total_penalties, 1):.2f}")
        report.append("")

        # Penalty breakdown
        if analysis.penalty_categories:
            report.append("### Penalty Categories")
            report.append("")
            for category, count in sorted(analysis.penalty_categories.items(), key=lambda x: -x[1]):
                pct = 100 * count / analysis.total_penalties
                report.append(f"- **{category.replace('_', ' ').title()}:** {count} ({pct:.1f}%)")
            report.append("")

        # Competitive analysis
        if analysis.competitive_analysis:
            report.append("### Competitive Performance Analysis")
            report.append("")
            report.append(f"- **{self.provider_display} unique failures:** "
                         f"{analysis.competitive_analysis.get('provider_only_failure', 0)} "
                         f"(all competitors passed)")
            report.append(f"- **Shared failures:** "
                         f"{analysis.competitive_analysis.get('all_failed_together', 0)} "
                         f"(all providers failed)")
            report.append("")

        # Domain breakdown
        if analysis.domain_penalties:
            report.append("### Domain-Specific Failures")
            report.append("")
            for domain, count in sorted(analysis.domain_penalties.items(), key=lambda x: -x[1]):
                pct = 100 * count / analysis.total_penalties
                report.append(f"- **{domain}:** {count} ({pct:.1f}%)")
            report.append("")

        # Root causes
        if analysis.root_causes:
            report.append("### Root Cause Analysis")
            report.append("")
            for cause, count in sorted(analysis.root_causes.items(), key=lambda x: -x[1]):
                report.append(f"- **{cause.replace('_', ' ').title()}:** {count}")
            report.append("")

        # Recommendations
        if analysis.improvement_opportunities:
            report.append("### Recommended Actions")
            report.append("")
            for i, rec in enumerate(analysis.improvement_opportunities, 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Detailed case studies
        report.append("---")
        report.append("")
        report.append("## Detailed Penalty Cases")
        report.append("")

        for i, case in enumerate(penalty_cases, 1):
            report.append(f"### Case {i}: {case.question_id}")
            report.append("")
            report.append(f"**Question:** {case.question}")
            report.append(f"**Target Answer:** {case.target_answer}")
            report.append("")
            report.append(f"**{self.provider_display} Answer:** {case.provider_answer}")
            report.append(f"**Grade:** {case.provider_grade}")
            report.append(f"**Confidence:** {case.provider_confidence:.3f}")
            report.append(f"**Penalty Points:** {case.penalty_points}")
            report.append("")

            # Judge analysis
            report.append(f"**Judge Reasoning:** {case.judge_reasoning}")
            report.append(f"**Judge Confidence:** {case.judge_confidence:.3f}")
            report.append("")

            # Abstention analysis
            if case.abstention_judge_recommendation:
                report.append(f"**Abstention Recommendation:** {case.abstention_judge_recommendation}")
                if case.abstention_judge_confidence:
                    report.append(f"**Abstention Judge Confidence:** {case.abstention_judge_confidence:.3f}")
                report.append("")

            # Competitor comparison
            if case.competitor_results:
                report.append("**Competitor Performance:**")
                report.append("")
                for comp_name, comp_data in case.competitor_results.items():
                    comp_display = PROVIDER_DISPLAY_NAMES.get(comp_name, comp_name)
                    status_icon = "✅" if comp_data['status'] == 'PASSED' else "❌"
                    report.append(f"- {status_icon} **{comp_display}:** Grade {comp_data['grade']}")
                    report.append(f"  - Answer: {comp_data['answer'][:200]}...")
                report.append("")

            report.append("---")
            report.append("")

        return "\n".join(report)

    def save_analysis(
        self,
        penalty_cases: List[PenaltyCase],
        analysis: PenaltyAnalysis
    ):
        """Save analysis results to files"""

        run_id = self.run_dir.name

        # Save engineering report (markdown)
        report_md = self.generate_engineering_report(penalty_cases, analysis)
        md_file = self.output_dir / f"{self.provider}_engineering_report_{run_id}.md"
        with open(md_file, 'w') as f:
            f.write(report_md)
        print(f"✓ Engineering report saved to: {md_file}")

        # Save structured data (JSON)
        json_data = {
            "metadata": {
                "run_id": run_id,
                "provider": self.provider_display,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "total_questions_evaluated": len(self.evaluations)
            },
            "penalty_cases": [
                {
                    'question_id': case.question_id,
                    'question': case.question,
                    'target_answer': case.target_answer,
                    f'{self.provider}_answer': case.provider_answer,
                    f'{self.provider}_grade': case.provider_grade,
                    f'{self.provider}_confidence': case.provider_confidence,
                    'penalty_points': case.penalty_points,
                    'penalty_type': case.penalty_type,
                    'domain': case.question_domain,
                    'complexity': case.question_complexity,
                    'judge_reasoning': case.judge_reasoning,
                    'judge_confidence': case.judge_confidence,
                    'competitor_results': case.competitor_results,
                    'abstention_recommendation': case.abstention_judge_recommendation,
                    'abstention_judge_confidence': case.abstention_judge_confidence
                }
                for case in penalty_cases
            ],
            "analysis": {
                "total_penalties": analysis.total_penalties,
                "total_penalty_points": analysis.total_penalty_points,
                "penalty_categories": analysis.penalty_categories,
                "domain_penalties": analysis.domain_penalties,
                "root_causes": analysis.root_causes,
                "competitive_analysis": analysis.competitive_analysis,
                "improvement_opportunities": analysis.improvement_opportunities
            }
        }

        json_file = self.output_dir / f"{self.provider}_penalty_analysis_{run_id}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ JSON analysis saved to: {json_file}")

        # Save CSV for easy spreadsheet analysis
        if penalty_cases:
            df_data = []
            for case in penalty_cases:
                row = {
                    'question_id': case.question_id,
                    'question': case.question,
                    'target_answer': case.target_answer,
                    f'{self.provider}_answer': case.provider_answer,
                    f'{self.provider}_grade': case.provider_grade,
                    f'{self.provider}_confidence': case.provider_confidence,
                    'penalty_points': case.penalty_points,
                    'penalty_type': case.penalty_type,
                    'domain': case.question_domain,
                    'complexity': case.question_complexity,
                    'judge_confidence': case.judge_confidence,
                    'abstention_recommendation': case.abstention_judge_recommendation
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            csv_file = self.output_dir / f"{self.provider}_penalty_cases_{run_id}.csv"
            df.to_csv(csv_file, index=False)
            print(f"✓ CSV data saved to: {csv_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Universal Penalty Deep-Dive Analysis for any RAG provider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CustomGPT failures
  python universal_penalty_analyzer.py --run-dir results/run_XXX --provider customgpt

  # Analyze OpenAI RAG failures
  python universal_penalty_analyzer.py --run-dir results/run_XXX --provider openai_rag

  # Analyze OpenAI Vanilla failures
  python universal_penalty_analyzer.py --run-dir results/run_XXX --provider openai_vanilla
        """
    )
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument(
        '--provider',
        required=True,
        choices=['customgpt', 'openai_rag', 'openai_vanilla'],
        help='Provider to analyze'
    )
    parser.add_argument('--output-dir', default=None, help='Output directory (default: {run_dir}/{provider}_penalty_analysis)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = UniversalPenaltyAnalyzer(args.run_dir, args.provider, args.output_dir)

    # Identify penalty cases
    print(f"\n{'='*60}")
    print(f"Analyzing {analyzer.provider_display} Penalty Cases")
    print(f"{'='*60}\n")

    penalty_cases = analyzer.identify_penalty_cases()

    if not penalty_cases:
        print(f"✓ No penalty cases found for {analyzer.provider_display}! 🎉")
        return

    # Analyze patterns
    analysis = analyzer.analyze_patterns(penalty_cases)

    # Save results
    analyzer.save_analysis(penalty_cases, analysis)

    print(f"\n{'='*60}")
    print(f"Analysis complete for {analyzer.provider_display}")
    print(f"Total Penalties: {analysis.total_penalties}")
    print(f"Total Penalty Points: {analysis.total_penalty_points}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
