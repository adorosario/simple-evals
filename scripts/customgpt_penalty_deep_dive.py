#!/usr/bin/env python3
"""
CustomGPT Penalty Deep-Dive Analysis
Comprehensive investigation of every question where CustomGPT incurred penalty points.

For CustomGPT engineering team analysis:
- Every penalty question with detailed breakdown
- Provider judge analysis for penalty decisions
- Abstention judge analysis
- Comparison with other providers on same questions
- Detailed error categorization and root cause analysis
- Actionable recommendations for CustomGPT improvements
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PenaltyCase:
    """Detailed penalty case analysis"""
    question_id: str
    question: str
    target_answer: str
    customgpt_answer: str
    customgpt_confidence: float
    customgpt_grade: str
    judge_reasoning: str
    penalty_type: str  # 'overconfidence', 'abstention_failure', 'incorrect_confident'
    penalty_points: float
    question_domain: str
    question_complexity: float
    answer_length: int

    # Comparative analysis
    openai_rag_answer: Optional[str] = None
    openai_rag_grade: Optional[str] = None
    openai_vanilla_answer: Optional[str] = None
    openai_vanilla_grade: Optional[str] = None

    # Judge analysis
    judge_confidence: float = 0.0
    abstention_judge_recommendation: Optional[str] = None
    abstention_judge_confidence: Optional[float] = None

@dataclass
class PenaltyAnalysis:
    """Analysis results for penalty patterns"""
    total_penalties: int
    penalty_categories: Dict[str, int]
    domain_penalties: Dict[str, int]
    complexity_penalties: Dict[str, List[float]]
    root_causes: Dict[str, int]
    improvement_opportunities: List[str]

class CustomGPTPenaltyAnalyzer:
    """Deep-dive analyzer for CustomGPT penalty cases"""

    def __init__(self, run_dir: str, output_dir: str = "customgpt_penalty_analysis"):
        self.run_dir = Path(run_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load all evaluation data
        self.evaluations = self._load_evaluations()
        self.abstention_data = self._load_abstention_classifications()

        print(f"Loaded {len(self.evaluations)} evaluations for CustomGPT penalty analysis")

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
        """Load abstention classification data"""
        abstention_data = {}

        abstention_file = self.run_dir / "abstention_classifications.jsonl"
        if not abstention_file.exists():
            print("Warning: No abstention classifications found")
            return {}

        with open(abstention_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    question_id = data['question_id']
                    provider = data['metadata']['real_provider_name']

                    if question_id not in abstention_data:
                        abstention_data[question_id] = {}

                    abstention_data[question_id][provider] = data

        return abstention_data

    def _classify_domain(self, question: str) -> str:
        """Classify question domain"""
        question_lower = question.lower()

        domain_keywords = {
            'science': ['atom', 'molecule', 'chemical', 'physics', 'biology', 'chemistry', 'element', 'theory', 'experiment', 'scientific'],
            'history': ['year', 'century', 'war', 'president', 'empire', 'ancient', 'historical', 'founded', 'revolution', 'era'],
            'geography': ['country', 'city', 'continent', 'mountain', 'river', 'capital', 'located', 'border', 'population', 'nation'],
            'mathematics': ['calculate', 'equation', 'number', 'formula', 'solve', 'probability', 'percentage', 'ratio', 'sum', 'average'],
            'literature': ['author', 'book', 'novel', 'poem', 'wrote', 'published', 'character', 'story', 'literature'],
            'arts': ['artist', 'painting', 'music', 'composer', 'artwork', 'created', 'style', 'museum', 'art'],
            'sports': ['team', 'player', 'game', 'sport', 'championship', 'won', 'season', 'league', 'olympic'],
            'technology': ['computer', 'software', 'internet', 'technology', 'algorithm', 'programming', 'digital', 'tech'],
            'entertainment': ['movie', 'film', 'actor', 'actress', 'tv', 'show', 'director', 'entertainment', 'celebrity'],
            'business': ['company', 'corporation', 'business', 'ceo', 'founded', 'revenue', 'market', 'industry'],
            'politics': ['government', 'political', 'election', 'vote', 'congress', 'senate', 'policy', 'law']
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'

    def _assess_complexity(self, question: str) -> float:
        """Assess question complexity"""
        complexity_indicators = {
            'multi_part': ['and', 'or', 'also', 'additionally', 'furthermore', 'moreover'],
            'temporal': ['when', 'during', 'before', 'after', 'while', 'until'],
            'causal': ['why', 'because', 'due to', 'caused by', 'resulted in', 'reason'],
            'comparative': ['compare', 'contrast', 'difference', 'similar', 'unlike', 'versus'],
            'numerical': ['how many', 'how much', 'percentage', 'ratio', 'calculate', 'count'],
            'analytical': ['analyze', 'explain', 'describe', 'evaluate', 'assess', 'determine']
        }

        question_lower = question.lower()
        complexity_score = 0.0

        # Base complexity from length
        complexity_score += min(len(question) / 100, 1.0)

        # Add complexity indicators
        for indicator_type, keywords in complexity_indicators.items():
            if any(keyword in question_lower for keyword in keywords):
                complexity_score += 0.2

        # Question marks (compound questions)
        complexity_score += question.count('?') * 0.1

        # Proper nouns (specific entities)
        words = question.split()
        capitalized_ratio = sum(1 for word in words if word[0].isupper() and len(word) > 2) / max(len(words), 1)
        complexity_score += capitalized_ratio * 0.3

        # Technical terms and specific dates
        technical_patterns = [
            r'\\d{4}',  # Years
            r'\\b[A-Z]{2,}\\b',  # Acronyms
            r'\\d+\\.\\d+',  # Decimal numbers
            r'\\$\\d+',  # Money
        ]

        for pattern in technical_patterns:
            if re.search(pattern, question):
                complexity_score += 0.1

        return min(complexity_score, 3.0)

    def identify_penalty_cases(self) -> List[PenaltyCase]:
        """
        Identify all CustomGPT cases that incurred penalty points

        Penalty scenarios:
        1. Overconfidence penalty: Confident but wrong (grade C/D/F with high confidence)
        2. Should have abstained: Wrong answer with confidence above threshold
        3. Quality-confidence mismatch: Grade doesn't match confidence level
        """

        print(f"\\n=== IDENTIFYING CUSTOMGPT PENALTY CASES ===")

        penalty_cases = []
        confidence_threshold = 0.8  # 80% threshold from run metadata
        penalty_ratio = 4.0

        customgpt_provider = "CustomGPT_RAG"

        for question_id, providers in self.evaluations.items():
            if customgpt_provider not in providers:
                continue

            customgpt_data = providers[customgpt_provider]

            try:
                # Extract CustomGPT evaluation details
                judge_response = json.loads(customgpt_data['judge']['response'])
                customgpt_answer = list(customgpt_data['provider_responses'].values())[0]

                grade = judge_response['grade']
                confidence = judge_response.get('confidence', 1.0)
                reasoning = judge_response.get('reasoning', '')

                # Calculate if this case incurred penalty
                penalty_points = 0
                penalty_type = None

                # Grade penalties - CORRECTED understanding:
                # A = CORRECT (no penalty)
                # B = INCORRECT (penalty points)
                # No C/D/F grades exist - it's binary correct/incorrect

                if grade == 'B':
                    # B grade = incorrect answer
                    penalty_points = penalty_ratio  # Should be 4.0 points per wrong answer
                    penalty_type = 'incorrect_answer'

                # Analyze ALL incorrect answers (B grades only)
                if grade == 'B':

                    # Get comparative data from other providers
                    openai_rag_answer = None
                    openai_rag_grade = None
                    openai_vanilla_answer = None
                    openai_vanilla_grade = None

                    if "OpenAI_RAG" in providers:
                        rag_judge = json.loads(providers["OpenAI_RAG"]['judge']['response'])
                        openai_rag_answer = list(providers["OpenAI_RAG"]['provider_responses'].values())[0]
                        openai_rag_grade = rag_judge['grade']

                    if "OpenAI_Vanilla" in providers:
                        vanilla_judge = json.loads(providers["OpenAI_Vanilla"]['judge']['response'])
                        openai_vanilla_answer = list(providers["OpenAI_Vanilla"]['provider_responses'].values())[0]
                        openai_vanilla_grade = vanilla_judge['grade']

                    # Get abstention analysis if available
                    abstention_judge_rec = None
                    abstention_judge_conf = None
                    if question_id in self.abstention_data and customgpt_provider in self.abstention_data[question_id]:
                        abs_data = self.abstention_data[question_id][customgpt_provider]
                        if 'classifier' in abs_data:
                            abstention_judge_rec = abs_data['classifier'].get('classification')
                            abstention_judge_conf = abs_data['classifier'].get('confidence', 0.0)

                    penalty_case = PenaltyCase(
                        question_id=question_id,
                        question=customgpt_data['question'],
                        target_answer=customgpt_data['target_answer'],
                        customgpt_answer=customgpt_answer,
                        customgpt_confidence=confidence,
                        customgpt_grade=grade,
                        judge_reasoning=reasoning,
                        penalty_type=penalty_type,
                        penalty_points=penalty_points,
                        question_domain=self._classify_domain(customgpt_data['question']),
                        question_complexity=self._assess_complexity(customgpt_data['question']),
                        answer_length=len(customgpt_answer),
                        openai_rag_answer=openai_rag_answer,
                        openai_rag_grade=openai_rag_grade,
                        openai_vanilla_answer=openai_vanilla_answer,
                        openai_vanilla_grade=openai_vanilla_grade,
                        judge_confidence=judge_response.get('confidence', 1.0),
                        abstention_judge_recommendation=abstention_judge_rec,
                        abstention_judge_confidence=abstention_judge_conf
                    )

                    penalty_cases.append(penalty_case)

            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                continue

        # Sort by penalty points (highest first)
        penalty_cases.sort(key=lambda x: x.penalty_points, reverse=True)

        print(f"Found {len(penalty_cases)} CustomGPT penalty cases")
        return penalty_cases

    def analyze_penalty_patterns(self, penalty_cases: List[PenaltyCase]) -> PenaltyAnalysis:
        """Analyze patterns in penalty cases for root cause identification"""

        print(f"\\n=== ANALYZING PENALTY PATTERNS ===")

        # Categorize penalties
        penalty_categories = Counter([case.penalty_type for case in penalty_cases])

        # Domain analysis
        domain_penalties = Counter([case.question_domain for case in penalty_cases])

        # Complexity analysis
        complexity_penalties = defaultdict(list)
        for case in penalty_cases:
            complexity_range = self._get_complexity_range(case.question_complexity)
            complexity_penalties[complexity_range].append(case.penalty_points)

        # Root cause analysis
        root_causes = self._identify_root_causes(penalty_cases)

        # Improvement opportunities
        improvement_opportunities = self._identify_improvement_opportunities(penalty_cases, root_causes)

        return PenaltyAnalysis(
            total_penalties=len(penalty_cases),
            penalty_categories=dict(penalty_categories),
            domain_penalties=dict(domain_penalties),
            complexity_penalties=dict(complexity_penalties),
            root_causes=root_causes,
            improvement_opportunities=improvement_opportunities
        )

    def _get_complexity_range(self, complexity: float) -> str:
        """Categorize complexity into ranges"""
        if complexity <= 0.5:
            return "Simple (0.0-0.5)"
        elif complexity <= 1.0:
            return "Moderate (0.5-1.0)"
        elif complexity <= 1.5:
            return "Complex (1.0-1.5)"
        else:
            return "Very Complex (1.5+)"

    def _identify_root_causes(self, penalty_cases: List[PenaltyCase]) -> Dict[str, int]:
        """Identify root causes of penalty cases through detailed analysis"""

        root_causes = defaultdict(int)

        for case in penalty_cases:

            # Analyze against target answer for factual correctness
            causes = self._analyze_answer_quality(case)
            for cause in causes:
                root_causes[cause] += 1

            # Compare with other providers
            if case.openai_rag_grade and case.openai_vanilla_grade:
                comparative_analysis = self._compare_provider_performance(case)
                for cause in comparative_analysis:
                    root_causes[cause] += 1

            # Analyze judge reasoning patterns
            reasoning_analysis = self._analyze_judge_reasoning(case)
            for cause in reasoning_analysis:
                root_causes[cause] += 1

            # Abstention analysis
            if case.abstention_judge_recommendation:
                abstention_analysis = self._analyze_abstention_decision(case)
                for cause in abstention_analysis:
                    root_causes[cause] += 1

        return dict(root_causes)

    def _analyze_answer_quality(self, case: PenaltyCase) -> List[str]:
        """Analyze the quality issues in CustomGPT's answer"""

        causes = []

        answer = case.customgpt_answer.lower()
        target = case.target_answer.lower()

        # Factual accuracy analysis - for ALL incorrect grades (B only)
        if case.customgpt_grade == 'B':
            # Check for common error patterns
            if "i don't know" in answer or 'not sure' in answer:
                causes.append('knowledge_gap_but_attempted')
            elif len(answer) < 10:
                causes.append('insufficient_detail')
            elif 'approximately' in answer or 'around' in answer or 'about' in answer:
                causes.append('imprecise_when_precision_needed')
            else:
                causes.append('factual_error')

        # Confidence calibration issues - analyze confidence for all incorrect answers
        if case.customgpt_grade == 'B':
            if case.customgpt_confidence > 0.9:
                causes.append('high_confidence_but_wrong')
            elif case.customgpt_confidence > 0.8:
                causes.append('overconfident_incorrect')
            # Note: Low confidence wrong answers are still wrong but less problematic

        # Length analysis
        if len(case.customgpt_answer) > 500:
            causes.append('overly_verbose')
        elif len(case.customgpt_answer) < 20:
            causes.append('too_brief')

        # Format issues
        if case.question_domain == 'mathematics' and not any(char.isdigit() for char in answer):
            causes.append('missing_numerical_answer')

        return causes

    def _compare_provider_performance(self, case: PenaltyCase) -> List[str]:
        """Compare CustomGPT performance with other providers"""

        causes = []

        # Compare grades - A is correct, B is incorrect
        rag_better = case.openai_rag_grade and case.openai_rag_grade == 'A' and case.customgpt_grade == 'B'
        vanilla_better = case.openai_vanilla_grade and case.openai_vanilla_grade == 'A' and case.customgpt_grade == 'B'

        if rag_better and vanilla_better:
            causes.append('both_competitors_outperformed')
        elif rag_better:
            causes.append('openai_rag_outperformed')
        elif vanilla_better:
            causes.append('openai_vanilla_outperformed')

        # RAG-specific analysis
        if rag_better:
            causes.append('rag_knowledge_base_advantage')

        return causes

    def _analyze_judge_reasoning(self, case: PenaltyCase) -> List[str]:
        """Analyze judge reasoning for insights"""

        causes = []
        reasoning = case.judge_reasoning.lower()

        # Common judge reasoning patterns
        if 'incorrect' in reasoning or 'wrong' in reasoning:
            causes.append('judge_found_factual_error')
        if 'incomplete' in reasoning or 'partial' in reasoning:
            causes.append('judge_found_incomplete_answer')
        if 'inaccurate' in reasoning or 'imprecise' in reasoning:
            causes.append('judge_found_precision_issue')
        if 'outdated' in reasoning or 'old' in reasoning:
            causes.append('judge_found_outdated_info')
        if 'misleading' in reasoning:
            causes.append('judge_found_misleading_info')

        return causes

    def _analyze_abstention_decision(self, case: PenaltyCase) -> List[str]:
        """Analyze abstention judge recommendations"""

        causes = []

        if case.abstention_judge_recommendation == 'abstain':
            causes.append('should_have_abstained')
            if case.abstention_judge_confidence and case.abstention_judge_confidence > 0.8:
                causes.append('high_confidence_abstention_recommendation')
        elif case.abstention_judge_recommendation == 'attempt':
            causes.append('abstention_judge_approved_attempt')

        return causes

    def _identify_improvement_opportunities(self, penalty_cases: List[PenaltyCase],
                                         root_causes: Dict[str, int]) -> List[str]:
        """Identify specific improvement opportunities for CustomGPT"""

        opportunities = []
        total_cases = len(penalty_cases)

        # High-frequency issues (>20% of cases)
        high_freq_threshold = total_cases * 0.2

        for cause, count in sorted(root_causes.items(), key=lambda x: x[1], reverse=True):
            if count >= high_freq_threshold:
                opportunity = self._get_improvement_recommendation(cause, count, total_cases)
                if opportunity:
                    opportunities.append(opportunity)

        # Domain-specific improvements
        domain_analysis = Counter([case.question_domain for case in penalty_cases])
        if domain_analysis:
            worst_domain = domain_analysis.most_common(1)[0]
            opportunities.append(f"Focus knowledge base improvement on {worst_domain[0]} domain ({worst_domain[1]} penalties)")

        # Confidence calibration
        overconfident_cases = [c for c in penalty_cases if c.penalty_type == 'overconfident_incorrect']
        if len(overconfident_cases) > total_cases * 0.3:
            opportunities.append("Implement confidence calibration training to reduce overconfidence")

        return opportunities

    def _get_improvement_recommendation(self, cause: str, count: int, total: int) -> Optional[str]:
        """Get specific improvement recommendation for a root cause"""

        percentage = (count / total) * 100

        recommendations = {
            'factual_error': f"Improve factual accuracy - {percentage:.1f}% of penalties from factual errors",
            'knowledge_gap_but_attempted': f"Enhance abstention logic - {percentage:.1f}% attempted answers despite knowledge gaps",
            'severe_overconfidence': f"Critical: Recalibrate confidence scoring - {percentage:.1f}% severe overconfidence cases",
            'rag_knowledge_base_advantage': f"Expand knowledge base - OpenAI RAG outperformed in {percentage:.1f}% of penalty cases",
            'judge_found_factual_error': f"Strengthen fact verification - Judge identified factual errors in {percentage:.1f}% of cases",
            'should_have_abstained': f"Improve abstention strategy - {percentage:.1f}% should have abstained per abstention judge",
            'overly_verbose': f"Optimize response length - {percentage:.1f}% penalties from excessive verbosity",
            'imprecise_when_precision_needed': f"Enhance precision for specific domains - {percentage:.1f}% penalties from imprecision"
        }

        return recommendations.get(cause)

    def generate_detailed_penalty_report(self, penalty_cases: List[PenaltyCase],
                                       analysis: PenaltyAnalysis) -> str:
        """Generate comprehensive penalty analysis report for CustomGPT team"""

        report = []
        report.append("# CUSTOMGPT PENALTY DEEP-DIVE ANALYSIS")
        report.append("## Engineering Team Investigation Report")
        report.append("=" * 80)
        report.append("")

        # Executive Summary
        report.append("## EXECUTIVE SUMMARY")
        report.append("")
        report.append(f"**Total Penalty Cases:** {analysis.total_penalties}")
        report.append(f"**Most Common Penalty Type:** {max(analysis.penalty_categories.items(), key=lambda x: x[1])[0] if analysis.penalty_categories else 'N/A'}")
        report.append(f"**Worst Performing Domain:** {max(analysis.domain_penalties.items(), key=lambda x: x[1])[0] if analysis.domain_penalties else 'N/A'}")
        report.append(f"**Top Root Cause:** {max(analysis.root_causes.items(), key=lambda x: x[1])[0] if analysis.root_causes else 'N/A'}")
        report.append("")

        # Priority Improvements
        report.append("### 🔥 PRIORITY IMPROVEMENTS")
        report.append("")
        for i, opportunity in enumerate(analysis.improvement_opportunities[:5], 1):
            report.append(f"{i}. {opportunity}")
        report.append("")

        # Penalty Categories Breakdown
        report.append("## PENALTY CATEGORIES ANALYSIS")
        report.append("")
        for category, count in sorted(analysis.penalty_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis.total_penalties) * 100
            report.append(f"**{category.replace('_', ' ').title()}:** {count} cases ({percentage:.1f}%)")
        report.append("")

        # Domain Analysis
        report.append("## DOMAIN PERFORMANCE ANALYSIS")
        report.append("")
        for domain, count in sorted(analysis.domain_penalties.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis.total_penalties) * 100
            report.append(f"**{domain.title()}:** {count} penalties ({percentage:.1f}%)")
        report.append("")

        # Root Cause Analysis
        report.append("## ROOT CAUSE ANALYSIS")
        report.append("")
        for cause, count in sorted(analysis.root_causes.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / analysis.total_penalties) * 100
            report.append(f"**{cause.replace('_', ' ').title()}:** {count} cases ({percentage:.1f}%)")
        report.append("")

        # Detailed Case Analysis - Top 20 Penalty Cases
        report.append("## 🔍 DETAILED CASE ANALYSIS")
        report.append("### Top 20 Highest Penalty Cases")
        report.append("")

        for i, case in enumerate(penalty_cases[:20], 1):
            report.append(f"### Case #{i} - Question ID: {case.question_id}")
            report.append(f"**Penalty Points:** {case.penalty_points:.2f}")
            report.append(f"**Penalty Type:** {case.penalty_type}")
            report.append(f"**Domain:** {case.question_domain} | **Complexity:** {case.question_complexity:.2f}")
            report.append("")

            report.append(f"**Question:** {case.question}")
            report.append(f"**Target Answer:** {case.target_answer}")
            report.append("")

            report.append(f"**CustomGPT Answer:** {case.customgpt_answer}")
            report.append(f"**CustomGPT Grade:** {case.customgpt_grade}")
            report.append(f"**CustomGPT Confidence:** {case.customgpt_confidence:.3f}")
            report.append("")

            # Competitor comparison
            if case.openai_rag_answer:
                report.append(f"**OpenAI RAG Answer:** {case.openai_rag_answer}")
                report.append(f"**OpenAI RAG Grade:** {case.openai_rag_grade}")
                report.append("")

            if case.openai_vanilla_answer:
                report.append(f"**OpenAI Vanilla Answer:** {case.openai_vanilla_answer}")
                report.append(f"**OpenAI Vanilla Grade:** {case.openai_vanilla_grade}")
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

            # Engineering insights
            report.append("**🔧 Engineering Insights:**")
            insights = self._get_engineering_insights(case)
            for insight in insights:
                report.append(f"- {insight}")
            report.append("")
            report.append("-" * 60)
            report.append("")

        # Comparative Analysis
        report.append("## 📊 COMPETITIVE ANALYSIS")
        report.append("")

        # Cases where both competitors outperformed
        both_better = [case for case in penalty_cases
                      if case.openai_rag_grade and case.openai_vanilla_grade
                      and case.openai_rag_grade in ['A', 'B']
                      and case.openai_vanilla_grade in ['A', 'B']
                      and case.customgpt_grade not in ['A', 'B']]

        if both_better:
            report.append(f"### Critical Cases: Both Competitors Outperformed ({len(both_better)} cases)")
            report.append("")
            for case in both_better[:5]:  # Top 5
                report.append(f"**Q{case.question_id}:** {case.question[:100]}...")
                report.append(f"- CustomGPT: Grade {case.customgpt_grade}, OpenAI RAG: Grade {case.openai_rag_grade}, OpenAI Vanilla: Grade {case.openai_vanilla_grade}")
                report.append("")

        # Knowledge base gaps (where RAG helped)
        rag_advantage = [case for case in penalty_cases
                        if case.openai_rag_grade and case.openai_rag_grade in ['A', 'B']
                        and case.customgpt_grade not in ['A', 'B']]

        if rag_advantage:
            report.append(f"### Knowledge Base Gaps: OpenAI RAG Advantage ({len(rag_advantage)} cases)")
            report.append("")
            domains = Counter([case.question_domain for case in rag_advantage])
            for domain, count in domains.most_common():
                report.append(f"- **{domain.title()}:** {count} cases")
            report.append("")

        # Implementation Recommendations
        report.append("## 🚀 IMPLEMENTATION RECOMMENDATIONS")
        report.append("")

        report.append("### Immediate Actions (High Priority)")
        immediate_actions = self._get_immediate_actions(penalty_cases, analysis)
        for i, action in enumerate(immediate_actions, 1):
            report.append(f"{i}. {action}")
        report.append("")

        report.append("### Medium-Term Improvements")
        medium_term = self._get_medium_term_improvements(penalty_cases, analysis)
        for i, improvement in enumerate(medium_term, 1):
            report.append(f"{i}. {improvement}")
        report.append("")

        report.append("### Long-Term Strategic Initiatives")
        long_term = self._get_long_term_initiatives(penalty_cases, analysis)
        for i, initiative in enumerate(long_term, 1):
            report.append(f"{i}. {initiative}")
        report.append("")

        return "\\n".join(report)

    def _get_engineering_insights(self, case: PenaltyCase) -> List[str]:
        """Generate specific engineering insights for a penalty case"""

        insights = []

        # Confidence calibration insights
        if case.penalty_type == 'overconfident_incorrect':
            if case.customgpt_confidence > 0.95:
                insights.append("🚨 Critical overconfidence - review confidence calculation algorithm")
            else:
                insights.append("⚠️ Moderate overconfidence - tune confidence threshold")

        # Knowledge base insights
        if case.openai_rag_grade and case.openai_rag_grade in ['A', 'B'] and case.customgpt_grade not in ['A', 'B']:
            insights.append("📚 Knowledge gap - competitor RAG system had better information")

        # Abstention insights
        if case.abstention_judge_recommendation == 'abstain':
            insights.append("🛑 Should have abstained - improve uncertainty detection")

        # Domain-specific insights
        if case.question_domain in ['science', 'mathematics']:
            insights.append("🔬 Technical domain - verify scientific/mathematical accuracy")

        # Answer quality insights
        if len(case.customgpt_answer) > 300 and case.customgpt_grade in ['D', 'F']:
            insights.append("📝 Verbose but incorrect - focus on accuracy over length")
        elif len(case.customgpt_answer) < 20:
            insights.append("📏 Too brief - may need more detailed responses")

        return insights

    def _get_immediate_actions(self, penalty_cases: List[PenaltyCase], analysis: PenaltyAnalysis) -> List[str]:
        """Get immediate action items"""

        actions = []

        if len(penalty_cases) == 0:
            actions.append("🎉 Excellent performance - no significant penalty cases detected!")
            actions.append("Continue current configuration and monitoring for consistency")
            return actions

        # Confidence calibration
        overconfident_rate = len([c for c in penalty_cases if c.penalty_type == 'overconfident_incorrect']) / len(penalty_cases)
        if overconfident_rate > 0.3:
            actions.append("Implement confidence calibration adjustment - 30%+ cases show overconfidence")

        # High-frequency domains
        if analysis.domain_penalties:
            worst_domain = max(analysis.domain_penalties.items(), key=lambda x: x[1])
            if worst_domain[1] > len(penalty_cases) * 0.25:
                actions.append(f"Review and expand {worst_domain[0]} knowledge base - {worst_domain[1]} penalty cases")

        # Abstention logic
        should_abstain = len([c for c in penalty_cases if c.abstention_judge_recommendation == 'abstain'])
        if should_abstain > len(penalty_cases) * 0.2:
            actions.append(f"Improve abstention decision logic - {should_abstain} cases should have abstained")

        return actions

    def _get_medium_term_improvements(self, penalty_cases: List[PenaltyCase], analysis: PenaltyAnalysis) -> List[str]:
        """Get medium-term improvement recommendations"""

        improvements = []

        improvements.append("Implement multi-stage validation for high-confidence answers")
        improvements.append("Develop domain-specific accuracy validation")
        improvements.append("Create answer quality assessment pipeline")
        improvements.append("Implement competitive benchmarking against OpenAI systems")

        return improvements

    def _get_long_term_initiatives(self, penalty_cases: List[PenaltyCase], analysis: PenaltyAnalysis) -> List[str]:
        """Get long-term strategic initiatives"""

        initiatives = []

        initiatives.append("Develop advanced uncertainty quantification system")
        initiatives.append("Implement knowledge base continuous improvement pipeline")
        initiatives.append("Create real-time answer quality monitoring")
        initiatives.append("Develop competitive intelligence system for provider comparison")

        return initiatives

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete CustomGPT penalty analysis"""

        print(f"\\n{'='*60}")
        print(f"CUSTOMGPT PENALTY DEEP-DIVE ANALYSIS")
        print(f"Run ID: {self.run_dir.name}")
        print(f"{'='*60}")

        # 1. Identify all penalty cases
        penalty_cases = self.identify_penalty_cases()

        # 2. Analyze penalty patterns
        analysis = self.analyze_penalty_patterns(penalty_cases)

        # 3. Generate comprehensive results
        results = {
            'metadata': {
                'run_id': self.run_dir.name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_evaluations': len(self.evaluations),
                'total_penalty_cases': len(penalty_cases)
            },
            'penalty_cases': [
                {
                    'question_id': case.question_id,
                    'question': case.question,
                    'target_answer': case.target_answer,
                    'customgpt_answer': case.customgpt_answer,
                    'customgpt_grade': case.customgpt_grade,
                    'customgpt_confidence': case.customgpt_confidence,
                    'penalty_points': case.penalty_points,
                    'penalty_type': case.penalty_type,
                    'domain': case.question_domain,
                    'complexity': case.question_complexity,
                    'judge_reasoning': case.judge_reasoning,
                    'openai_rag_grade': case.openai_rag_grade,
                    'openai_vanilla_grade': case.openai_vanilla_grade,
                    'abstention_recommendation': case.abstention_judge_recommendation
                }
                for case in penalty_cases
            ],
            'analysis': {
                'penalty_categories': analysis.penalty_categories,
                'domain_penalties': analysis.domain_penalties,
                'complexity_penalties': analysis.complexity_penalties,
                'root_causes': analysis.root_causes,
                'improvement_opportunities': analysis.improvement_opportunities
            }
        }

        # 4. Save results and generate report
        self._save_results(results, penalty_cases, analysis)

        return results

    def _save_results(self, results: Dict[str, Any], penalty_cases: List[PenaltyCase], analysis: PenaltyAnalysis):
        """Save comprehensive results and report"""

        # Save JSON results
        json_file = self.output_dir / f"customgpt_penalty_analysis_{self.run_dir.name}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate and save detailed report
        report_content = self.generate_detailed_penalty_report(penalty_cases, analysis)
        report_file = self.output_dir / f"customgpt_penalty_report_{self.run_dir.name}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        # Save CSV for easy analysis
        csv_data = []
        for case in penalty_cases:
            csv_data.append({
                'question_id': case.question_id,
                'penalty_points': case.penalty_points,
                'penalty_type': case.penalty_type,
                'domain': case.question_domain,
                'complexity': case.question_complexity,
                'customgpt_grade': case.customgpt_grade,
                'customgpt_confidence': case.customgpt_confidence,
                'openai_rag_grade': case.openai_rag_grade,
                'openai_vanilla_grade': case.openai_vanilla_grade,
                'should_have_abstained': case.abstention_judge_recommendation == 'abstain'
            })

        csv_file = self.output_dir / f"customgpt_penalty_cases_{self.run_dir.name}.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)

        print(f"\\n=== CUSTOMGPT PENALTY ANALYSIS COMPLETE ===")
        print(f"JSON results: {json_file}")
        print(f"Detailed report: {report_file}")
        print(f"CSV data: {csv_file}")
        print(f"Total penalty cases: {len(penalty_cases)}")
        if penalty_cases:
            print(f"Highest penalty: {max([c.penalty_points for c in penalty_cases]):.2f} points")
        else:
            print("No penalty cases found - excellent performance!")

def main():
    parser = argparse.ArgumentParser(description='CustomGPT Penalty Deep-Dive Analysis')
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--output-dir', default='customgpt_penalty_analysis', help='Output directory')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CustomGPTPenaltyAnalyzer(args.run_dir, args.output_dir)

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    # Print summary
    print(f"\\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total penalty cases: {results['metadata']['total_penalty_cases']}")

    if results['analysis']['penalty_categories']:
        most_common = max(results['analysis']['penalty_categories'].items(), key=lambda x: x[1])
        print(f"Most common penalty type: {most_common[0]} ({most_common[1]} cases)")

    if results['analysis']['domain_penalties']:
        worst_domain = max(results['analysis']['domain_penalties'].items(), key=lambda x: x[1])
        print(f"Worst performing domain: {worst_domain[0]} ({worst_domain[1]} penalties)")

if __name__ == "__main__":
    main()