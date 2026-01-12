#!/usr/bin/env python3
"""
CustomGPT Explainability Post-Mortem Analysis

Analyzes failed queries from SimpleQA benchmarks using CustomGPT's new
Explainability API (Verify Responses feature) to identify root causes
and generate actionable engineering reports.

Usage:
    # Analyze all failures from a benchmark run
    python scripts/explainability_postmortem.py --run-id run_20251214_152848_133

    # Analyze a single question
    python scripts/explainability_postmortem.py --run-id run_20251214_152848_133 --question-id simpleqa_0099

    # Dry run (show what would be analyzed)
    python scripts/explainability_postmortem.py --run-id run_20251214_152848_133 --dry-run
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sampler.explainability_client import ExplainabilityClient, ExplainabilityResult


# =============================================================================
# Root Cause Categories
# =============================================================================

class RootCauseCategory:
    """Failure taxonomy based on explainability signals"""
    HALLUCINATION = "hallucination"
    PARTIAL_KNOWLEDGE = "partial_knowledge"
    KB_GAP = "kb_gap"
    RETRIEVAL_MISS = "retrieval_miss"
    REASONING_ERROR = "reasoning_error"
    SPECIFICITY_FAILURE = "specificity_failure"
    UNKNOWN = "unknown"


ROOT_CAUSE_DESCRIPTIONS = {
    RootCauseCategory.HALLUCINATION: "Claim made without KB source - model fabricated information",
    RootCauseCategory.PARTIAL_KNOWLEDGE: "KB has related but incomplete info - some claims sourced, key claim unsourced",
    RootCauseCategory.KB_GAP: "Information doesn't exist in KB - no citations or sources found",
    RootCauseCategory.RETRIEVAL_MISS: "KB has info but wasn't retrieved - low trust score despite KB coverage",
    RootCauseCategory.REASONING_ERROR: "Retrieved right info, concluded wrongly - sources present but conclusion incorrect",
    RootCauseCategory.SPECIFICITY_FAILURE: "Got general answer, needed specific - partial answer (e.g., 'July 2023' vs '15 July 2023')",
    RootCauseCategory.UNKNOWN: "Unable to determine root cause from available data",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailedQuery:
    """A failed query from the benchmark"""
    question_id: str
    question: str
    target_answer: str
    customgpt_answer: str
    grade: str
    judge_reasoning: str
    judge_confidence: float

    # API context (from provider_requests.jsonl)
    project_id: str
    session_id: str
    prompt_id: int
    citations: List[int] = field(default_factory=list)
    latency_ms: float = 0.0

    # Competitor results
    competitor_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Penalty info
    penalty_points: float = 4.0
    penalty_type: str = "incorrect_answer"


@dataclass
class RootCause:
    """Root cause analysis result"""
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    evidence_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Recommendations:
    """Actionable recommendations"""
    kb_remediation: str = ""
    retrieval_improvement: str = ""
    response_quality: str = ""
    priority: str = "MEDIUM"
    expected_impact: str = ""


@dataclass
class PostMortemResult:
    """Complete post-mortem analysis result for a single query"""
    question_id: str
    question: str
    target_answer: str
    customgpt_answer: str
    grade: str
    judge_reasoning: str
    judge_confidence: float

    api_context: Dict[str, Any] = field(default_factory=dict)
    explainability: Optional[Dict[str, Any]] = None
    root_cause: Optional[RootCause] = None
    recommendations: Optional[Recommendations] = None
    competitor_context: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    error: Optional[str] = None
    analysis_timestamp: str = ""


# =============================================================================
# Data Loaders
# =============================================================================

def load_penalty_analysis(run_dir: Path) -> List[Dict[str, Any]]:
    """Load failed queries from penalty analysis JSON"""
    penalty_dir = run_dir / "customgpt_penalty_analysis"

    # Check if directory exists
    if not penalty_dir.exists():
        print(f"No CustomGPT penalty analysis directory found: {penalty_dir}")
        print("CustomGPT may have achieved 100% accuracy in this benchmark run.")
        return []

    # Find the penalty analysis file
    penalty_files = list(penalty_dir.glob("customgpt_penalty_analysis_*.json"))
    if not penalty_files:
        print(f"No penalty analysis files found in {penalty_dir}")
        return []

    penalty_file = penalty_files[0]
    print(f"Loading penalty analysis from: {penalty_file}")

    with open(penalty_file) as f:
        data = json.load(f)

    return data.get("penalty_cases", [])


def load_provider_requests(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load provider requests indexed by question_id for CustomGPT"""
    requests_file = run_dir / "provider_requests.jsonl"
    if not requests_file.exists():
        raise FileNotFoundError(f"Provider requests file not found: {requests_file}")

    requests_by_question = {}

    with open(requests_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("provider") == "CustomGPT_RAG":
                    question_id = entry.get("question_id")
                    if question_id:
                        requests_by_question[question_id] = entry
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(requests_by_question)} CustomGPT requests")
    return requests_by_question


def build_failed_queries(
    penalty_cases: List[Dict[str, Any]],
    provider_requests: Dict[str, Dict[str, Any]]
) -> List[FailedQuery]:
    """Build FailedQuery objects by joining penalty data with API metadata"""
    failed_queries = []

    for case in penalty_cases:
        question_id = case.get("question_id")
        request_data = provider_requests.get(question_id, {})
        metadata = request_data.get("metadata", {})

        # Extract API context
        project_id = metadata.get("project_id", "")
        session_id = metadata.get("session_id", "")
        prompt_id = metadata.get("prompt_id", 0)
        citations = metadata.get("citations", [])
        latency_ms = request_data.get("response", {}).get("latency_ms", 0.0)

        # Skip if we don't have the required API context
        if not all([project_id, session_id, prompt_id]):
            print(f"Warning: Missing API context for {question_id}, skipping")
            continue

        failed_query = FailedQuery(
            question_id=question_id,
            question=case.get("question", ""),
            target_answer=case.get("target_answer", ""),
            customgpt_answer=case.get("customgpt_answer", ""),
            grade=case.get("customgpt_grade", "B"),
            judge_reasoning=case.get("judge_reasoning", ""),
            judge_confidence=case.get("judge_confidence", 0.0),
            project_id=str(project_id),
            session_id=str(session_id),
            prompt_id=int(prompt_id),
            citations=citations or [],
            latency_ms=latency_ms,
            competitor_results=case.get("competitor_results", {}),
            penalty_points=case.get("penalty_points", 4.0),
            penalty_type=case.get("penalty_type", "incorrect_answer")
        )

        failed_queries.append(failed_query)

    return failed_queries


# =============================================================================
# Root Cause Analysis
# =============================================================================

def classify_root_cause(
    failed_query: FailedQuery,
    explainability: ExplainabilityResult
) -> RootCause:
    """
    Classify the root cause of a failure based on explainability data.

    Uses the failure taxonomy:
    - HALLUCINATION: Flagged claims with no source
    - PARTIAL_KNOWLEDGE: Some claims sourced, key claim unsourced
    - KB_GAP: No citations, no sources found
    - RETRIEVAL_MISS: Low trust score despite KB coverage
    - REASONING_ERROR: Sources present but conclusion incorrect
    - SPECIFICITY_FAILURE: Partial answer missing key details
    """
    evidence_chain = []
    categories = []

    # Analyze claims
    total_claims = len(explainability.claims)
    flagged_claims = [c for c in explainability.claims if c.flagged]
    sourced_claims = [c for c in explainability.claims if c.source is not None]

    # Calculate ratios
    flagged_ratio = len(flagged_claims) / total_claims if total_claims > 0 else 0
    sourced_ratio = len(sourced_claims) / total_claims if total_claims > 0 else 0

    # Check for hallucination (flagged claims = no source)
    if flagged_claims:
        categories.append(RootCauseCategory.HALLUCINATION)
        for claim in flagged_claims:
            evidence_chain.append(f"Claim flagged (no source): '{claim.text[:100]}...'")

    # Check for KB gap (no citations at all)
    if not failed_query.citations and not sourced_claims:
        categories.append(RootCauseCategory.KB_GAP)
        evidence_chain.append("No citations returned from CustomGPT API")
        evidence_chain.append("No sourced claims found in explainability analysis")

    # Check for partial knowledge (some sourced, some not)
    if 0 < sourced_ratio < 1 and flagged_claims:
        if RootCauseCategory.PARTIAL_KNOWLEDGE not in categories:
            categories.append(RootCauseCategory.PARTIAL_KNOWLEDGE)
        evidence_chain.append(f"Partial knowledge: {len(sourced_claims)}/{total_claims} claims sourced")

    # Check for specificity failure (common pattern: got month/year but not day)
    target_lower = failed_query.target_answer.lower()
    answer_lower = failed_query.customgpt_answer.lower()

    # Check for date specificity issues
    date_keywords = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
    has_date_in_both = any(m in target_lower and m in answer_lower for m in date_keywords)

    if has_date_in_both:
        # Check if target has a day number that answer is missing
        import re
        target_days = re.findall(r'\b(\d{1,2})\b', failed_query.target_answer)
        answer_days = re.findall(r'\b(\d{1,2})\b', failed_query.customgpt_answer)

        if target_days and not any(d in answer_days for d in target_days):
            categories.append(RootCauseCategory.SPECIFICITY_FAILURE)
            evidence_chain.append(f"Date specificity: target has '{target_days}', answer missing specific day")

    # Check for reasoning error (sources present but wrong conclusion)
    if sourced_claims and not flagged_claims and failed_query.grade == "B":
        categories.append(RootCauseCategory.REASONING_ERROR)
        evidence_chain.append("All claims sourced but answer still incorrect - reasoning/interpretation error")

    # Check trust score for retrieval issues
    if explainability.trust_score:
        trust = explainability.trust_score.overall
        if trust < 0.5 and failed_query.citations:
            categories.append(RootCauseCategory.RETRIEVAL_MISS)
            evidence_chain.append(f"Low trust score ({trust:.2f}) despite {len(failed_query.citations)} citations")

    # Default if no category identified
    if not categories:
        categories.append(RootCauseCategory.UNKNOWN)
        evidence_chain.append("Unable to determine root cause from available data")

    # Calculate confidence based on evidence strength
    confidence = min(0.9, 0.3 + (0.2 * len(evidence_chain)))

    return RootCause(
        primary_category=categories[0],
        secondary_categories=categories[1:] if len(categories) > 1 else [],
        evidence_chain=evidence_chain,
        confidence=confidence
    )


def generate_recommendations(
    failed_query: FailedQuery,
    root_cause: RootCause,
    explainability: ExplainabilityResult
) -> Recommendations:
    """Generate actionable recommendations based on root cause"""

    recs = Recommendations()

    if root_cause.primary_category == RootCauseCategory.HALLUCINATION:
        recs.kb_remediation = f"Add authoritative source covering: '{failed_query.question[:100]}'"
        recs.retrieval_improvement = "Improve citation-to-answer alignment checks"
        recs.response_quality = "Consider abstaining when key facts cannot be sourced"
        recs.priority = "HIGH"
        recs.expected_impact = "Reduce hallucination rate on unsourced claims"

    elif root_cause.primary_category == RootCauseCategory.KB_GAP:
        recs.kb_remediation = f"Gap identified: Add content covering '{failed_query.target_answer}'"
        recs.retrieval_improvement = "Review KB indexing for coverage gaps"
        recs.response_quality = "Abstain when no relevant KB content found"
        recs.priority = "HIGH"
        recs.expected_impact = "Close knowledge base coverage gaps"

    elif root_cause.primary_category == RootCauseCategory.PARTIAL_KNOWLEDGE:
        recs.kb_remediation = "Enrich existing KB articles with more specific details"
        recs.retrieval_improvement = "Improve granularity of indexed content"
        recs.response_quality = "Flag responses when some claims cannot be sourced"
        recs.priority = "MEDIUM"
        recs.expected_impact = "Improve completeness of KB coverage"

    elif root_cause.primary_category == RootCauseCategory.SPECIFICITY_FAILURE:
        recs.kb_remediation = "Add specific dates/numbers/details to KB content"
        recs.retrieval_improvement = "Improve extraction of specific facts from documents"
        recs.response_quality = "Avoid providing partial answers when specifics are required"
        recs.priority = "MEDIUM"
        recs.expected_impact = "Improve precision on specific fact queries"

    elif root_cause.primary_category == RootCauseCategory.REASONING_ERROR:
        recs.kb_remediation = "Review source content for ambiguity or conflicting info"
        recs.retrieval_improvement = "Improve context window for multi-fact reasoning"
        recs.response_quality = "Add cross-validation step for derived conclusions"
        recs.priority = "MEDIUM"
        recs.expected_impact = "Reduce reasoning errors on complex queries"

    elif root_cause.primary_category == RootCauseCategory.RETRIEVAL_MISS:
        recs.kb_remediation = "Review KB indexing and embedding quality"
        recs.retrieval_improvement = "Tune retrieval parameters for better recall"
        recs.response_quality = "Investigate why relevant content wasn't retrieved"
        recs.priority = "HIGH"
        recs.expected_impact = "Improve retrieval accuracy"

    else:
        recs.kb_remediation = "Manual review required"
        recs.priority = "LOW"

    return recs


# =============================================================================
# Post-Mortem Analysis
# =============================================================================

def analyze_single_failure(
    client: ExplainabilityClient,
    failed_query: FailedQuery
) -> PostMortemResult:
    """Run post-mortem analysis on a single failed query"""

    result = PostMortemResult(
        question_id=failed_query.question_id,
        question=failed_query.question,
        target_answer=failed_query.target_answer,
        customgpt_answer=failed_query.customgpt_answer,
        grade=failed_query.grade,
        judge_reasoning=failed_query.judge_reasoning,
        judge_confidence=failed_query.judge_confidence,
        api_context={
            "project_id": failed_query.project_id,
            "session_id": failed_query.session_id,
            "prompt_id": failed_query.prompt_id,
            "citations_returned": failed_query.citations,
            "latency_ms": failed_query.latency_ms
        },
        competitor_context=failed_query.competitor_results,
        analysis_timestamp=datetime.now().isoformat()
    )

    try:
        # Fetch explainability data
        print(f"\nAnalyzing {failed_query.question_id}...")
        print(f"  Project: {failed_query.project_id}")
        print(f"  Session: {failed_query.session_id}")
        print(f"  Prompt: {failed_query.prompt_id}")

        explainability = client.get_explainability(
            failed_query.project_id,
            failed_query.session_id,
            str(failed_query.prompt_id)
        )

        if explainability.error:
            result.error = explainability.error
            return result

        # Convert explainability to dict for storage
        result.explainability = {
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "source": c.source,
                    "flagged": c.flagged,
                    "confidence": c.confidence,
                    "stakeholder_assessments": c.stakeholder_assessments
                }
                for c in explainability.claims
            ],
            "trust_score": {
                "overall": explainability.trust_score.overall if explainability.trust_score else 0,
                "sourced_claims_ratio": explainability.trust_score.sourced_claims_ratio if explainability.trust_score else 0,
                "flagged_claims_count": explainability.trust_score.flagged_claims_count if explainability.trust_score else 0,
            } if explainability.trust_score else None,
            "stakeholder_analysis": explainability.stakeholder_analysis,
            "overall_status": explainability.overall_status,
            "analysis_cost_queries": explainability.analysis_cost_queries,
            "raw_claims_response": explainability.raw_claims_response,
            "raw_trust_response": explainability.raw_trust_response
        }

        # Classify root cause
        root_cause = classify_root_cause(failed_query, explainability)
        result.root_cause = root_cause

        # Generate recommendations
        recommendations = generate_recommendations(failed_query, root_cause, explainability)
        result.recommendations = recommendations

        print(f"  Root cause: {root_cause.primary_category}")
        print(f"  Evidence: {len(root_cause.evidence_chain)} items")

    except Exception as e:
        result.error = f"Analysis failed: {str(e)}"
        print(f"  Error: {result.error}")

    return result


def run_postmortem(
    run_dir: Path,
    question_id: Optional[str] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None
) -> List[PostMortemResult]:
    """Run post-mortem analysis on failed queries"""

    print(f"\n{'='*60}")
    print("CustomGPT Explainability Post-Mortem Analysis")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")

    # Load data
    penalty_cases = load_penalty_analysis(run_dir)
    provider_requests = load_provider_requests(run_dir)

    # Handle case where there are no penalty cases
    if not penalty_cases:
        print("\nNo CustomGPT failures to analyze - CustomGPT achieved 100% accuracy!")
        print("The explainability post-mortem requires failed queries to analyze.")
        return []

    # Build failed query objects
    failed_queries = build_failed_queries(penalty_cases, provider_requests)
    print(f"Found {len(failed_queries)} failed queries with API context")

    # Filter by question_id if specified
    if question_id:
        failed_queries = [q for q in failed_queries if q.question_id == question_id]
        if not failed_queries:
            print(f"No failed query found with ID: {question_id}")
            return []
        print(f"Analyzing single query: {question_id}")

    # Dry run mode
    if dry_run:
        print(f"\n[DRY RUN] Would analyze {len(failed_queries)} queries:")
        for q in failed_queries:
            print(f"  - {q.question_id}: {q.question[:60]}...")
            print(f"    API: project={q.project_id}, session={q.session_id}, prompt={q.prompt_id}")
        print(f"\nEstimated cost: {len(failed_queries) * 4} queries (4 per analysis)")
        return []

    # Initialize client
    client = ExplainabilityClient()

    # Analyze each failure
    results = []
    for i, query in enumerate(failed_queries, 1):
        print(f"\n[{i}/{len(failed_queries)}] {query.question_id}")
        result = analyze_single_failure(client, query)
        results.append(result)

    # Set up output directory
    if output_dir is None:
        output_dir = run_dir / "explainability_postmortem"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_output = output_dir / f"postmortem_results_{timestamp}.json"
    with open(json_output, 'w') as f:
        json.dump(
            {
                "metadata": {
                    "run_id": run_dir.name,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "total_failures_analyzed": len(results),
                    "total_analysis_cost_queries": len(results) * 4
                },
                "results": [
                    {
                        **{k: v for k, v in asdict(r).items() if k not in ['root_cause', 'recommendations']},
                        "root_cause": asdict(r.root_cause) if r.root_cause else None,
                        "recommendations": asdict(r.recommendations) if r.recommendations else None
                    }
                    for r in results
                ]
            },
            f,
            indent=2,
            default=str
        )
    print(f"\nResults saved to: {json_output}")

    # Print summary
    print(f"\n{'='*60}")
    print("Post-Mortem Summary")
    print(f"{'='*60}")

    # Count by root cause
    root_cause_counts = {}
    for r in results:
        if r.root_cause:
            cat = r.root_cause.primary_category
            root_cause_counts[cat] = root_cause_counts.get(cat, 0) + 1

    print("\nRoot Cause Breakdown:")
    for cat, count in sorted(root_cause_counts.items(), key=lambda x: -x[1]):
        desc = ROOT_CAUSE_DESCRIPTIONS.get(cat, "Unknown")
        print(f"  {cat}: {count} ({desc[:50]}...)")

    print(f"\nTotal queries analyzed: {len(results)}")
    print(f"Total analysis cost: {len(results) * 4} queries")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CustomGPT Explainability Post-Mortem Analysis"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g., run_20251214_152848_133) or full path to run directory"
    )
    parser.add_argument(
        "--question-id",
        help="Analyze a single question ID (e.g., simpleqa_0099)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be analyzed without making API calls"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: {run_dir}/explainability_postmortem)"
    )

    args = parser.parse_args()

    # Resolve run directory
    if os.path.isdir(args.run_id):
        run_dir = Path(args.run_id)
    else:
        # Assume it's a run ID under results/
        results_dir = Path(__file__).parent.parent / "results"
        run_dir = results_dir / args.run_id

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Run analysis
    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_postmortem(
        run_dir,
        question_id=args.question_id,
        dry_run=args.dry_run,
        output_dir=output_dir
    )

    if results:
        print(f"\nAnalysis complete. {len(results)} queries processed.")

        # Auto-generate HTML report
        from scripts.generate_explainability_report import generate_report
        json_files = list((run_dir / "explainability_postmortem").glob("postmortem_results_*.json"))
        if json_files:
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                report_path = generate_report(latest_json)
                print(f"HTML report: {report_path}")
            except Exception as e:
                print(f"Warning: Could not generate HTML report: {e}")


if __name__ == "__main__":
    main()
