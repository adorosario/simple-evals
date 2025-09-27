#!/usr/bin/env python3
"""
GPT-5 Powered Failure Analysis Script
Uses advanced reasoning to analyze why providers failed specific questions.

This script provides deep engineering insights for each penalty case,
comparing provider answers with competitors and identifying root causes.
"""

import json
import argparse
import asyncio
import openai
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configure OpenAI for GPT-5 analysis
client = openai.AsyncClient()

async def analyze_failure_with_gpt5(
    question_id: str,
    question: str,
    target_answer: str,
    provider_answer: str,
    provider_confidence: float,
    provider_grade: str,
    openai_rag_answer: str,
    openai_rag_grade: str,
    openai_vanilla_answer: str,
    openai_vanilla_grade: str,
    judge_reasoning: str,
    domain: str,
    complexity: float
) -> Dict[str, Any]:
    """Use GPT-5 to perform deep failure analysis on a specific question"""

    analysis_prompt = f"""You are an expert AI system evaluator analyzing why a RAG provider failed on a specific question. Provide comprehensive engineering insights.

QUESTION DETAILS:
- ID: {question_id}
- Domain: {domain}
- Complexity Score: {complexity:.3f}
- Question: {question}
- Target Answer: {target_answer}

PROVIDER PERFORMANCE:
- Provider Answer: {provider_answer}
- Provider Confidence: {provider_confidence}
- Provider Grade: {provider_grade} (B = incorrect, receives -4.0 penalty)

COMPETITOR COMPARISON:
- OpenAI RAG Answer: {openai_rag_answer}
- OpenAI RAG Grade: {openai_rag_grade}
- OpenAI Vanilla Answer: {openai_vanilla_answer}
- OpenAI Vanilla Grade: {openai_vanilla_grade}

JUDGE EVALUATION:
- Judge Reasoning: {judge_reasoning}

ANALYSIS REQUIREMENTS:
Provide a comprehensive engineering analysis covering:

1. **FAILURE ROOT CAUSE**: What specifically went wrong? Was it:
   - Knowledge base gap (missing information)
   - Retrieval failure (wrong documents retrieved)
   - Reasoning error (misinterpreted information)
   - Confidence miscalibration (overconfident on wrong answer)
   - Factual error in training data
   - Complex reasoning requirement beyond capability

2. **COMPETITIVE ANALYSIS**: How did competitors perform and why?
   - What enabled OpenAI RAG to succeed/fail?
   - What enabled OpenAI Vanilla to succeed/fail?
   - What does this reveal about knowledge sources vs reasoning?

3. **TECHNICAL RECOMMENDATIONS**: Specific engineering fixes:
   - Knowledge base improvements needed
   - Retrieval algorithm adjustments
   - Confidence calibration fixes
   - Training data corrections
   - Reasoning chain improvements

4. **CONFIDENCE ANALYSIS**: Why was the provider confident ({provider_confidence}) despite being wrong?
   - Was the retrieved information misleading?
   - Was the reasoning chain logically sound but factually incorrect?
   - Is this a systematic overconfidence pattern?

5. **PRIORITY ASSESSMENT**: How critical is fixing this failure?
   - High: Fundamental capability gap affecting many questions
   - Medium: Domain-specific knowledge gap
   - Low: Edge case or very specific factual error

Provide actionable insights that an engineering team can use to improve the system.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",  # Using latest available model
            messages=[
                {"role": "system", "content": "You are an expert AI system evaluator providing comprehensive technical analysis for engineering teams."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        return {
            "question_id": question_id,
            "analysis": response.choices[0].message.content,
            "model_used": "gpt-4o",
            "analysis_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "question_id": question_id,
            "analysis": f"Error during analysis: {str(e)}",
            "model_used": "gpt-4o",
            "analysis_timestamp": datetime.now().isoformat(),
            "error": True
        }

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

def load_run_data(run_dir: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load run metadata and judge evaluations"""
    metadata_file = Path(run_dir) / "run_metadata.json"
    judge_file = Path(run_dir) / "judge_evaluations.jsonl"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    evaluations = []
    with open(judge_file, 'r') as f:
        for line in f:
            if line.strip():
                evaluations.append(json.loads(line))

    return metadata, evaluations

def get_competitor_answers(question_id: str, evaluations: List[Dict]) -> Dict[str, Any]:
    """Get OpenAI RAG and vanilla answers for comparison"""
    openai_rag_answer = "Not found"
    openai_rag_grade = "Not found"
    openai_vanilla_answer = "Not found"
    openai_vanilla_grade = "Not found"

    for eval_data in evaluations:
        if eval_data['question_id'] == question_id:
            provider = eval_data['metadata']['real_provider_name']
            try:
                judge_response = json.loads(eval_data['judge']['response'])
                grade = judge_response['grade']
                answer = eval_data['sampler']['response']

                if 'OpenAI_RAG' in provider:
                    openai_rag_answer = answer
                    openai_rag_grade = grade
                elif 'OpenAI_Vanilla' in provider:
                    openai_vanilla_answer = answer
                    openai_vanilla_grade = grade
            except:
                continue

    return {
        'openai_rag_answer': openai_rag_answer,
        'openai_rag_grade': openai_rag_grade,
        'openai_vanilla_answer': openai_vanilla_answer,
        'openai_vanilla_grade': openai_vanilla_grade
    }

async def analyze_all_failures(run_dir: str, provider_name: str = "customgpt") -> Dict[str, Any]:
    """Analyze all penalty cases with GPT-5 reasoning"""

    print(f"=== GPT-5 FAILURE ANALYSIS ===")
    print(f"Run: {Path(run_dir).name}")
    print(f"Provider: {provider_name}")
    print("=" * 50)

    # Load all necessary data
    penalty_analysis = load_penalty_analysis(run_dir, provider_name)
    metadata, evaluations = load_run_data(run_dir)

    penalty_cases = penalty_analysis['penalty_cases']
    print(f"Analyzing {len(penalty_cases)} penalty cases with advanced reasoning...")
    print()

    # Analyze each failure
    failure_analyses = []

    for i, case in enumerate(penalty_cases, 1):
        print(f"Analyzing case {i}/{len(penalty_cases)}: {case['question_id']}")

        # Get competitor answers
        competitor_data = get_competitor_answers(case['question_id'], evaluations)

        # Perform GPT-5 analysis
        analysis = await analyze_failure_with_gpt5(
            question_id=case['question_id'],
            question=case['question'],
            target_answer=case['target_answer'],
            provider_answer=case['customgpt_answer'],
            provider_confidence=case['customgpt_confidence'],
            provider_grade=case['customgpt_grade'],
            openai_rag_answer=competitor_data['openai_rag_answer'],
            openai_rag_grade=competitor_data['openai_rag_grade'],
            openai_vanilla_answer=competitor_data['openai_vanilla_answer'],
            openai_vanilla_grade=competitor_data['openai_vanilla_grade'],
            judge_reasoning=case['judge_reasoning'],
            domain=case['domain'],
            complexity=case['complexity']
        )

        failure_analyses.append(analysis)

        # Brief progress update
        if 'error' not in analysis:
            print(f"  ✓ Analysis completed")
        else:
            print(f"  ✗ Analysis failed: {analysis.get('error', 'Unknown error')}")

    print(f"\nCompleted advanced reasoning analysis of all {len(penalty_cases)} failures.")

    return {
        "metadata": {
            "run_id": Path(run_dir).name,
            "provider": provider_name,
            "total_failures": len(penalty_cases),
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": "gpt-4o"
        },
        "failure_analyses": failure_analyses
    }

async def main():
    parser = argparse.ArgumentParser(description='GPT-5 powered failure analysis for penalty cases')
    parser.add_argument('--run-dir', required=True, help='Path to evaluation run directory')
    parser.add_argument('--provider', default='customgpt', help='Provider name to analyze')
    parser.add_argument('--output', help='Output file path (default: auto-generated in run directory)')

    args = parser.parse_args()

    # Perform analysis
    results = await analyze_all_failures(args.run_dir, args.provider)

    # Save results - default to run directory to keep root clean
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(args.run_dir) / f"{args.provider}_penalty_analysis" / f"gpt5_failure_analysis_{Path(args.run_dir).name}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAdvanced failure analysis saved to: {output_file}")
    print(f"Total failures analyzed: {results['metadata']['total_failures']}")

if __name__ == "__main__":
    asyncio.run(main())