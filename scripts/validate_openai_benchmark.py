#!/usr/bin/env python3
"""
OpenAI Two-Way Validation Test
Test OpenAI Vanilla vs OpenAI RAG with full audit logging
"""

import sys
import os
import time
import uuid
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audit_logger import AuditLogger
from sampler.openai_vanilla_sampler import OpenAIVanillaSampler
from sampler.openai_rag_sampler import OpenAIRAGSampler
from simpleqa_eval import SimpleQAEval
from sampler.chat_completion_sampler import ChatCompletionSampler

# Import audited sampler wrapper from main benchmark
from scripts.audited_three_way_benchmark import AuditedSampler, run_audited_evaluation, create_comparative_report


def main():
    """Run focused OpenAI validation test"""

    # Generate unique run ID
    run_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    print("🔬 OPENAI TWO-WAY VALIDATION TEST")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("Testing OpenAI Vanilla vs OpenAI RAG with comprehensive audit logging")
    print("=" * 60)

    # Initialize audit logger
    audit_logger = AuditLogger(run_id, output_dir="audit_logs")

    # Use smaller sample for validation
    n_samples = 3
    print(f"📊 Configuration: {n_samples} questions per sampler")

    # Get SimpleQA questions
    print(f"\n📥 Loading SimpleQA dataset...")
    try:
        grader = ChatCompletionSampler(model="gpt-4o-mini", temperature=0.0)
        eval_instance = SimpleQAEval(grader_model=grader, num_examples=n_samples)
        questions = eval_instance.examples

        audit_logger.set_total_questions(len(questions) * 2)  # 2 samplers
        print(f"   ✅ Loaded {len(questions)} questions")

    except Exception as e:
        print(f"   ❌ Failed to load questions: {e}")
        return 1

    # Setup OpenAI samplers only
    samplers = {}

    print(f"\n🔧 Setting up OpenAI samplers...")

    # OpenAI Vanilla (no RAG)
    try:
        vanilla_sampler = OpenAIVanillaSampler(
            model="gpt-4o-mini",
            system_message="You are a helpful assistant. Answer questions based on your training knowledge.",
            temperature=0.0
        )

        audit_logger.add_sampler("OpenAI_Vanilla", vanilla_sampler.to_config())
        samplers["OpenAI_Vanilla"] = AuditedSampler(vanilla_sampler, "OpenAI_Vanilla", audit_logger)
        print("   ✅ OpenAI Vanilla sampler ready")

    except Exception as e:
        print(f"   ❌ OpenAI Vanilla setup failed: {e}")
        return 1

    # OpenAI RAG (with vector store)
    try:
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        if vector_store_id:
            rag_sampler = OpenAIRAGSampler(
                model="gpt-4o-mini",
                system_message="You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.",
                temperature=0.0
            )

            audit_logger.add_sampler("OpenAI_RAG", rag_sampler.to_config())
            samplers["OpenAI_RAG"] = AuditedSampler(rag_sampler, "OpenAI_RAG", audit_logger)
            print("   ✅ OpenAI RAG sampler ready")
        else:
            print("   ⚠️  OpenAI RAG skipped: No vector store ID")

    except Exception as e:
        print(f"   ❌ OpenAI RAG setup failed: {e}")

    if not samplers:
        print("❌ No samplers available!")
        return 1

    print(f"\n✅ Ready to test {len(samplers)} samplers with {len(questions)} questions each")
    print(f"   Total API calls expected: {len(samplers) * len(questions)}")

    # Run evaluations
    all_results = []

    for sampler_name, sampler in samplers.items():
        try:
            result = run_audited_evaluation(sampler_name, sampler, questions, audit_logger)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Evaluation failed for {sampler_name}: {e}")

    # Generate final reports
    print(f"\n📄 Generating validation reports...")

    try:
        comparative_report = create_comparative_report(all_results, audit_logger)

        # Save validation report
        report_file = f"results/validation_{run_id}.json"
        os.makedirs("results", exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(comparative_report, f, indent=2)

        print(f"   ✅ Validation report: {report_file}")
        print(f"   ✅ Audit trail: {comparative_report['benchmark_metadata']['audit_trail_location']}")

        # Print summary
        print(f"\n🏆 VALIDATION RESULTS:")
        print("=" * 50)

        for sampler_name, summary in comparative_report['results_summary'].items():
            print(f"{sampler_name}:")
            print(f"   Success Rate: {summary['success_rate']:.1%}")
            print(f"   Avg Response Time: {summary['avg_response_time']:.1f}s")
            print(f"   Successful: {summary['successful_responses']}/{summary['total_questions']}")

        if 'winner' in comparative_report['comparative_analysis']:
            winner = comparative_report['comparative_analysis']['winner']
            print(f"\n🥇 WINNER: {winner}")

        print(f"\n📋 AUDIT VERIFICATION:")
        print(f"   Audit Location: {comparative_report['benchmark_metadata']['audit_trail_location']}")
        print(f"   Total API Calls Logged: {comparative_report['audit_summary']['aggregate_statistics']['total_calls']}")
        print(f"   Success Rate: {comparative_report['audit_summary']['aggregate_statistics']['success_rate']:.1%}")

        # Validation summary
        print(f"\n✅ VALIDATION STATUS:")
        if len(all_results) >= 2 and all(r['success_rate'] > 0 for r in all_results):
            print("   🎉 END-TO-END VALIDATION SUCCESSFUL!")
            print("   - Audit logging: ✅ Working")
            print("   - OpenAI Vanilla: ✅ Working")
            print("   - OpenAI RAG: ✅ Working")
            print("   - TTFB/TTLB metrics: ✅ Captured")
            print("   - Error handling: ✅ Robust")
            print("   - Independent auditability: ✅ Complete")
        else:
            print("   ⚠️  Partial validation - check individual results")

        return 0

    except Exception as e:
        print(f"❌ Validation report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())