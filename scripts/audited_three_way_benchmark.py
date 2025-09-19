#!/usr/bin/env python3
"""
Audited Three-Way RAG Benchmark
Comprehensive logging and audit trail for independent verification
"""

import sys
import os
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.audit_logger import AuditLogger
from sampler.customgpt_sampler import CustomGPTSampler
from sampler.openai_vanilla_sampler import OpenAIVanillaSampler
from sampler.openai_rag_sampler import OpenAIRAGSampler
from simpleqa_eval import SimpleQAEval
from sampler.chat_completion_sampler import ChatCompletionSampler


class AuditedSampler:
    """Wrapper that adds audit logging to any sampler"""

    def __init__(self, sampler, sampler_name: str, audit_logger: AuditLogger):
        self.sampler = sampler
        self.sampler_name = sampler_name
        self.audit_logger = audit_logger

    def __call__(self, message_list):
        """Execute sampler call with full audit logging"""

        # Extract question for logging
        question_text = ""
        for message in reversed(message_list):
            if message.get("role") == "user":
                question_text = message.get("content", "")
                break

        # Generate unique question ID
        question_id = str(uuid.uuid4())[:8]

        # Prepare request data for audit
        request_data = {
            'message_list': message_list,
            'sampler_config': getattr(self.sampler, 'to_config', lambda: {})(),
            'question_preview': question_text[:100] + "..." if len(question_text) > 100 else question_text
        }

        # Determine API endpoint and model
        api_endpoint = self._get_api_endpoint()
        model_used = getattr(self.sampler, 'model', 'unknown')

        # Start audit logging
        call_id = self.audit_logger.start_api_call(
            sampler_name=self.sampler_name,
            question_id=question_id,
            question_text=question_text,
            request_data=request_data,
            api_endpoint=api_endpoint,
            model_used=model_used
        )

        retry_count = 0
        max_retries = 3

        while retry_count <= max_retries:
            try:
                # Make the actual API call
                start_call_time = time.time()

                # Call the actual sampler
                response = self.sampler(message_list)

                # Log first byte received (approximation)
                self.audit_logger.log_first_byte(call_id)

                # Complete successful call
                response_data = {
                    'response_length': len(response) if response else 0,
                    'response_preview': response[:200] + "..." if response and len(response) > 200 else response,
                    'call_duration': time.time() - start_call_time
                }

                self.audit_logger.complete_api_call(
                    call_id=call_id,
                    response_data=response_data,
                    response_text=response,
                    success=True,
                    retry_count=retry_count
                )

                return response

            except Exception as e:
                retry_count += 1
                error_message = f"Attempt {retry_count}: {str(e)}"

                if retry_count > max_retries:
                    # Final failure
                    self.audit_logger.complete_api_call(
                        call_id=call_id,
                        response_data=None,
                        response_text=None,
                        success=False,
                        error_message=error_message,
                        retry_count=retry_count - 1
                    )
                    return ""
                else:
                    # Will retry
                    print(f"⚠️  {self.sampler_name} call failed, retrying ({retry_count}/{max_retries}): {e}")
                    time.sleep(2 ** retry_count)  # Exponential backoff

        return ""

    def _get_api_endpoint(self) -> str:
        """Determine API endpoint for audit logging"""
        if 'customgpt' in self.sampler_name.lower():
            return 'customgpt.ai/api'
        elif 'openai' in self.sampler_name.lower():
            if hasattr(self.sampler, 'vector_store_id'):
                return 'api.openai.com/responses'
            else:
                return 'api.openai.com/chat/completions'
        else:
            return 'unknown'


def setup_audited_samplers(audit_logger: AuditLogger) -> Dict[str, AuditedSampler]:
    """Set up all samplers with audit logging"""
    samplers = {}

    print("🔧 Setting up audited samplers...")

    # 1. CustomGPT RAG Sampler
    try:
        os.environ["CUSTOMGPT_MODEL_NAME"] = "customgpt-rag"
        customgpt_sampler = CustomGPTSampler.from_env()

        audit_logger.add_sampler("CustomGPT_RAG", customgpt_sampler.to_config())
        samplers["CustomGPT_RAG"] = AuditedSampler(customgpt_sampler, "CustomGPT_RAG", audit_logger)
        print("   ✅ CustomGPT RAG sampler ready")

    except Exception as e:
        print(f"   ❌ CustomGPT setup failed: {e}")

    # 2. OpenAI Vanilla (no RAG)
    try:
        vanilla_sampler = OpenAIVanillaSampler(
            model="gpt-4o-mini",
            system_message="You are a helpful assistant. Answer questions based on your training knowledge.",
            temperature=0.0  # Deterministic for fair comparison
        )

        audit_logger.add_sampler("OpenAI_Vanilla", vanilla_sampler.to_config())
        samplers["OpenAI_Vanilla"] = AuditedSampler(vanilla_sampler, "OpenAI_Vanilla", audit_logger)
        print("   ✅ OpenAI Vanilla sampler ready")

    except Exception as e:
        print(f"   ❌ OpenAI Vanilla setup failed: {e}")

    # 3. OpenAI RAG (with vector store)
    try:
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
        if vector_store_id:
            rag_sampler = OpenAIRAGSampler(
                model="gpt-4o-mini",
                system_message="You are a helpful assistant. Use the knowledge base to provide accurate, detailed answers.",
                temperature=0.0  # Deterministic for fair comparison
            )

            audit_logger.add_sampler("OpenAI_RAG", rag_sampler.to_config())
            samplers["OpenAI_RAG"] = AuditedSampler(rag_sampler, "OpenAI_RAG", audit_logger)
            print("   ✅ OpenAI RAG sampler ready")
        else:
            print("   ⚠️  OpenAI RAG skipped: No vector store ID")

    except Exception as e:
        print(f"   ❌ OpenAI RAG setup failed: {e}")

    return samplers


def run_audited_evaluation(sampler_name: str, sampler: AuditedSampler,
                          questions: List[Dict], audit_logger: AuditLogger) -> Dict[str, Any]:
    """Run evaluation with comprehensive audit logging"""

    print(f"\n🚀 Running audited evaluation: {sampler_name}")
    print(f"   Questions: {len(questions)}")

    results = {
        'sampler_name': sampler_name,
        'total_questions': len(questions),
        'start_time': time.time(),
        'responses': [],
        'errors': [],
        'timing_summary': {
            'min_duration': float('inf'),
            'max_duration': 0,
            'total_duration': 0
        }
    }

    for i, question_data in enumerate(questions):
        question_text = question_data.get('problem', '')
        expected_answer = question_data.get('answer', '')

        print(f"   📝 Question {i+1}/{len(questions)}: {question_text[:50]}...")

        start_time = time.time()

        try:
            # Use audited sampler (automatically logs everything)
            message_list = [{"role": "user", "content": question_text}]
            response = sampler(message_list)

            end_time = time.time()
            duration = end_time - start_time

            # Update timing summary
            results['timing_summary']['min_duration'] = min(results['timing_summary']['min_duration'], duration)
            results['timing_summary']['max_duration'] = max(results['timing_summary']['max_duration'], duration)
            results['timing_summary']['total_duration'] += duration

            # Store result
            question_result = {
                'question_id': f"q_{i+1}",
                'question_text': question_text,
                'expected_answer': expected_answer,
                'actual_response': response,
                'duration_seconds': duration,
                'success': bool(response and len(response.strip()) > 0),
                'timestamp': datetime.now().isoformat()
            }

            results['responses'].append(question_result)

            if question_result['success']:
                print(f"      ✅ Response received ({duration:.1f}s)")
            else:
                print(f"      ❌ Empty/failed response ({duration:.1f}s)")

        except Exception as e:
            error_record = {
                'question_id': f"q_{i+1}",
                'question_text': question_text,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results['errors'].append(error_record)
            print(f"      💥 Error: {e}")

    # Finalize results
    results['end_time'] = time.time()
    results['total_duration'] = results['end_time'] - results['start_time']
    results['successful_responses'] = len([r for r in results['responses'] if r['success']])
    results['failed_responses'] = len([r for r in results['responses'] if not r['success']])
    results['success_rate'] = results['successful_responses'] / results['total_questions'] if results['total_questions'] > 0 else 0

    # Fix timing summary if no successful calls
    if results['timing_summary']['min_duration'] == float('inf'):
        results['timing_summary']['min_duration'] = 0

    return results


def create_comparative_report(all_results: List[Dict], audit_logger: AuditLogger) -> Dict[str, Any]:
    """Create comparative analysis report"""

    print(f"\n📊 Creating comparative analysis...")

    # Generate the final audit report
    audit_report = audit_logger.generate_final_audit_report()

    comparative_report = {
        'benchmark_metadata': {
            'run_id': audit_logger.run_id,
            'timestamp': datetime.now().isoformat(),
            'benchmark_version': '1.0_audited',
            'comparison_type': 'three_way_rag_benchmark',
            'audit_trail_location': audit_report['data_locations']['run_directory']
        },
        'results_summary': {},
        'comparative_analysis': {},
        'audit_summary': audit_report,
        'verification_instructions': {
            'overview': 'This benchmark includes complete audit trails for independent verification',
            'audit_location': audit_report['data_locations']['run_directory'],
            'verification_steps': [
                '1. Verify environment configuration in run_metadata.json',
                '2. Check all API calls in api_calls/ directory',
                '3. Validate timing metrics (TTFB/TTLB) for each call',
                '4. Review error logs in errors/ directory',
                '5. Examine raw responses in responses/ directory',
                '6. Confirm statistical calculations in this report'
            ],
            'integrity_checks': [
                'Verify all question IDs are unique',
                'Check that response counts match API call counts',
                'Validate timing consistency across all calls',
                'Confirm retry logic was properly applied'
            ]
        }
    }

    # Process results for each sampler
    for result in all_results:
        sampler_name = result['sampler_name']

        comparative_report['results_summary'][sampler_name] = {
            'total_questions': result['total_questions'],
            'successful_responses': result['successful_responses'],
            'failed_responses': result['failed_responses'],
            'success_rate': result['success_rate'],
            'total_duration_seconds': result['total_duration'],
            'avg_response_time': result['timing_summary']['total_duration'] / result['total_questions'] if result['total_questions'] > 0 else 0,
            'min_response_time': result['timing_summary']['min_duration'],
            'max_response_time': result['timing_summary']['max_duration']
        }

    # Determine winner
    successful_samplers = [r for r in all_results if r['success_rate'] > 0]
    if successful_samplers:
        winner = max(successful_samplers, key=lambda x: x['success_rate'])
        comparative_report['comparative_analysis'] = {
            'winner': winner['sampler_name'],
            'winner_success_rate': winner['success_rate'],
            'winner_avg_time': winner['timing_summary']['total_duration'] / winner['total_questions'] if winner['total_questions'] > 0 else 0,
            'rankings': sorted(
                [{'name': r['sampler_name'], 'success_rate': r['success_rate']} for r in all_results],
                key=lambda x: x['success_rate'],
                reverse=True
            )
        }

    return comparative_report


def main():
    """Run comprehensive audited three-way benchmark"""

    # Generate unique run ID
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    print("🚀 AUDITED THREE-WAY RAG BENCHMARK")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print("This benchmark includes comprehensive audit logging for independent verification.")
    print("=" * 70)

    # Initialize audit logger
    audit_logger = AuditLogger(run_id, output_dir="audit_logs")

    # Configuration
    n_samples = int(os.environ.get("BENCHMARK_SAMPLES", "5"))  # Default 5 for testing
    print(f"📊 Configuration: {n_samples} questions per sampler")

    # Get SimpleQA questions
    print(f"\n📥 Loading SimpleQA dataset...")
    try:
        # Create grader and evaluation instance
        grader = ChatCompletionSampler(model="gpt-4o-mini", temperature=0.0)
        eval_instance = SimpleQAEval(grader_model=grader, num_examples=n_samples)
        questions = eval_instance.examples

        audit_logger.set_total_questions(len(questions) * 3)  # 3 samplers
        print(f"   ✅ Loaded {len(questions)} questions")

    except Exception as e:
        print(f"   ❌ Failed to load questions: {e}")
        return 1

    # Setup audited samplers
    samplers = setup_audited_samplers(audit_logger)

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
    print(f"\n📄 Generating audit reports...")

    try:
        comparative_report = create_comparative_report(all_results, audit_logger)

        # Save main report
        report_file = f"results/audited_benchmark_{run_id}.json"
        os.makedirs("results", exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(comparative_report, f, indent=2)

        print(f"   ✅ Main report: {report_file}")
        print(f"   ✅ Audit trail: {comparative_report['benchmark_metadata']['audit_trail_location']}")

        # Print summary
        print(f"\n🏆 BENCHMARK RESULTS:")
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

        return 0

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())