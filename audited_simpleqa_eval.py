"""
Enhanced SimpleQA Evaluation with Comprehensive Audit Logging
Uses GPT-4.1 as LLM-As-A-Judge with full explainability
"""

import random
import re
import time
import uuid
from typing import Dict, List

import pandas
import requests
import io

from custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from sampler.chat_completion_sampler import ChatCompletionSampler
import common


ENHANCED_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

I will first give examples of each grade, then you will grade a new example. Your response MUST include:
1. A detailed reasoning section explaining your analysis
2. A final grade (A, B, or C)

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.

Additional grading guidelines:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
- Do not punish for typos in people's name if it's clearly the same name.

Here is the new example to grade:
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

ANALYSIS AND REASONING:
First, analyze the predicted answer step by step:
1. Does the predicted answer contain the key information from the gold target?
2. Does the predicted answer contradict any information in the gold target?
3. Is the answer attempting to provide the requested information or declining to answer?
4. Consider the specific grading guidelines above.

Provide your detailed reasoning here, explaining your analysis of the predicted answer against the gold target.

FINAL GRADE:
Based on my analysis above, I grade this predicted answer as:

A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no additional text after the grade.
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


class AuditedSimpleQAEval(Eval):
    """
    Enhanced SimpleQA evaluation with comprehensive audit logging
    """

    def __init__(
        self,
        grader_model: SamplerBase = None,
        num_examples: int | None = None,
        n_repeats: int = 1,
        audit_logger=None
    ):
        # Use GPT-4.1 as default grader
        if grader_model is None:
            grader_model = ChatCompletionSampler(
                model="gpt-4.1",  # GPT-4.1
                temperature=0.0
            )

        with requests.get("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv") as response:
            blob_file = io.BytesIO(response.content)
            df = pandas.read_csv(blob_file)

        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)

        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.audit_logger = audit_logger

        # Add question IDs for tracking
        for i, example in enumerate(self.examples):
            example['question_id'] = f"simpleqa_{i:04d}"

    def grade_sample_with_explanation(
        self,
        question_id: str,
        question: str,
        target: str,
        predicted_answer: str,
        provider_name: str
    ) -> Dict[str, any]:
        """
        Grade a sample with full explanation and audit logging
        """
        start_time = time.time()

        grader_prompt = ENHANCED_GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]

        try:
            grading_response = self.grader_model(prompt_messages)
            latency_ms = (time.time() - start_time) * 1000

            # Extract reasoning and grade
            reasoning = ""
            grade_letter = "C"  # Default to NOT_ATTEMPTED

            # Look for the reasoning section
            reasoning_match = re.search(r"ANALYSIS AND REASONING:\s*(.*?)\s*FINAL GRADE:", grading_response, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            # Extract the final grade
            grade_match = re.search(r"FINAL GRADE:.*?([ABC])", grading_response, re.DOTALL)
            if grade_match:
                grade_letter = grade_match.group(1)
            else:
                # Fallback: look for any A, B, or C
                fallback_match = re.search(r"(A|B|C)", grading_response)
                if fallback_match:
                    grade_letter = fallback_match.group(0)

            # Log the judge evaluation if audit logger is available
            if self.audit_logger:
                self.audit_logger.log_judge_evaluation(
                    question_id=question_id,
                    question=question,
                    target_answer=target,
                    provider_responses={provider_name: predicted_answer},
                    judge_prompt=grader_prompt,
                    judge_response=grading_response,
                    grades={provider_name: CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")},
                    reasoning=reasoning,
                    latency_ms=latency_ms
                )

            return {
                "grade_letter": grade_letter,
                "grade_string": CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED"),
                "reasoning": reasoning,
                "full_response": grading_response,
                "latency_ms": latency_ms
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            if self.audit_logger:
                self.audit_logger.log_error(
                    component="simpleqa_grader",
                    error=str(e),
                    context={
                        "question_id": question_id,
                        "question": question,
                        "provider": provider_name
                    }
                )

            # Return default grade with error info
            return {
                "grade_letter": "C",
                "grade_string": "NOT_ATTEMPTED",
                "reasoning": f"Grading failed due to error: {str(e)}",
                "full_response": "",
                "latency_ms": latency_ms,
                "error": str(e)
            }

    def __call__(self, sampler: SamplerBase, provider_name: str = None) -> EvalResult:
        """
        Run the evaluation with comprehensive audit logging
        """
        if provider_name is None:
            provider_name = sampler.__class__.__name__.replace("Sampler", "").replace("Audited", "")

        def fn(row: dict):
            question_id = row.get("question_id", f"q_{uuid.uuid4().hex[:8]}")
            question = row.get("problem", "")
            target = row.get("answer", "")

            prompt_messages = [
                sampler._pack_message(content=question, role="user")
            ]

            # Call the sampler (which will handle its own audit logging if it's an audited sampler)
            if hasattr(sampler, '__call__') and 'question_id' in sampler.__call__.__code__.co_varnames:
                response_text = sampler(prompt_messages, question_id=question_id)
            else:
                response_text = sampler(prompt_messages)

            # Grade the response with full explanation
            grading_result = self.grade_sample_with_explanation(
                question_id=question_id,
                question=question,
                target=target,
                predicted_answer=response_text,
                provider_name=provider_name
            )

            # Metrics based on grading response
            is_correct = grading_result["grade_letter"] == "A"
            is_incorrect = grading_result["grade_letter"] == "B"
            is_not_attempted = grading_result["grade_letter"] == "C"

            score = is_correct

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=target,
                extracted_answer=response_text,
                grading_explanation=grading_result["reasoning"],
                grade=grading_result["grade_string"]
            )

            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "is_correct": is_correct,
                    "is_incorrect": is_incorrect,
                    "is_not_attempted": is_not_attempted,
                    "judge_latency_ms": grading_result["latency_ms"]
                    # Note: question_id and grading_reasoning are stored in HTML/convo, not metrics
                    # to avoid breaking NumPy aggregation which expects only numeric values
                }
            )

        # Update audit logger with total questions
        if self.audit_logger:
            self.audit_logger.update_progress(0, len(self.examples))

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples)

        # Update final progress
        if self.audit_logger:
            self.audit_logger.update_progress(len(results))

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
            "is_not_attempted": sum(result.metrics["is_not_attempted"] for result in results) / len(results),
        }
        aggregate_metrics["is_given_attempted"] = aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]

        # Calculate accuracy_given_attempted
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"] / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )

        # Add judge performance metrics
        judge_latencies = [r.metrics["judge_latency_ms"] for r in results if "judge_latency_ms" in r.metrics]
        if judge_latencies:
            aggregate_metrics["avg_judge_latency_ms"] = sum(judge_latencies) / len(judge_latencies)
            aggregate_metrics["total_judge_time_s"] = sum(judge_latencies) / 1000

        print("AGGREGATE METRICS")
        print(aggregate_metrics)
        print("##################")

        output_d = {
            "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
            "f1": (
                2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
                / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
                if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
                else 0
            )
        }

        print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {output_d['f1']:.3f}")

        # Get the aggregated results from common.py
        base_result = common.aggregate_results(results)

        # Add our calculated aggregate metrics to the final result
        base_result.metrics.update(aggregate_metrics)
        base_result.metrics.update(output_d)

        return base_result