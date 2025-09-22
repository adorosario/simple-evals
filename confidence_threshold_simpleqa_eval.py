"""
Enhanced SimpleQA Evaluation with Confidence Threshold Framework
Implements the theoretical insights from OpenAI's "Why Language Models Hallucinate" paper
Supports penalty-aware scoring and behavioral calibration evaluation
"""

import random
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas
import requests
import io
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

from custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from sampler.chat_completion_sampler import ChatCompletionSampler
import common


@dataclass
class ConfidenceThreshold:
    """Configuration for a confidence threshold scenario"""
    threshold: float  # 0.5, 0.75, 0.9
    penalty_ratio: float  # k in the paper: wrong=-k, correct=+1, idk=0
    name: str  # "Conservative", "Balanced", "Aggressive"

    @property
    def description(self) -> str:
        return f"Answer only if >{self.threshold*100:.0f}% confident (penalty ratio: {self.penalty_ratio})"


# Predefined confidence threshold configurations based on OpenAI paper recommendations
CONFIDENCE_THRESHOLDS = [
    ConfidenceThreshold(0.5, 1.0, "Balanced"),      # t=0.5, penalty=1
    ConfidenceThreshold(0.75, 3.0, "Conservative"), # t=0.75, penalty=3
    ConfidenceThreshold(0.9, 9.0, "Cautious"),      # t=0.9, penalty=9
]


def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a dataset
    Returns: (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0

    data_array = np.array(data)
    mean = np.mean(data_array)

    # Check if data is binary (only 0s and 1s)
    unique_values = np.unique(data_array)
    is_binary = len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values)

    if is_binary:
        # Use Wilson Score Interval for binary proportions
        return wilson_score_interval(data, confidence_level)
    else:
        # Use t-distribution for continuous data
        std_err = stats.sem(data_array)
        df = len(data) - 1
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin_error = t_critical * std_err
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error
        return mean, lower_bound, upper_bound


def wilson_score_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate Wilson Score Interval for binary proportion data
    More accurate than normal approximation for small samples
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    p = np.mean(data)  # Sample proportion
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha/2)  # Critical value

    # Wilson Score Interval formula
    denominator = 1 + (z**2 / n)
    center = (p + (z**2) / (2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + (z**2)/(4*n)) / n) / denominator

    lower_bound = center - margin
    upper_bound = center + margin

    # Ensure bounds are within [0, 1] for proportions
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)

    return p, lower_bound, upper_bound


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size for two groups
    Returns: effect size (small: 0.2, medium: 0.5, large: 0.8)
    """
    if not group1 or not group2:
        return 0.0

    arr1, arr2 = np.array(group1), np.array(group2)

    # Calculate pooled standard deviation
    n1, n2 = len(arr1), len(arr2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(arr1, ddof=1) + (n2 - 1) * np.var(arr2, ddof=1)) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(arr1) - np.mean(arr2)) / pooled_std


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cramer's V effect size for categorical variables
    Returns: effect size (small: 0.1, medium: 0.3, large: 0.5)
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1

    if min_dim == 0 or n == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def interpret_effect_size(effect_size: float, measure_type: str = "cohens_d") -> str:
    """
    Interpret effect size magnitude
    """
    if measure_type == "cohens_d":
        if abs(effect_size) < 0.2:
            return "negligible"
        elif abs(effect_size) < 0.5:
            return "small"
        elif abs(effect_size) < 0.8:
            return "medium"
        else:
            return "large"
    elif measure_type == "cramers_v":
        if effect_size < 0.1:
            return "negligible"
        elif effect_size < 0.3:
            return "small"
        elif effect_size < 0.5:
            return "medium"
        else:
            return "large"

    return "unknown"


def apply_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons
    Returns: (list of corrected significance results, corrected alpha level)
    """
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons
    corrected_significant = [p < corrected_alpha for p in p_values]
    return corrected_significant, corrected_alpha


CONFIDENCE_THRESHOLD_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

This evaluation is being conducted under the {threshold_name} confidence threshold framework (threshold: {threshold_value:.0%}, penalty ratio: {penalty_ratio}).

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

CONFIDENCE THRESHOLD EVALUATION CONTEXT:
This response is being evaluated under the {threshold_name} threshold framework where:
- Threshold: {threshold_value:.0%} confidence required
- Penalty ratio: {penalty_ratio} (wrong answers receive -{penalty_ratio} points)
- Scoring: Correct = +1, Wrong = -{penalty_ratio}, Not attempted = 0

When grading, assess the natural confidence signals in the response:
- Does the response demonstrate appropriate certainty/uncertainty?
- Would this response be suitable under a {threshold_value:.0%} confidence threshold?
- Consider the penalty ratio of {penalty_ratio} when evaluating risk vs. reward

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
4. What confidence signals does the response contain (certainty, hedging, uncertainty expressions)?
5. Under the {threshold_name} framework (threshold: {threshold_value:.0%}, penalty: {penalty_ratio}), how appropriate is this response?
6. Consider the specific grading guidelines above.

Provide your detailed reasoning here, explaining your analysis of the predicted answer against the gold target and the confidence threshold framework.

FINAL GRADE:
Based on my analysis above, I grade this predicted answer as:

A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no additional text after the grade.
""".strip()


CHOICE_LETTERS = ["A", "B"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


class ConfidenceThresholdSimpleQAEval(Eval):
    """
    Enhanced SimpleQA evaluation with confidence threshold framework
    Implements penalty-aware scoring and behavioral calibration
    """

    def __init__(
        self,
        grader_model: SamplerBase = None,
        num_examples: int | None = None,
        n_repeats: int = 1,
        audit_logger=None,
        confidence_thresholds: List[ConfidenceThreshold] = None
    ):
        # Use GPT-5-mini as default grader
        if grader_model is None:
            grader_model = ChatCompletionSampler(
                model="gpt-5-mini",
                temperature=0.0
            )

        with requests.get("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv") as response:
            blob_file = io.BytesIO(response.content)
            df = pandas.read_csv(blob_file)

        examples = [row.to_dict() for _, row in df.iterrows()]

        # Use single randomization seed for all operations to ensure reproducibility
        rng = random.Random(42)  # Fixed seed for all random operations

        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            examples = rng.sample(examples, num_examples)

        # Randomize question order for experimental control (using same RNG)
        rng.shuffle(examples)

        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.audit_logger = audit_logger
        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS

        # Add thread safety for audit logging
        self._audit_lock = threading.Lock()

        # Add question IDs for tracking
        for i, example in enumerate(self.examples):
            example['question_id'] = f"simpleqa_{i:04d}"

    def _create_clean_prompt(self, question: str) -> str:
        """Create a clean question prompt without threshold instructions"""
        return question

    def grade_sample_with_explanation(
        self,
        question_id: str,
        question: str,
        target: str,
        predicted_answer: str,
        provider_name: str,
        threshold: ConfidenceThreshold
    ) -> Dict[str, any]:
        """
        Grade a sample with confidence threshold context and full explanation
        """
        start_time = time.time()

        confidence_instructions = (
            f"Answer only if >{threshold.threshold*100:.0f}% confident "
            f"(penalty ratio: {threshold.penalty_ratio})"
        )

        grader_prompt = CONFIDENCE_THRESHOLD_GRADER_TEMPLATE.format(
            threshold_name=threshold.name,
            threshold_value=threshold.threshold,
            penalty_ratio=threshold.penalty_ratio,
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
                fallback_match = re.search(r"(A|B)", grading_response)
                if fallback_match:
                    grade_letter = fallback_match.group(0)

            # Calculate threshold-aware scores (judge can only give A or B)
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            # Provider abstention is handled separately, not from judge evaluation
            is_not_attempted = False  # Judge never says "not attempted"

            # Volume strategy score (traditional binary)
            volume_score = 1.0 if is_correct else 0.0

            # Quality strategy score (penalty-aware)
            if is_correct:
                quality_score = 1.0
            elif is_incorrect:
                quality_score = -threshold.penalty_ratio
            else:  # not_attempted
                quality_score = 0.0

            # Log the judge evaluation if audit logger is available (thread-safe)
            if self.audit_logger:
                with self._audit_lock:
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
                "latency_ms": latency_ms,
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
                "is_not_attempted": is_not_attempted,
                "volume_score": volume_score,
                "quality_score": quality_score
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            if self.audit_logger:
                with self._audit_lock:
                    self.audit_logger.log_error(
                        component="confidence_threshold_grader",
                        error=str(e),
                        context={
                            "question_id": question_id,
                            "question": question,
                            "provider": provider_name,
                            "threshold": threshold.name
                        }
                    )

            # Return default grade with error info
            return {
                "grade_letter": "C",
                "grade_string": "NOT_ATTEMPTED",
                "reasoning": f"Grading failed due to error: {str(e)}",
                "full_response": "",
                "latency_ms": latency_ms,
                "error": str(e),
                "is_correct": False,
                "is_incorrect": False,
                "is_not_attempted": True,
                "volume_score": 0.0,
                "quality_score": 0.0
            }

    def _call_provider_single(self, sampler: SamplerBase, row: Dict, provider_name: str) -> Dict[str, any]:
        """Call provider for a single question"""
        question_id = row.get("question_id", f"q_{uuid.uuid4().hex[:8]}")
        question = row.get("problem", "")
        target = row.get("answer", "")

        # Create clean prompt without threshold instructions
        clean_prompt = self._create_clean_prompt(question)

        prompt_messages = [
            sampler._pack_message(content=clean_prompt, role="user")
        ]

        # Call the sampler once
        if hasattr(sampler, '__call__') and 'question_id' in sampler.__call__.__code__.co_varnames:
            response_text = sampler(prompt_messages, question_id=question_id)
        else:
            response_text = sampler(prompt_messages)

        return {
            "question_id": question_id,
            "question": question,
            "target": target,
            "response": response_text,
            "prompt_messages": prompt_messages,
            "provider_name": provider_name
        }

    def evaluate_provider_responses(self, sampler: SamplerBase, provider_name: str = None) -> List[Dict[str, any]]:
        """Get natural responses from provider without threshold contamination"""
        if provider_name is None:
            provider_name = sampler.__class__.__name__.replace("Sampler", "").replace("Audited", "")

        # Determine max workers based on provider type
        # CustomGPT rate limit: 5 concurrent calls
        # OpenAI providers: Can handle more concurrent calls
        if "CustomGPT" in provider_name:
            max_workers = 5
        else:
            max_workers = 10

        print(f"   Using {max_workers} parallel workers for {provider_name}")

        provider_responses = []
        failed_responses = []
        completed_count = 0
        total_count = len(self.examples)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self._call_provider_single, sampler, row, provider_name): row
                for row in self.examples
            }

            # Collect results as they complete with progress tracking
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                completed_count += 1

                try:
                    result = future.result()
                    provider_responses.append(result)
                except Exception as e:
                    failed_responses.append({
                        "question_id": row.get("question_id", f"q_{uuid.uuid4().hex[:8]}"),
                        "error": str(e),
                        "row": row
                    })
                    print(f"   ❌ Failed to get response for question {row.get('question_id', 'unknown')}: {str(e)}")

                # Progress reporting every 10% or every 10 items (whichever is smaller)
                progress_interval = max(1, min(10, total_count // 10))
                if completed_count % progress_interval == 0 or completed_count == total_count:
                    progress_pct = (completed_count / total_count) * 100
                    print(f"   📊 Provider progress: {completed_count}/{total_count} ({progress_pct:.1f}%)")

        # Convert failed responses to abstention entries
        for failed_response in failed_responses:
            row = failed_response["row"]
            abstention_response = {
                "question_id": failed_response["question_id"],
                "question": row.get("problem", ""),
                "target": row.get("answer", ""),
                "response": "",  # Empty response indicates abstention
                "prompt_messages": [],
                "provider_name": provider_name,
                "error": failed_response["error"]
            }
            provider_responses.append(abstention_response)

        if failed_responses:
            print(f"   ⚠️  {len(failed_responses)} questions failed and will be marked as abstentions")

        print(f"   ✅ Collected {len(provider_responses)} total responses ({len(provider_responses) - len(failed_responses)} successful, {len(failed_responses)} abstentions)")
        return provider_responses

    def _evaluate_single_response(self, response_data: Dict[str, any], threshold: ConfidenceThreshold) -> SingleEvalResult:
        """Evaluate a single response against a threshold"""

        # Check if provider abstained (empty/invalid response)
        provider_response = response_data["response"]
        if not provider_response or provider_response.strip() == "":
            # Provider abstained - don't send to judge
            grading_result = {
                "grade_letter": "C",
                "grade_string": "NOT_ATTEMPTED",
                "reasoning": "Provider did not provide a response",
                "full_response": "",
                "latency_ms": 0,
                "is_correct": False,
                "is_incorrect": False,
                "is_not_attempted": True,
                "volume_score": 0.0,
                "quality_score": 0.0
            }
        else:
            # Grade the response with confidence threshold context
            grading_result = self.grade_sample_with_explanation(
                question_id=response_data["question_id"],
                question=response_data["question"],
                target=response_data["target"],
                predicted_answer=provider_response,
                provider_name=response_data["provider_name"],
                threshold=threshold
            )

        # Create HTML for each sample result
        html = common.jinja_env.from_string(common.HTML_JINJA).render(
            prompt_messages=response_data["prompt_messages"],
            next_message=dict(content=response_data["response"], role="assistant"),
            score=grading_result["volume_score"],
            correct_answer=response_data["target"],
            extracted_answer=response_data["response"],
            grading_explanation=grading_result["reasoning"],
            grade=grading_result["grade_string"]
        )

        convo = response_data["prompt_messages"] + [dict(content=response_data["response"], role="assistant")]

        return SingleEvalResult(
            html=html,
            score=grading_result["volume_score"],  # Keep traditional score for compatibility
            convo=convo,
            metrics={
                "is_correct": grading_result["is_correct"],
                "is_incorrect": grading_result["is_incorrect"],
                "is_not_attempted": grading_result["is_not_attempted"],
                "volume_score": grading_result["volume_score"],
                "quality_score": grading_result["quality_score"],
                "judge_latency_ms": grading_result["latency_ms"]
                # Note: threshold_value, penalty_ratio, and threshold_name stored in HTML/convo to avoid NumPy aggregation issues
            }
        )

    def evaluate_single_threshold(self, provider_responses: List[Dict[str, any]], threshold: ConfidenceThreshold) -> List[SingleEvalResult]:
        """Evaluate provider responses against a specific confidence threshold"""
        # Use 10 workers for judge evaluations - GPT-4.1/5-mini can handle more concurrent requests
        max_workers = 10

        print(f"   Using {max_workers} parallel workers for judge evaluations")

        results = []
        failed_evaluations = []
        completed_count = 0
        total_count = len(provider_responses)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_response = {
                executor.submit(self._evaluate_single_response, response_data, threshold): response_data
                for response_data in provider_responses
            }

            # Collect results as they complete with progress tracking
            for future in as_completed(future_to_response):
                response_data = future_to_response[future]
                completed_count += 1

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    failed_evaluations.append({
                        "question_id": response_data.get("question_id", "unknown"),
                        "error": str(e),
                        "response_data": response_data
                    })
                    print(f"   ❌ Failed to evaluate response for question {response_data.get('question_id', 'unknown')}: {str(e)}")

                # Progress reporting every 10% or every 10 items (whichever is smaller)
                progress_interval = max(1, min(10, total_count // 10))
                if completed_count % progress_interval == 0 or completed_count == total_count:
                    progress_pct = (completed_count / total_count) * 100
                    print(f"   📊 Judge progress: {completed_count}/{total_count} ({progress_pct:.1f}%)")

        if failed_evaluations:
            print(f"   ⚠️  {len(failed_evaluations)} evaluations failed out of {len(provider_responses)}")

        print(f"   ✅ Completed {len(results)} judge evaluations")
        return results

    def __call__(self, sampler: SamplerBase, provider_name: str = None) -> Dict[str, EvalResult]:
        """
        Run multi-threshold evaluation and return results for each threshold
        """
        if provider_name is None:
            provider_name = sampler.__class__.__name__.replace("Sampler", "").replace("Audited", "")

        print(f"\n🎯 Running Confidence Threshold Evaluation for {provider_name}")
        print("=" * 60)

        # Update audit logger with total questions
        if self.audit_logger:
            self.audit_logger.update_progress(0, len(self.examples))

        # First, get natural responses from provider (called once per question)
        print(f"\n📞 Getting natural responses from {provider_name} ({len(self.examples)} questions)")
        provider_responses = self.evaluate_provider_responses(sampler, provider_name)

        # Update progress after getting responses
        if self.audit_logger:
            self.audit_logger.update_progress(len(provider_responses))

        threshold_results = {}

        # Then evaluate these responses against each threshold
        for threshold in self.confidence_thresholds:
            print(f"\n📊 Evaluating against {threshold.name} threshold (t={threshold.threshold}, penalty={threshold.penalty_ratio})")
            print(f"   Evaluation criteria: {threshold.description}")

            # Evaluate the natural responses against this threshold
            results = self.evaluate_single_threshold(provider_responses, threshold)

            # Calculate threshold-specific metrics
            aggregate_metrics = self._calculate_threshold_metrics(results, threshold)

            # Create EvalResult for this threshold
            base_result = common.aggregate_results(results)
            base_result.metrics.update(aggregate_metrics)

            threshold_results[threshold.name] = base_result

            # Print summary for this threshold
            print(f"   📈 Results:")
            print(f"      Volume Score (traditional): {aggregate_metrics['volume_score_mean']:.3f}")
            print(f"      Quality Score (penalty-aware): {aggregate_metrics['quality_score_mean']:.3f}")
            print(f"      Attempted Rate: {aggregate_metrics['attempted_rate']:.3f}")
            print(f"      Success on Attempted: {aggregate_metrics['accuracy_given_attempted']:.3f}")

        return threshold_results

    def analyze_statistical_significance(self, provider_results: Dict[str, Dict[str, EvalResult]]) -> Dict[str, any]:
        """
        Perform cross-provider statistical analysis
        Returns statistical comparisons between providers across thresholds
        """
        statistical_analysis = {
            "pairwise_comparisons": {},
            "effect_sizes": {},
            "grade_distribution_tests": {},
            "summary": {}
        }

        providers = list(provider_results.keys())
        thresholds = list(next(iter(provider_results.values())).keys())

        # Pairwise comparisons for each threshold
        for threshold in thresholds:
            statistical_analysis["pairwise_comparisons"][threshold] = {}
            statistical_analysis["effect_sizes"][threshold] = {}
            statistical_analysis["grade_distribution_tests"][threshold] = {}

            # Get data for this threshold
            threshold_data = {}
            for provider in providers:
                result = provider_results[provider][threshold]
                threshold_data[provider] = {
                    "volume_scores": result.metrics.get("volume_scores_raw", []),
                    "quality_scores": result.metrics.get("quality_scores_raw", []),
                    "n_correct": result.metrics.get("n_correct", 0),
                    "n_incorrect": result.metrics.get("n_incorrect", 0),
                    "n_not_attempted": result.metrics.get("n_not_attempted", 0)
                }

            # Pairwise comparisons between providers
            for i, provider1 in enumerate(providers):
                for provider2 in providers[i+1:]:
                    comparison_key = f"{provider1}_vs_{provider2}"

                    # t-tests for continuous metrics
                    volume_comparison = self._compare_continuous_metrics(
                        threshold_data[provider1]["volume_scores"],
                        threshold_data[provider2]["volume_scores"],
                        "Volume Score"
                    )

                    quality_comparison = self._compare_continuous_metrics(
                        threshold_data[provider1]["quality_scores"],
                        threshold_data[provider2]["quality_scores"],
                        "Quality Score"
                    )

                    statistical_analysis["pairwise_comparisons"][threshold][comparison_key] = {
                        "volume_score": volume_comparison,
                        "quality_score": quality_comparison
                    }

                    # Effect sizes
                    volume_effect = cohens_d(
                        threshold_data[provider1]["volume_scores"],
                        threshold_data[provider2]["volume_scores"]
                    )
                    quality_effect = cohens_d(
                        threshold_data[provider1]["quality_scores"],
                        threshold_data[provider2]["quality_scores"]
                    )

                    statistical_analysis["effect_sizes"][threshold][comparison_key] = {
                        "volume_score_cohens_d": volume_effect,
                        "volume_score_interpretation": interpret_effect_size(volume_effect, "cohens_d"),
                        "quality_score_cohens_d": quality_effect,
                        "quality_score_interpretation": interpret_effect_size(quality_effect, "cohens_d")
                    }

                    # Chi-square test for grade distributions
                    chi2_result = self._compare_grade_distributions(
                        threshold_data[provider1],
                        threshold_data[provider2],
                        provider1,
                        provider2
                    )

                    statistical_analysis["grade_distribution_tests"][threshold][comparison_key] = chi2_result

        # Summary statistics across all comparisons
        statistical_analysis["summary"] = self._summarize_statistical_results(statistical_analysis)

        return statistical_analysis

    def _compare_continuous_metrics(self, group1: List[float], group2: List[float], metric_name: str) -> Dict[str, any]:
        """Compare continuous metrics between two groups using t-test"""
        if len(group1) < 2 or len(group2) < 2:
            return {
                "metric": metric_name,
                "n1": len(group1),
                "n2": len(group2),
                "mean1": np.mean(group1) if group1 else 0,
                "mean2": np.mean(group2) if group2 else 0,
                "t_statistic": None,
                "p_value": None,
                "significant": False,
                "note": "Insufficient data for statistical test"
            }

        # Perform t-test
        try:
            t_stat, p_value = ttest_ind(group1, group2)

            # Note: Multiple comparison correction will be applied at summary level
            # Individual p-values are raw, correction applied to final significance
            significant_raw = p_value < 0.05

            return {
                "metric": metric_name,
                "n1": len(group1),
                "n2": len(group2),
                "mean1": np.mean(group1),
                "mean2": np.mean(group2),
                "std1": np.std(group1, ddof=1),
                "std2": np.std(group2, ddof=1),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_raw": significant_raw,
                "note": "Welch's t-test (unequal variances assumed)"
            }
        except Exception as e:
            return {
                "metric": metric_name,
                "error": str(e),
                "significant": False,
                "note": "Statistical test failed"
            }

    def _compare_grade_distributions(self, data1: Dict, data2: Dict, provider1: str, provider2: str) -> Dict[str, any]:
        """Compare grade distributions using chi-square test"""
        # Create contingency table
        contingency_table = np.array([
            [data1["n_correct"], data1["n_incorrect"], data1["n_not_attempted"]],
            [data2["n_correct"], data2["n_incorrect"], data2["n_not_attempted"]]
        ])

        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            cramers_v_value = cramers_v(contingency_table)

            # Note: Multiple comparison correction will be applied at summary level
            significant_raw = p_value < 0.05

            return {
                "contingency_table": contingency_table.tolist(),
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": float(cramers_v_value),
                "cramers_v_interpretation": interpret_effect_size(cramers_v_value, "cramers_v"),
                "significant_raw": significant_raw,
                "providers": [provider1, provider2],
                "grade_categories": ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
            }
        except Exception as e:
            return {
                "error": str(e),
                "significant": False,
                "note": "Chi-square test failed"
            }

    def _summarize_statistical_results(self, analysis: Dict) -> Dict[str, any]:
        """Create summary of statistical analysis with Bonferroni correction"""

        # Collect all p-values for Bonferroni correction
        all_p_values = []
        comparison_types = []  # Track what each p-value represents

        for threshold in analysis["pairwise_comparisons"]:
            for comparison in analysis["pairwise_comparisons"][threshold]:
                # Volume score p-values
                vol_p = analysis["pairwise_comparisons"][threshold][comparison]["volume_score"].get("p_value")
                if vol_p is not None:
                    all_p_values.append(vol_p)
                    comparison_types.append(f"{threshold}_{comparison}_volume")

                # Quality score p-values
                qual_p = analysis["pairwise_comparisons"][threshold][comparison]["quality_score"].get("p_value")
                if qual_p is not None:
                    all_p_values.append(qual_p)
                    comparison_types.append(f"{threshold}_{comparison}_quality")

                # Distribution test p-values
                dist_p = analysis["grade_distribution_tests"][threshold][comparison].get("p_value")
                if dist_p is not None:
                    all_p_values.append(dist_p)
                    comparison_types.append(f"{threshold}_{comparison}_distribution")

        # Apply Bonferroni correction
        if all_p_values:
            corrected_significant, corrected_alpha = apply_bonferroni_correction(all_p_values)
        else:
            corrected_significant = []
            corrected_alpha = 0.05

        summary = {
            "total_comparisons": len(analysis["pairwise_comparisons"].get("Balanced", {})),
            "total_statistical_tests": len(all_p_values),
            "bonferroni_corrected_alpha": corrected_alpha,
            "significant_volume_comparisons_raw": 0,
            "significant_quality_comparisons_raw": 0,
            "significant_distribution_comparisons_raw": 0,
            "significant_volume_comparisons_corrected": 0,
            "significant_quality_comparisons_corrected": 0,
            "significant_distribution_comparisons_corrected": 0,
            "large_effect_sizes": 0,
            "medium_effect_sizes": 0,
            "small_effect_sizes": 0
        }

        # Count raw and corrected significance
        p_idx = 0
        for threshold in analysis["pairwise_comparisons"]:
            for comparison in analysis["pairwise_comparisons"][threshold]:

                # Volume score results
                vol_data = analysis["pairwise_comparisons"][threshold][comparison]["volume_score"]
                if vol_data.get("p_value") is not None:
                    if vol_data.get("significant_raw", False):
                        summary["significant_volume_comparisons_raw"] += 1
                    if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                        summary["significant_volume_comparisons_corrected"] += 1
                    p_idx += 1

                # Quality score results
                qual_data = analysis["pairwise_comparisons"][threshold][comparison]["quality_score"]
                if qual_data.get("p_value") is not None:
                    if qual_data.get("significant_raw", False):
                        summary["significant_quality_comparisons_raw"] += 1
                    if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                        summary["significant_quality_comparisons_corrected"] += 1
                    p_idx += 1

                # Distribution test results
                dist_data = analysis["grade_distribution_tests"][threshold][comparison]
                if dist_data.get("p_value") is not None:
                    if dist_data.get("significant_raw", False):
                        summary["significant_distribution_comparisons_raw"] += 1
                    if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                        summary["significant_distribution_comparisons_corrected"] += 1
                    p_idx += 1

                # Count effect sizes
                effect_data = analysis["effect_sizes"][threshold][comparison]
                for interpretation in [effect_data["volume_score_interpretation"], effect_data["quality_score_interpretation"]]:
                    if interpretation == "large":
                        summary["large_effect_sizes"] += 1
                    elif interpretation == "medium":
                        summary["medium_effect_sizes"] += 1
                    elif interpretation == "small":
                        summary["small_effect_sizes"] += 1

        return summary

    def _calculate_threshold_metrics(self, results: List[SingleEvalResult], threshold: ConfidenceThreshold) -> Dict[str, float]:
        """Calculate comprehensive metrics for a threshold"""
        n_total = len(results)

        # Basic counts
        n_correct = sum(r.metrics["is_correct"] for r in results)
        n_incorrect = sum(r.metrics["is_incorrect"] for r in results)
        n_not_attempted = sum(r.metrics["is_not_attempted"] for r in results)
        n_attempted = n_total - n_not_attempted

        # Volume strategy metrics (traditional)
        volume_score_mean = sum(r.metrics["volume_score"] for r in results) / n_total

        # Quality strategy metrics (penalty-aware)
        quality_scores = [r.metrics["quality_score"] for r in results]
        quality_score_mean = sum(quality_scores) / n_total

        # Behavioral metrics
        attempted_rate = n_attempted / n_total if n_total > 0 else 0
        abstention_rate = n_not_attempted / n_total if n_total > 0 else 0
        accuracy_given_attempted = n_correct / n_attempted if n_attempted > 0 else 0
        error_rate_given_attempted = n_incorrect / n_attempted if n_attempted > 0 else 0

        # Conservative strategy analysis (from the paper)
        conservative_penalty = n_incorrect * threshold.penalty_ratio
        conservative_benefit = n_not_attempted * 0  # Abstention = 0 points

        # Overconfidence penalty (questions answered incorrectly when should have abstained)
        overconfidence_penalty = n_incorrect

        # Judge performance metrics
        judge_latencies = [r.metrics["judge_latency_ms"] for r in results if "judge_latency_ms" in r.metrics]
        avg_judge_latency_ms = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0

        # Statistical analysis - confidence intervals for key metrics
        volume_scores = [r.metrics["volume_score"] for r in results]
        quality_scores = [r.metrics["quality_score"] for r in results]

        # Calculate confidence intervals
        volume_ci = calculate_confidence_interval(volume_scores)
        quality_ci = calculate_confidence_interval(quality_scores)

        # Binary outcome confidence intervals (using binomial distribution)
        attempted_ci = calculate_confidence_interval([1 if r.metrics["is_correct"] or r.metrics["is_incorrect"] else 0 for r in results])
        accuracy_attempted_scores = [r.metrics["is_correct"] for r in results if r.metrics["is_correct"] or r.metrics["is_incorrect"]]
        accuracy_attempted_ci = calculate_confidence_interval(accuracy_attempted_scores) if accuracy_attempted_scores else (0, 0, 0)

        return {
            # Core metrics
            "volume_score_mean": volume_score_mean,
            "quality_score_mean": quality_score_mean,

            # Behavioral metrics
            "attempted_rate": attempted_rate,
            "abstention_rate": abstention_rate,
            "accuracy_given_attempted": accuracy_given_attempted,
            "error_rate_given_attempted": error_rate_given_attempted,

            # Raw counts
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_not_attempted": n_not_attempted,
            "n_attempted": n_attempted,
            "n_total": n_total,

            # Conservative strategy analysis
            "conservative_penalty": conservative_penalty,
            "overconfidence_penalty": overconfidence_penalty,

            # Threshold configuration (numeric only - strings excluded to avoid NumPy aggregation issues)
            "threshold_value": threshold.threshold,
            "penalty_ratio": threshold.penalty_ratio,

            # Performance
            "avg_judge_latency_ms": avg_judge_latency_ms,

            # Statistical analysis - confidence intervals (95%)
            "volume_score_ci_lower": volume_ci[1],
            "volume_score_ci_upper": volume_ci[2],
            "quality_score_ci_lower": quality_ci[1],
            "quality_score_ci_upper": quality_ci[2],
            "attempted_rate_ci_lower": attempted_ci[1],
            "attempted_rate_ci_upper": attempted_ci[2],
            "accuracy_attempted_ci_lower": accuracy_attempted_ci[1],
            "accuracy_attempted_ci_upper": accuracy_attempted_ci[2],

            # Raw data for cross-provider analysis (stored separately to avoid aggregation issues)
            # Note: volume_scores_raw and quality_scores_raw excluded from metrics to prevent NumPy issues

            # Legacy compatibility
            "is_correct": n_correct / n_total,
            "is_incorrect": n_incorrect / n_total,
            "is_not_attempted": n_not_attempted / n_total,
        }