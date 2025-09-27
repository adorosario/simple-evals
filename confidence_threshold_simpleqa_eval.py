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
from datetime import datetime

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


# Single confidence threshold configuration based on OpenAI paper recommendations
# Using 80% confidence threshold as recommended in arXiv:2509.04664v1
CONFIDENCE_THRESHOLDS = [
    ConfidenceThreshold(0.8, 4.0, "Conservative"),  # t=0.8, penalty=4 (balanced penalty for 80% threshold)
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
Grade this answer as CORRECT (A) or INCORRECT (B).

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

GRADING RULES:
- CORRECT: Answer contains key information from gold target without contradictions
- INCORRECT: Answer contradicts gold target or provides wrong information
- Semantic meaning matters, not formatting/capitalization
- Allow hedging if core facts are correct

Your JSON response must include:
- reasoning: Brief analysis
- grade: "A" (CORRECT) or "B" (INCORRECT)
- confidence: 0.0-1.0 rating of your certainty
- consistency_check: Confirm reasoning matches grade
""".strip()


CHOICE_LETTERS = ["A", "B"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


def classify_response_intent(response: str, question: str) -> Dict[str, any]:
    """
    Use GPT-5-nano to classify provider response intent with few-shot examples.
    Returns: {"type": "abstention"|"attempt", "confidence": float, "reasoning": str}
    """
    if not response or response.strip() == "":
        return {
            "type": "abstention",
            "confidence": 1.0,
            "reasoning": "Empty response indicates abstention"
        }

    # Create classifier using GPT-5-nano (lightweight, fast)
    from sampler.chat_completion_sampler import ChatCompletionSampler
    classifier = ChatCompletionSampler(
        model="gpt-5-nano",
        seed=42,  # Fixed seed for deterministic classification
        reasoning_effort="minimal"  # Fast, consistent classification
    )

    classification_prompt = f"""Your task is to classify whether a provider's response represents an ABSTENTION (intentionally declining to answer) or an ATTEMPT (trying to provide an answer, even if uncertain or wrong).

CLASSIFICATION RULES:
- ABSTENTION: Provider explicitly declines to answer, states they lack information, or says they "could not find" the requested information
- ATTEMPT: Provider gives an answer, even if hedged, uncertain, or potentially incorrect

KEY ABSTENTION INDICATORS:
- "I don't know" / "I do not know"
- "I cannot" / "I can't" / "unable to"
- "I could not find" / "could not locate"
- "I don't have access to" / "no access"
- "My knowledge base does not" / "not in my knowledge base"
- "I'm sorry, but I could not" / "sorry, I cannot"
- "Sorry" followed by inability expressions (very strong abstention signal)

IMPORTANT: "Sorry" is typically used when providers CANNOT provide information, not when they're uncertain about correct answers.

ABSTENTION EXAMPLES:
```
Question: What are the names of Barack Obama's children?
Response: I don't know the names of Barack Obama's children.
→ ABSTENTION (explicit "I don't know")

Question: What is the capital of Bhutan?
Response: I don't have enough information in my knowledge base to answer this question.
→ ABSTENTION (knowledge limitation)

Question: When was the first iPhone released?
Response: I cannot answer this question without access to current information.
→ ABSTENTION (cannot answer)

Question: What is the population of Tokyo?
Response: My training data doesn't include current population figures for Tokyo.
→ ABSTENTION (training data limitation)

Question: What is the DOI for paper X?
Response: I'm sorry, but I could not find a DOI for the paper titled "X".
→ ABSTENTION (could not find requested information)

Question: What happened in the meeting?
Response: I don't have access to that specific meeting information.
→ ABSTENTION (no access to information)

Question: What's the latest stock price?
Response: I cannot provide current stock prices as I don't have real-time data.
→ ABSTENTION (cannot provide due to data limitations)

Question: What is the DOI for paper X?
Response: Sorry, I don't have access to DOI information for that paper.
→ ABSTENTION (sorry + no access)

Question: What happened at the meeting?
Response: I'm sorry, but I don't have information about that specific meeting.
→ ABSTENTION (sorry + don't have information)

Question: What's the password for the account?
Response: Sorry, I cannot provide password information for security reasons.
→ ABSTENTION (sorry + cannot provide)
```

ATTEMPT EXAMPLES:
```
Question: What are the names of Barack Obama's children?
Response: I believe they are Malia and Sasha, though I'm not completely certain.
→ ATTEMPT (provides answer with uncertainty hedge)

Question: What is the capital of Bhutan?
Response: I think it might be Thimphu, but I should double-check that.
→ ATTEMPT (provides answer despite uncertainty)

Question: When was the first iPhone released?
Response: It was released sometime in 2007, if I remember correctly.
→ ATTEMPT (provides answer with hedge)

Question: What is the population of Tokyo?
Response: Tokyo has approximately 14 million people, though this number changes frequently.
→ ATTEMPT (provides answer with caveat)

Question: Who won the 2020 World Series?
Response: I'm not sure, but I think it might have been the Dodgers or possibly the Rays.
→ ATTEMPT (uncertain but provides candidates)
```

Now classify this response:

Question: {question}
Response: {response}

Provide your classification as:
CLASSIFICATION: ABSTENTION or ATTEMPT
CONFIDENCE: 0.0 to 1.0
REASONING: Brief explanation of your decision

Focus on whether the provider is TRYING to answer (even if uncertain) vs DECLINING to answer."""

    try:
        classification_messages = [
            classifier._pack_message(content=classification_prompt, role="user")
        ]

        classification_response = classifier(classification_messages)

        # Parse the response
        classification_type = "attempt"  # default
        confidence = 0.8  # default
        reasoning = "Default classification due to parsing error"

        # Extract classification
        if "CLASSIFICATION:" in classification_response:
            class_line = classification_response.split("CLASSIFICATION:")[1].split("\n")[0].strip().lower()
            if "abstention" in class_line:
                classification_type = "abstention"
            elif "attempt" in class_line:
                classification_type = "attempt"

        # Extract confidence
        if "CONFIDENCE:" in classification_response:
            conf_line = classification_response.split("CONFIDENCE:")[1].split("\n")[0].strip()
            try:
                confidence = float(conf_line)
            except:
                confidence = 0.8

        # Extract reasoning
        if "REASONING:" in classification_response:
            reasoning = classification_response.split("REASONING:")[1].strip()

        return {
            "type": classification_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "full_response": classification_response
        }

    except Exception as e:
        # Fallback to basic heuristics if GPT-5-nano fails
        response_lower = response.lower().strip()

        # Strong abstention indicators
        strong_abstention_phrases = [
            "i don't know", "i do not know", "cannot answer", "can't answer",
            "unable to answer", "my knowledge base does not", "not in my knowledge base",
            "don't have enough information", "insufficient information",
            "could not find", "could not locate", "i could not find",
            "sorry, but i could not", "sorry, i cannot", "i'm sorry, but i could not",
            "don't have access to", "do not have access to",
            # Sorry patterns (strong abstention signals)
            "sorry, i don't", "sorry, i do not", "sorry, i can't", "sorry, i cannot",
            "i'm sorry, but i don't", "i'm sorry, but i can't", "i'm sorry, but i cannot",
            "sorry, but i don't", "sorry, but i can't"
        ]

        for phrase in strong_abstention_phrases:
            if phrase in response_lower:
                return {
                    "type": "abstention",
                    "confidence": 0.9,
                    "reasoning": f"Fallback classification: Contains strong abstention phrase '{phrase}'"
                }

        # Special check: "sorry" at the start is often an abstention signal
        if response_lower.startswith("sorry") or response_lower.startswith("i'm sorry"):
            # Check if it's followed by inability/limitation words
            inability_words = ["i don't", "i do not", "i can't", "i cannot", "but i", "i could not", "no", "not"]
            if any(word in response_lower for word in inability_words):
                return {
                    "type": "abstention",
                    "confidence": 0.85,
                    "reasoning": "Fallback classification: Starts with 'sorry' and contains inability expressions"
                }

        return {
            "type": "attempt",
            "confidence": 0.7,
            "reasoning": f"Fallback classification: No clear abstention signals detected. Classifier error: {str(e)}"
        }


def detect_abstention(response: str) -> bool:
    """
    Legacy wrapper for backwards compatibility.
    Uses intelligent classification instead of string matching.
    """
    classification = classify_response_intent(response, "")
    return classification["type"] == "abstention"


def validate_judge_reasoning_structured(reasoning: str, grade: str, question: str, target: str, predicted: str) -> dict:
    """
    Use GPT-5-nano to validate judge reasoning consistency with structured output.
    Replaces brittle string pattern matching with reliable LLM-based validation.
    """

    grade_meaning = "CORRECT" if grade == "A" else "INCORRECT" if grade == "B" else "NOT_ATTEMPTED"

    validation_prompt = f"""You are a validation expert analyzing judge consistency. Determine if the judge's reasoning logically supports their assigned grade.

QUESTION: {question}
TARGET ANSWER: {target}
PREDICTED ANSWER: {predicted}
JUDGE GRADE: {grade} ({grade_meaning})
JUDGE REASONING: {reasoning}

Your task: Analyze if the reasoning logically supports the grade assigned.

Key validation criteria:
- If reasoning says the answer is correct/accurate/matches, grade should be A
- If reasoning says the answer is incorrect/wrong/contradicts, grade should be B
- If reasoning is unclear or contradictory, note the inconsistency

Respond in this exact format:
VALIDATION: CONSISTENT or INCONSISTENT
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of your assessment]
SUGGESTED_GRADE: [A, B, or C - only if INCONSISTENT, otherwise same as original]"""

    try:
        # Use GPT-5-nano for structured validation (same as abstention classifier)
        classifier = ChatCompletionSampler(model="gpt-5-nano", temperature=0.0)
        validation_messages = [
            classifier._pack_message(content=validation_prompt, role="user")
        ]

        validation_response = classifier(validation_messages)

        # Parse the structured response
        validation = "CONSISTENT"  # default
        confidence = 0.8  # default
        validator_reasoning = "Default due to parsing error"
        suggested_grade = grade  # default to original

        # Extract validation decision
        if "VALIDATION:" in validation_response:
            val_line = validation_response.split("VALIDATION:")[1].split("\n")[0].strip().upper()
            if "INCONSISTENT" in val_line:
                validation = "INCONSISTENT"
            elif "CONSISTENT" in val_line:
                validation = "CONSISTENT"

        # Extract confidence
        if "CONFIDENCE:" in validation_response:
            conf_line = validation_response.split("CONFIDENCE:")[1].split("\n")[0].strip()
            try:
                confidence = float(conf_line)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            except:
                confidence = 0.8

        # Extract reasoning
        if "REASONING:" in validation_response:
            reasoning_section = validation_response.split("REASONING:")[1].split("SUGGESTED_GRADE:")[0].strip()
            validator_reasoning = reasoning_section

        # Extract suggested grade (only if inconsistent)
        if validation == "INCONSISTENT" and "SUGGESTED_GRADE:" in validation_response:
            grade_line = validation_response.split("SUGGESTED_GRADE:")[1].split("\n")[0].strip().upper()
            if grade_line in ["A", "B", "C"]:
                suggested_grade = grade_line

        return {
            "validation": validation,
            "validation_passed": validation == "CONSISTENT",
            "confidence": confidence,
            "reasoning": validator_reasoning,
            "suggested_grade": suggested_grade,
            "original_grade": grade,
            "full_response": validation_response,
            "validation_method": "structured_gpt5_nano"
        }

    except Exception as e:
        # Fallback: accept original grade if validation fails
        return {
            "validation": "CONSISTENT",
            "validation_passed": True,
            "confidence": 0.5,
            "reasoning": f"Validation failed, accepting original grade: {str(e)}",
            "suggested_grade": grade,
            "original_grade": grade,
            "full_response": "",
            "validation_method": "fallback_due_to_error",
            "validation_error": str(e)
        }


def validate_judge_consistency(reasoning: str, final_grade: str, full_response: str) -> dict:
    """
    Validate consistency between judge reasoning and final grade.
    Returns validation metadata and suggested corrections.
    """
    validation_result = {
        "reasoning_grade_consistent": True,
        "confidence_signals_detected": [],
        "validation_passed": True,
        "inconsistency_type": None,
        "suggested_grade": final_grade,
        "reasoning_indicators": []
    }

    # Convert grade to reasoning-friendly format
    grade_meaning = "CORRECT" if final_grade == "A" else "INCORRECT"

    # Extract confidence signals from reasoning
    reasoning_lower = reasoning.lower()

    # Detect explicit abstention signals
    abstention_signals = [
        "does not specify", "don't know", "cannot answer", "not sure",
        "abstain", "decline", "not attempted", "knowledge base does not",
        "would have to", "need more", "unclear"
    ]

    for signal in abstention_signals:
        if signal in reasoning_lower:
            validation_result["confidence_signals_detected"].append(f"abstention: {signal}")

    # Detect certainty signals
    certainty_signals = [
        "exactly", "explicitly states", "clearly", "directly", "factually correct",
        "matches exactly", "definite", "precise", "accurate"
    ]

    for signal in certainty_signals:
        if signal in reasoning_lower:
            validation_result["confidence_signals_detected"].append(f"certainty: {signal}")

    # Check for reasoning-grade inconsistencies
    reasoning_suggests_correct = any([
        "should be graded correct" in reasoning_lower,
        "therefore it should be graded as correct" in reasoning_lower,
        "final grade: a" in reasoning_lower,
        "grade this predicted answer as: a" in reasoning_lower,
        "the predicted answer is correct" in reasoning_lower
    ])

    reasoning_suggests_incorrect = any([
        "should be graded incorrect" in reasoning_lower,
        "therefore it should be graded as incorrect" in reasoning_lower,
        "final grade: b" in reasoning_lower,
        "grade this predicted answer as: b" in reasoning_lower,
        "the predicted answer is incorrect" in reasoning_lower,
        "contradicts the gold target" in reasoning_lower
    ])

    # Track reasoning indicators
    if reasoning_suggests_correct:
        validation_result["reasoning_indicators"].append("suggests_correct")
    if reasoning_suggests_incorrect:
        validation_result["reasoning_indicators"].append("suggests_incorrect")

    # Detect inconsistencies
    if reasoning_suggests_correct and final_grade == "B":
        validation_result["reasoning_grade_consistent"] = False
        validation_result["inconsistency_type"] = "reasoning_says_correct_grade_says_incorrect"
        validation_result["suggested_grade"] = "A"
        validation_result["validation_passed"] = False

    elif reasoning_suggests_incorrect and final_grade == "A":
        validation_result["reasoning_grade_consistent"] = False
        validation_result["inconsistency_type"] = "reasoning_says_incorrect_grade_says_correct"
        validation_result["suggested_grade"] = "B"
        validation_result["validation_passed"] = False

    # Check for abstention misclassification
    if validation_result["confidence_signals_detected"] and any("abstention" in signal for signal in validation_result["confidence_signals_detected"]):
        # If strong abstention signals but graded as incorrect, may need review
        if final_grade == "B" and not reasoning_suggests_incorrect:
            validation_result["reasoning_indicators"].append("possible_abstention_misclassified_as_incorrect")

    return validation_result


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
        confidence_thresholds: List[ConfidenceThreshold] = None,
        use_flex_tier: bool = False,
        enable_validation: bool = True
    ):
        # Use GPT-5 as default grader for improved reliability
        if grader_model is None:
            if use_flex_tier:
                grader_model = ChatCompletionSampler(
                    model="gpt-5",
                    service_tier="flex",
                    temperature=0,  # Deterministic evaluation
                    seed=42,  # Fixed seed for deterministic judge behavior
                    reasoning_effort="minimal"  # Fast, consistent responses
                )
            else:
                grader_model = ChatCompletionSampler(
                    model="gpt-5",
                    temperature=0,  # Deterministic evaluation
                    seed=42,  # Fixed seed for deterministic judge behavior
                    reasoning_effort="minimal",  # Fast, consistent responses
                    # Add structured output support
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "judge_evaluation",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Step-by-step analysis of the predicted answer against the gold target"
                                    },
                                    "grade": {
                                        "type": "string",
                                        "enum": ["A", "B"],
                                        "description": "A for CORRECT, B for INCORRECT"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Confidence in this evaluation (0.0 to 1.0)"
                                    },
                                    "consistency_check": {
                                        "type": "string",
                                        "description": "Verification that reasoning supports the chosen grade"
                                    }
                                },
                                "required": ["reasoning", "grade", "confidence", "consistency_check"],
                                "additionalProperties": False
                            }
                        }
                    }
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

        # Initialize blind evaluation system - provider anonymization
        # Use class variable to persist mapping across instances
        if not hasattr(ConfidenceThresholdSimpleQAEval, '_global_provider_anonymization'):
            ConfidenceThresholdSimpleQAEval._global_provider_anonymization = {}
        if not hasattr(ConfidenceThresholdSimpleQAEval, '_global_blind_evaluation_enabled'):
            ConfidenceThresholdSimpleQAEval._global_blind_evaluation_enabled = True

        self.provider_anonymization = ConfidenceThresholdSimpleQAEval._global_provider_anonymization
        self.blind_evaluation_enabled = ConfidenceThresholdSimpleQAEval._global_blind_evaluation_enabled

        # Store validation configuration
        self.enable_validation = enable_validation

        # Initialize validation statistics for monitoring (audit-only)
        self.validation_stats = {
            "total_validations": 0,
            "audit_flags": 0,  # Cases where audit flagged concerns
            "audit_consistent": 0,  # Cases where audit found no issues
            "validation_errors": 0,
            "high_confidence_flags": 0,  # High confidence audit flags
            "low_confidence_flags": 0,   # Low confidence audit flags
            "validation_bypassed": 0
        }

    @classmethod
    def reset_blind_evaluation(cls):
        """Reset the global blind evaluation system for a new benchmark run"""
        cls._global_provider_anonymization = {}
        cls._global_blind_evaluation_enabled = True
        print("🔒 Blind evaluation system reset for new benchmark run")

    def _create_clean_prompt(self, question: str) -> str:
        """Create a clean question prompt without threshold instructions"""
        return question

    def _get_anonymous_provider_id(self, provider_name: str) -> str:
        """Get or create anonymous ID for provider to enable blind evaluation"""
        if not self.blind_evaluation_enabled:
            return provider_name

        if provider_name not in self.provider_anonymization:
            # Create deterministic but anonymous ID
            provider_count = len(self.provider_anonymization) + 1
            anon_id = f"Provider_{provider_count:02d}"
            self.provider_anonymization[provider_name] = anon_id
            print(f"   🔒 Blind evaluation: {provider_name} → {anon_id}")

        return self.provider_anonymization[provider_name]

    def reveal_provider_mapping(self) -> Dict[str, str]:
        """Reveal the provider anonymization mapping for final reporting"""
        if not self.blind_evaluation_enabled or not self.provider_anonymization:
            return {}

        print("\n🔓 Revealing Provider Anonymization Mapping:")
        print("=" * 50)
        mapping = {}
        for real_name, anon_id in self.provider_anonymization.items():
            print(f"   {anon_id} → {real_name}")
            mapping[anon_id] = real_name
        print("=" * 50)
        return mapping

    def _extract_judge_model_config(self) -> Dict[str, any]:
        """Extract judge model configuration for audit logging"""
        if not hasattr(self.grader_model, '__dict__'):
            return {"model": "UNKNOWN", "error": "No model attributes accessible"}

        config = {}

        # Extract key parameters from the judge model
        if hasattr(self.grader_model, 'model'):
            config["model"] = self.grader_model.model
        if hasattr(self.grader_model, 'temperature'):
            config["temperature"] = self.grader_model.temperature
        if hasattr(self.grader_model, 'response_format'):
            config["response_format"] = getattr(self.grader_model, 'response_format', None)
        if hasattr(self.grader_model, 'service_tier'):
            config["service_tier"] = getattr(self.grader_model, 'service_tier', None)
        if hasattr(self.grader_model, 'max_tokens'):
            config["max_tokens"] = getattr(self.grader_model, 'max_tokens', None)
        if hasattr(self.grader_model, 'top_p'):
            config["top_p"] = getattr(self.grader_model, 'top_p', None)
        if hasattr(self.grader_model, 'seed'):
            config["seed"] = getattr(self.grader_model, 'seed', None)
        if hasattr(self.grader_model, 'reasoning_effort'):
            config["reasoning_effort"] = getattr(self.grader_model, 'reasoning_effort', None)

        # Add timestamp for when config was captured
        config["config_captured_at"] = datetime.utcnow().isoformat()

        return config

    def print_judge_configuration_summary(self):
        """Print complete judge configuration for transparency"""
        config = self._extract_judge_model_config()
        print("\n🔧 JUDGE MODEL CONFIGURATION (Full Transparency for Independent Audit):")
        print("=" * 80)
        for key, value in config.items():
            if key == "response_format" and isinstance(value, dict):
                print(f"   {key}: {value.get('type', 'N/A')} ({value.get('json_schema', {}).get('name', 'N/A')})")
            else:
                print(f"   {key}: {value}")
        print("=" * 80)

    def validate_judge_consistency(self, sample_responses: List[Dict[str, any]], threshold: ConfidenceThreshold, n_runs: int = 3, sample_size: int = 10) -> Dict[str, any]:
        """
        Validate judge consistency by re-evaluating a sample of responses multiple times
        This detects non-deterministic behavior that could indicate bias or technical issues
        """
        print(f"\n🔍 JUDGE CONSISTENCY VALIDATION:")
        print(f"   Testing {sample_size} responses with {n_runs} runs each")
        print(f"   Expected: 100% consistency with temperature=0.0")

        # Select random sample of responses to test
        import random
        rng = random.Random(42)  # Fixed seed for reproducibility
        sample = rng.sample(sample_responses, min(sample_size, len(sample_responses)))

        consistency_results = []
        total_inconsistencies = 0

        for i, response_data in enumerate(sample):
            print(f"   📊 Testing response {i+1}/{len(sample)}: {response_data['question_id']}")

            # Run multiple evaluations of the same response
            evaluations = []
            for run in range(n_runs):
                try:
                    result = self.grade_sample_with_explanation(
                        question_id=f"{response_data['question_id']}_consistency_run_{run+1}",
                        question=response_data["question"],
                        target=response_data["target"],
                        predicted_answer=response_data["response"],
                        provider_name=f"ConsistencyTest_{response_data.get('provider_name', 'Unknown')}",
                        threshold=threshold
                    )
                    evaluations.append({
                        "run": run + 1,
                        "grade": result["grade_letter"],
                        "reasoning": result["reasoning"],
                        "latency_ms": result["latency_ms"]
                    })
                except Exception as e:
                    print(f"      ❌ Run {run+1} failed: {e}")
                    evaluations.append({
                        "run": run + 1,
                        "grade": "ERROR",
                        "reasoning": f"Evaluation failed: {str(e)}",
                        "latency_ms": 0
                    })

            # Analyze consistency
            grades = [e["grade"] for e in evaluations if e["grade"] != "ERROR"]
            is_consistent = len(set(grades)) <= 1

            if not is_consistent:
                total_inconsistencies += 1
                print(f"      ⚠️  INCONSISTENT: Grades={grades}")
            else:
                print(f"      ✅ Consistent: Grade={grades[0] if grades else 'N/A'}")

            consistency_results.append({
                "question_id": response_data["question_id"],
                "question": response_data["question"],
                "target": response_data["target"],
                "predicted_answer": response_data["response"],
                "evaluations": evaluations,
                "is_consistent": is_consistent,
                "unique_grades": list(set(grades)),
                "grade_distribution": {grade: grades.count(grade) for grade in set(grades)}
            })

        # Calculate consistency statistics
        consistent_responses = sum(1 for r in consistency_results if r["is_consistent"])
        consistency_rate = consistent_responses / len(consistency_results) if consistency_results else 0

        consistency_summary = {
            "total_responses_tested": len(consistency_results),
            "consistent_responses": consistent_responses,
            "inconsistent_responses": total_inconsistencies,
            "consistency_rate": consistency_rate,
            "runs_per_response": n_runs,
            "expected_consistency_rate": 1.0,  # Should be 100% with temperature=0.0
            "consistency_gap": 1.0 - consistency_rate,
            "detailed_results": consistency_results
        }

        # Print summary
        print(f"\n📊 CONSISTENCY VALIDATION RESULTS:")
        print(f"   Responses tested: {len(consistency_results)}")
        print(f"   Consistent responses: {consistent_responses}")
        print(f"   Inconsistent responses: {total_inconsistencies}")
        print(f"   Consistency rate: {consistency_rate:.1%}")

        if consistency_rate < 1.0:
            print(f"   🚨 CRITICAL: Judge showing non-deterministic behavior!")
            print(f"   🚨 Expected 100% consistency with temperature=0.0")
            print(f"   🚨 {total_inconsistencies} responses had inconsistent grades")
        else:
            print(f"   ✅ EXCELLENT: Judge is fully consistent")

        # Log consistency validation results to audit trail
        if self.audit_logger:
            with self._audit_lock:
                self.audit_logger.log_judge_consistency_validation(
                    consistency_summary=consistency_summary,
                    run_timestamp=datetime.utcnow().isoformat()
                )

        return consistency_summary

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
            print(f"   🔧 Calling judge for question {question_id}...")
            grading_response = self.grader_model(prompt_messages)
            latency_ms = (time.time() - start_time) * 1000
            print(f"   ✅ Judge responded in {latency_ms:.0f}ms")

            # Extract reasoning and grade from JSON response
            reasoning = ""
            grade_letter = "C"  # Default to NOT_ATTEMPTED
            judge_confidence = 0.0
            consistency_check = ""

            try:
                # Parse JSON response from judge
                import json
                judge_data = json.loads(grading_response)
                reasoning = judge_data.get("reasoning", "")
                grade_letter = judge_data.get("grade", "C")
                judge_confidence = judge_data.get("confidence", 0.0)
                consistency_check = judge_data.get("consistency_check", "")
            except json.JSONDecodeError:
                # Fallback to old text parsing if JSON fails
                reasoning_match = re.search(r"ANALYSIS AND REASONING:\s*(.*?)\s*FINAL GRADE:", grading_response, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()

                grade_match = re.search(r"FINAL GRADE:.*?([AB])", grading_response, re.DOTALL)
                if grade_match:
                    grade_letter = grade_match.group(1)
                else:
                    fallback_match = re.search(r"(A|B)", grading_response)
                    if fallback_match:
                        grade_letter = fallback_match.group(0)

            # Store original grade before any corrections
            original_grade = grade_letter

            # Validate judge consistency using structured GPT-5-nano validation
            if self.enable_validation:
                self.validation_stats["total_validations"] += 1

                try:
                    validation_result = validate_judge_reasoning_structured(
                        reasoning=reasoning,
                        grade=grade_letter,
                        question=question,
                        target=target,
                        predicted=predicted_answer
                    )

                    # Track audit results (no overrides, just monitoring)
                    if validation_result.get("validation_passed", True):
                        self.validation_stats["audit_consistent"] += 1
                except Exception as e:
                    print(f"   ⚠️ Structured validation error: {e}")
                    self.validation_stats["validation_errors"] += 1
                    validation_result = {
                        "validation": "CONSISTENT",
                        "validation_passed": True,
                        "confidence": 0.5,
                        "reasoning": f"Validation failed, accepting original grade: {str(e)}",
                        "suggested_grade": grade_letter,
                        "original_grade": grade_letter,
                        "validation_method": "fallback_due_to_error",
                        "validation_error": str(e)
                    }
            else:
                # Validation bypassed - accept original grade
                self.validation_stats["validation_bypassed"] += 1
                validation_result = {
                    "validation": "BYPASSED",
                    "validation_passed": True,
                    "confidence": 1.0,
                    "reasoning": "Validation bypassed by configuration",
                    "suggested_grade": grade_letter,
                    "original_grade": grade_letter,
                    "validation_method": "bypassed"
                }

            # REMOVED: No more overrides - GPT-5 judge decision is FINAL
            # Log audit analysis for transparency but do not change grades
            if validation_result and not validation_result.get("validation_passed", True):
                print(f"   🔍 Audit flagged potential concern (NO OVERRIDE)")
                print(f"   📝 Audit reasoning: {validation_result.get('reasoning', '')}")
                print(f"   📊 Judge confidence: {judge_confidence:.2f}, Audit confidence: {validation_result.get('confidence', 0.0):.2f}")
                print(f"   ✅ GPT-5 judge decision stands: {grade_letter}")

                # Track audit flags for monitoring (no overrides)
                self.validation_stats['audit_flags'] += 1
                audit_confidence = validation_result.get("confidence", 0.0)
                if audit_confidence >= 0.8:
                    self.validation_stats['high_confidence_flags'] += 1
                else:
                    self.validation_stats['low_confidence_flags'] += 1

                # Log audit analysis to trail (no grade changes)
                if self.audit_logger:
                    with self._audit_lock:
                        anon_provider_id = self._get_anonymous_provider_id(provider_name)
                        # Log audit analysis (grade remains unchanged)
                        self.audit_logger.log_judge_validation_override(
                            question_id=question_id,
                            question=question,
                            target_answer=target,
                            predicted_answer=predicted_answer,
                            original_grade=grade_letter,  # Same as final
                            validated_grade=grade_letter,  # No change
                            validation_result=validation_result,
                            judge_reasoning=reasoning,
                            validator_reasoning=validation_result.get("reasoning", ""),
                            metadata={
                                "real_provider_name": provider_name,
                                "blind_provider_id": anon_provider_id,
                                "audit_only": True,  # Flag this as audit-only
                                "judge_confidence": judge_confidence
                            }
                        )

            # Calculate threshold-aware scores (judge can give A, B, or sometimes NOT_ATTEMPTED)
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            # Check if judge determined it was not attempted (grade not A or B)
            is_not_attempted = grade_letter not in ["A", "B"]

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
                    # Use anonymous provider ID for judge evaluation logging to maintain blindness
                    anon_provider_id = self._get_anonymous_provider_id(provider_name)
                    # Extract judge model configuration for complete audit trail
                    judge_config = self._extract_judge_model_config()
                    self.audit_logger.log_judge_evaluation(
                        question_id=question_id,
                        question=question,
                        target_answer=target,
                        provider_responses={anon_provider_id: predicted_answer},
                        judge_prompt=grader_prompt,
                        judge_response=grading_response,
                        grades={anon_provider_id: CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")},
                        reasoning=reasoning,
                        latency_ms=latency_ms,
                        # CRITICAL: Include complete judge model configuration for independent audit
                        judge_model_config=judge_config,
                        # Add metadata to track real provider name for final reporting
                        metadata={"real_provider_name": provider_name, "blind_provider_id": anon_provider_id}
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
                "quality_score": quality_score,
                "judge_validation": {
                    **validation_result,
                    "original_grade": original_grade,
                    "final_grade": grade_letter,
                    "grade_corrected": original_grade != grade_letter
                }
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            if self.audit_logger:
                with self._audit_lock:
                    # Include judge config even in error cases for complete audit trail
                    judge_config = self._extract_judge_model_config()
                    self.audit_logger.log_error(
                        component="confidence_threshold_grader",
                        error=str(e),
                        context={
                            "question_id": question_id,
                            "question": question,
                            "provider": provider_name,
                            "threshold": threshold.name,
                            "judge_config": judge_config
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

        # Convert failed responses to API error entries (NOT abstentions)
        for failed_response in failed_responses:
            row = failed_response["row"]
            api_error_response = {
                "question_id": failed_response["question_id"],
                "question": row.get("problem", ""),
                "target": row.get("answer", ""),
                "response": "",  # Empty but marked as API error
                "prompt_messages": [],
                "provider_name": provider_name,
                "api_error": failed_response["error"],  # Mark as API error, not abstention
                "response_type": "api_error"
            }
            provider_responses.append(api_error_response)

        if failed_responses:
            print(f"   ⚠️  {len(failed_responses)} questions failed due to API errors (not abstentions)")

        print(f"   ✅ Collected {len(provider_responses)} total responses ({len(provider_responses) - len(failed_responses)} successful, {len(failed_responses)} API errors)")
        return provider_responses

    def _evaluate_single_response(self, response_data: Dict[str, any], threshold: ConfidenceThreshold) -> SingleEvalResult:
        """Evaluate a single response against a threshold with proper error/abstention separation"""

        provider_response = response_data["response"]

        # First check if this is an API error (exclude from all scoring)
        if response_data.get("response_type") == "api_error":
            # API error - exclude from scoring entirely
            grading_result = {
                "grade_letter": "E",  # E for Error
                "grade_string": "API_ERROR",
                "reasoning": f"API error occurred: {response_data.get('api_error', 'Unknown error')}",
                "full_response": "",
                "latency_ms": 0,
                "is_correct": False,
                "is_incorrect": False,
                "is_not_attempted": False,
                "is_api_error": True,
                "volume_score": None,  # Excluded from volume calculations
                "quality_score": None,  # Excluded from quality calculations
                "abstention_detected": False,
                "abstention_type": None,
                "api_error": response_data.get("api_error")
            }
        # Check if provider abstained (empty response or explicit abstention)
        elif not provider_response or provider_response.strip() == "":
            # Empty response - could be API error or abstention, check context
            if response_data.get("api_error"):
                # This is actually an API error, not an abstention
                grading_result = {
                    "grade_letter": "E",
                    "grade_string": "API_ERROR",
                    "reasoning": f"API error resulted in empty response: {response_data.get('api_error')}",
                    "full_response": "",
                    "latency_ms": 0,
                    "is_correct": False,
                    "is_incorrect": False,
                    "is_not_attempted": False,
                    "is_api_error": True,
                    "volume_score": None,
                    "quality_score": None,
                    "abstention_detected": False,
                    "abstention_type": None,
                    "api_error": response_data.get("api_error")
                }
            else:
                # True empty response abstention
                grading_result = {
                    "grade_letter": "C",
                    "grade_string": "NOT_ATTEMPTED",
                    "reasoning": "Provider did not provide a response",
                    "full_response": "",
                    "latency_ms": 0,
                    "is_correct": False,
                    "is_incorrect": False,
                    "is_not_attempted": True,
                    "is_api_error": False,
                    "volume_score": 0.0,
                    "quality_score": 0.0,
                    "abstention_detected": True,
                    "abstention_type": "empty_response"
                }
        else:
            # Non-empty response - use intelligent classification
            classification = classify_response_intent(provider_response, response_data["question"])

            if classification["type"] == "abstention":
                # Intelligent abstention detection - give 0 points, don't send to judge
                print(f"   🛡️ Abstention detected: '{provider_response[:60]}...' (confidence: {classification['confidence']:.2f})")

                # Log abstention classifier decision to audit trail
                if self.audit_logger:
                    with self._audit_lock:
                        anon_provider_id = self._get_anonymous_provider_id(response_data["provider_name"])
                        self.audit_logger.log_abstention_classification(
                            question_id=response_data["question_id"],
                            question=response_data["question"],
                            provider_response=provider_response,
                            provider_name=anon_provider_id,  # Use anonymous ID
                            classification_type=classification["type"],
                            confidence=classification["confidence"],
                            reasoning=classification["reasoning"],
                            classifier_model="gpt-5-nano",
                            metadata={"real_provider_name": response_data["provider_name"]}
                        )

                grading_result = {
                    "grade_letter": "C",
                    "grade_string": "NOT_ATTEMPTED",
                    "reasoning": f"Provider abstained: {classification['reasoning']}",
                    "full_response": provider_response,
                    "latency_ms": 0,
                    "is_correct": False,
                    "is_incorrect": False,
                    "is_not_attempted": True,
                    "is_api_error": False,
                    "volume_score": 0.0,
                    "quality_score": 0.0,
                    "abstention_detected": True,
                    "abstention_type": "intelligent_classification",
                    "abstention_confidence": classification["confidence"],
                    "abstention_reasoning": classification["reasoning"]
                }
            else:
                # Response classified as attempt - send to judge
                print(f"   📝 Attempt detected: '{provider_response[:60]}...' (confidence: {classification['confidence']:.2f})")

                # Log attempt classifier decision to audit trail
                if self.audit_logger:
                    with self._audit_lock:
                        anon_provider_id = self._get_anonymous_provider_id(response_data["provider_name"])
                        self.audit_logger.log_abstention_classification(
                            question_id=response_data["question_id"],
                            question=response_data["question"],
                            provider_response=provider_response,
                            provider_name=anon_provider_id,  # Use anonymous ID
                            classification_type=classification["type"],
                            confidence=classification["confidence"],
                            reasoning=classification["reasoning"],
                            classifier_model="gpt-5-nano",
                            metadata={"real_provider_name": response_data["provider_name"]}
                        )

                grading_result = self.grade_sample_with_explanation(
                    question_id=response_data["question_id"],
                    question=response_data["question"],
                    target=response_data["target"],
                    predicted_answer=provider_response,
                    provider_name=response_data["provider_name"],
                    threshold=threshold
                )

                # Add classification metadata for attempts
                grading_result["abstention_detected"] = False
                grading_result["abstention_type"] = None
                grading_result["is_api_error"] = False
                grading_result["attempt_confidence"] = classification["confidence"]
                grading_result["attempt_reasoning"] = classification["reasoning"]

        # Create HTML for each sample result (handle None scores for API errors)
        display_score = grading_result.get("volume_score")
        if display_score is None:
            display_score = "N/A (API Error)"

        html = common.jinja_env.from_string(common.HTML_JINJA).render(
            prompt_messages=response_data["prompt_messages"],
            next_message=dict(content=response_data["response"], role="assistant"),
            score=display_score,
            correct_answer=response_data["target"],
            extracted_answer=response_data["response"] if not grading_result.get("is_api_error") else f"[API ERROR: {grading_result.get('api_error', 'Unknown')}]",
            grading_explanation=grading_result["reasoning"],
            grade=grading_result["grade_string"]
        )

        convo = response_data["prompt_messages"] + [dict(content=response_data["response"], role="assistant")]

        return SingleEvalResult(
            html=html,
            score=grading_result.get("volume_score"),  # May be None for API errors
            convo=convo,
            metrics={
                "is_correct": grading_result["is_correct"],
                "is_incorrect": grading_result["is_incorrect"],
                "is_not_attempted": grading_result["is_not_attempted"],
                "is_api_error": grading_result.get("is_api_error", False),
                "volume_score": grading_result["volume_score"],
                "quality_score": grading_result["quality_score"],
                "judge_latency_ms": grading_result.get("latency_ms", 0),
                "api_error": grading_result.get("api_error"),
                "abstention_detected": grading_result.get("abstention_detected", False),
                "abstention_type": grading_result.get("abstention_type"),
                # Store classification metadata
                "abstention_confidence": grading_result.get("abstention_confidence"),
                "attempt_confidence": grading_result.get("attempt_confidence")
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

    def __call__(self, sampler: SamplerBase, provider_name: str = None) -> EvalResult:
        """
        Run evaluation against the single 80% confidence threshold
        """
        if provider_name is None:
            provider_name = sampler.__class__.__name__.replace("Sampler", "").replace("Audited", "")

        print(f"\n🎯 Running Quality-Based Evaluation for {provider_name}")
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

        # Use the single threshold (80% confidence)
        threshold = self.confidence_thresholds[0]
        print(f"\n📊 Evaluating with {threshold.name} methodology (t={threshold.threshold}, penalty={threshold.penalty_ratio})")
        print(f"   Evaluation criteria: {threshold.description}")

        # Evaluate the natural responses against the threshold
        results = self.evaluate_single_threshold(provider_responses, threshold)

        # Calculate metrics
        aggregate_metrics = self._calculate_threshold_metrics(results, threshold)

        # Run judge consistency validation on a sample of provider responses
        print(f"\n🔍 Running judge consistency validation...")
        try:
            consistency_validation = self.validate_judge_consistency(
                sample_responses=provider_responses,
                threshold=threshold,
                n_runs=3,  # Test each response 3 times
                sample_size=min(10, len(provider_responses))  # Test up to 10 responses
            )
            # Add consistency metrics to the results
            aggregate_metrics["judge_consistency"] = {
                "consistency_rate": consistency_validation["consistency_rate"],
                "inconsistent_responses": consistency_validation["inconsistent_responses"],
                "total_tested": consistency_validation["total_responses_tested"],
                "consistency_gap": consistency_validation["consistency_gap"]
            }
        except Exception as e:
            print(f"   ⚠️ Consistency validation failed: {e}")
            aggregate_metrics["judge_consistency"] = {
                "consistency_rate": None,
                "inconsistent_responses": None,
                "total_tested": 0,
                "error": str(e)
            }

        # Create EvalResult
        base_result = common.aggregate_results(results)
        base_result.metrics.update(aggregate_metrics)

        # Print summary
        print(f"   📈 Results:")
        print(f"      Volume Score (traditional): {aggregate_metrics['volume_score_mean']:.3f}")
        print(f"      Quality Score (penalty-aware): {aggregate_metrics['quality_score_mean']:.3f}")
        print(f"      Attempted Rate: {aggregate_metrics['attempted_rate']:.3f}")
        print(f"      Success on Attempted: {aggregate_metrics['accuracy_given_attempted']:.3f}")

        return base_result

    def analyze_statistical_significance(self, provider_results: Dict[str, EvalResult]) -> Dict[str, any]:
        """
        Perform cross-provider statistical analysis
        Returns statistical comparisons between providers
        """
        statistical_analysis = {
            "pairwise_comparisons": {},
            "effect_sizes": {},
            "grade_distribution_tests": {},
            "summary": {}
        }

        providers = list(provider_results.keys())

        # Get data for all providers
        provider_data = {}
        for provider in providers:
            result = provider_results[provider]
            provider_data[provider] = {
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
                    provider_data[provider1]["volume_scores"],
                    provider_data[provider2]["volume_scores"],
                    "Volume Score"
                )

                quality_comparison = self._compare_continuous_metrics(
                    provider_data[provider1]["quality_scores"],
                    provider_data[provider2]["quality_scores"],
                    "Quality Score"
                )

                statistical_analysis["pairwise_comparisons"][comparison_key] = {
                    "volume_score": volume_comparison,
                    "quality_score": quality_comparison
                }

                # Effect sizes
                volume_effect = cohens_d(
                    provider_data[provider1]["volume_scores"],
                    provider_data[provider2]["volume_scores"]
                )
                quality_effect = cohens_d(
                    provider_data[provider1]["quality_scores"],
                    provider_data[provider2]["quality_scores"]
                )

                statistical_analysis["effect_sizes"][comparison_key] = {
                    "volume_score_cohens_d": volume_effect,
                    "volume_score_interpretation": interpret_effect_size(volume_effect, "cohens_d"),
                    "quality_score_cohens_d": quality_effect,
                    "quality_score_interpretation": interpret_effect_size(quality_effect, "cohens_d")
                }

                # Chi-square test for grade distributions
                chi2_result = self._compare_grade_distributions(
                    provider_data[provider1],
                    provider_data[provider2],
                    provider1,
                    provider2
                )

                statistical_analysis["grade_distribution_tests"][comparison_key] = chi2_result

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

        for comparison in analysis["pairwise_comparisons"]:
            # Volume score p-values
            vol_p = analysis["pairwise_comparisons"][comparison]["volume_score"].get("p_value")
            if vol_p is not None:
                all_p_values.append(vol_p)
                comparison_types.append(f"{comparison}_volume")

            # Quality score p-values
            qual_p = analysis["pairwise_comparisons"][comparison]["quality_score"].get("p_value")
            if qual_p is not None:
                all_p_values.append(qual_p)
                comparison_types.append(f"{comparison}_quality")

            # Distribution test p-values
            dist_p = analysis["grade_distribution_tests"][comparison].get("p_value")
            if dist_p is not None:
                all_p_values.append(dist_p)
                comparison_types.append(f"{comparison}_distribution")

        # Apply Bonferroni correction
        if all_p_values:
            corrected_significant, corrected_alpha = apply_bonferroni_correction(all_p_values)
        else:
            corrected_significant = []
            corrected_alpha = 0.05

        summary = {
            "total_comparisons": len(analysis["pairwise_comparisons"]),
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
        for comparison in analysis["pairwise_comparisons"]:
            # Volume score results
            vol_data = analysis["pairwise_comparisons"][comparison]["volume_score"]
            if vol_data.get("p_value") is not None:
                if vol_data.get("significant_raw", False):
                    summary["significant_volume_comparisons_raw"] += 1
                if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                    summary["significant_volume_comparisons_corrected"] += 1
                p_idx += 1

            # Quality score results
            qual_data = analysis["pairwise_comparisons"][comparison]["quality_score"]
            if qual_data.get("p_value") is not None:
                if qual_data.get("significant_raw", False):
                    summary["significant_quality_comparisons_raw"] += 1
                if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                    summary["significant_quality_comparisons_corrected"] += 1
                p_idx += 1

            # Distribution test results
            dist_data = analysis["grade_distribution_tests"][comparison]
            if dist_data.get("p_value") is not None:
                if dist_data.get("significant_raw", False):
                    summary["significant_distribution_comparisons_raw"] += 1
                if p_idx < len(corrected_significant) and corrected_significant[p_idx]:
                    summary["significant_distribution_comparisons_corrected"] += 1
                p_idx += 1

            # Count effect sizes
            effect_data = analysis["effect_sizes"][comparison]
            for interpretation in [effect_data["volume_score_interpretation"], effect_data["quality_score_interpretation"]]:
                if interpretation == "large":
                    summary["large_effect_sizes"] += 1
                elif interpretation == "medium":
                    summary["medium_effect_sizes"] += 1
                elif interpretation == "small":
                    summary["small_effect_sizes"] += 1

        return summary

    def _calculate_threshold_metrics(self, results: List[SingleEvalResult], threshold: ConfidenceThreshold) -> Dict[str, float]:
        """Calculate comprehensive metrics for a threshold with proper API error handling"""
        n_total = len(results)

        # Separate API errors from valid responses
        api_error_results = [r for r in results if r.metrics.get("is_api_error", False)]
        valid_results = [r for r in results if not r.metrics.get("is_api_error", False)]

        n_api_errors = len(api_error_results)
        n_valid = len(valid_results)

        # Basic counts from VALID responses only (API errors excluded)
        n_correct = sum(r.metrics["is_correct"] for r in valid_results)
        n_incorrect = sum(r.metrics["is_incorrect"] for r in valid_results)
        n_not_attempted = sum(r.metrics["is_not_attempted"] for r in valid_results)
        n_attempted = n_correct + n_incorrect  # Only count actual attempts

        # Volume strategy metrics (traditional) - calculated on valid responses only
        valid_volume_scores = [r.metrics["volume_score"] for r in valid_results if r.metrics["volume_score"] is not None]
        volume_score_mean = sum(valid_volume_scores) / len(valid_volume_scores) if valid_volume_scores else 0

        # Quality strategy metrics (penalty-aware) - calculated on valid responses only
        valid_quality_scores = [r.metrics["quality_score"] for r in valid_results if r.metrics["quality_score"] is not None]
        quality_score_mean = sum(valid_quality_scores) / len(valid_quality_scores) if valid_quality_scores else 0

        # Behavioral metrics (based on valid responses, excluding API errors)
        attempted_rate = n_attempted / n_valid if n_valid > 0 else 0
        abstention_rate = n_not_attempted / n_valid if n_valid > 0 else 0
        accuracy_given_attempted = n_correct / n_attempted if n_attempted > 0 else 0
        error_rate_given_attempted = n_incorrect / n_attempted if n_attempted > 0 else 0

        # API error metrics
        api_error_rate = n_api_errors / n_total if n_total > 0 else 0

        # Conservative strategy analysis (from the paper)
        conservative_penalty = n_incorrect * threshold.penalty_ratio
        conservative_benefit = n_not_attempted * 0  # Abstention = 0 points

        # Overconfidence penalty (questions answered incorrectly when should have abstained)
        overconfidence_penalty = n_incorrect

        # Judge performance metrics (only for responses that went to judge)
        judge_latencies = [r.metrics["judge_latency_ms"] for r in results if r.metrics.get("judge_latency_ms") and r.metrics["judge_latency_ms"] > 0]
        avg_judge_latency_ms = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0

        # Statistical analysis - confidence intervals for key metrics (valid responses only)
        volume_scores = [r.metrics["volume_score"] for r in valid_results if r.metrics.get("volume_score") is not None]
        quality_scores = [r.metrics["quality_score"] for r in valid_results if r.metrics.get("quality_score") is not None]

        # Calculate confidence intervals (only for valid responses)
        volume_ci = calculate_confidence_interval(volume_scores) if volume_scores else (0, 0, 0)
        quality_ci = calculate_confidence_interval(quality_scores) if quality_scores else (0, 0, 0)

        # Binary outcome confidence intervals (using binomial distribution, valid responses only)
        attempted_scores = [1 if r.metrics["is_correct"] or r.metrics["is_incorrect"] else 0 for r in valid_results]
        attempted_ci = calculate_confidence_interval(attempted_scores) if attempted_scores else (0, 0, 0)

        accuracy_attempted_scores = [r.metrics["is_correct"] for r in valid_results if r.metrics["is_correct"] or r.metrics["is_incorrect"]]
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

            # Raw counts (valid responses only)
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_not_attempted": n_not_attempted,
            "n_attempted": n_attempted,
            "n_valid": n_valid,  # Valid responses (excluding API errors)
            "n_total": n_total,  # Total including API errors

            # API error metrics
            "n_api_errors": n_api_errors,
            "api_error_rate": api_error_rate,

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

            # Legacy compatibility (based on valid responses, excluding API errors)
            "is_correct": n_correct / n_valid if n_valid > 0 else 0,
            "is_incorrect": n_incorrect / n_valid if n_valid > 0 else 0,
            "is_not_attempted": n_not_attempted / n_valid if n_valid > 0 else 0,
        }