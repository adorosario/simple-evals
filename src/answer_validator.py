"""
Answer Validator for SimpleQA-Verified Knowledge Base

Uses hybrid approach:
1. Fast string matching with answer variants
2. LLM fallback for cases where string matching fails

This validates that the knowledge base actually contains
the answers needed for RAG evaluation.
"""

import re
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single question's answer in content."""
    question_index: int
    question: str
    answer: str
    found: bool
    method: str  # "string_match", "llm_verified", "llm_partial", "not_found"
    matched_variant: Optional[str] = None
    llm_explanation: Optional[str] = None
    content_preview: Optional[str] = None


@dataclass
class ValidationReport:
    """Aggregate validation report."""
    total_questions: int = 0
    answers_found: int = 0
    answers_not_found: int = 0
    by_method: Dict[str, int] = field(default_factory=dict)
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def coverage_rate(self) -> float:
        return self.answers_found / max(self.total_questions, 1)

    def to_dict(self) -> dict:
        return {
            'summary': {
                'total_questions': self.total_questions,
                'answers_found': self.answers_found,
                'answers_not_found': self.answers_not_found,
                'coverage_rate': self.coverage_rate,
                'by_method': self.by_method,
            },
            'not_found_details': [
                {
                    'question_index': r.question_index,
                    'question': r.question,
                    'answer': r.answer,
                    'content_preview': r.content_preview[:300] if r.content_preview else None,
                }
                for r in self.results if not r.found
            ],
            'all_results': [
                {
                    'question_index': r.question_index,
                    'found': r.found,
                    'method': r.method,
                    'matched_variant': r.matched_variant,
                }
                for r in self.results
            ]
        }


class AnswerValidator:
    """
    Validates that KB content contains answers to benchmark questions.

    Uses hybrid approach:
    1. Generate normalized answer variants
    2. Try fast string matching
    3. If no match, use LLM to verify
    """

    def __init__(
        self,
        use_llm_fallback: bool = True,
        llm_model: str = "gpt-4.1",
        api_key: Optional[str] = None,
    ):
        """
        Initialize answer validator.

        Args:
            use_llm_fallback: Whether to use LLM for failed string matches
            llm_model: OpenAI model for LLM validation
            api_key: OpenAI API key (or from env)
        """
        self.use_llm_fallback = use_llm_fallback
        self.llm_model = llm_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._client is None and self.use_llm_fallback:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI not installed, LLM fallback disabled")
                self.use_llm_fallback = False
        return self._client

    def generate_answer_variants(self, answer: str) -> List[str]:
        """
        Generate normalized variants of an answer for matching.

        Args:
            answer: Original answer string

        Returns:
            List of variants to search for
        """
        variants = [answer]
        answer_lower = answer.lower().strip()

        # Add lowercase version
        variants.append(answer_lower)

        # Handle numbers with commas
        # "120,000" -> "120000", "120 000"
        if re.search(r'\d{1,3}(,\d{3})+', answer):
            no_comma = re.sub(r',', '', answer)
            variants.append(no_comma)
            variants.append(no_comma.lower())

        # Handle currency
        # "120,000 euros" -> "€120,000", "120000 EUR"
        currency_patterns = [
            (r'(\d+(?:,?\d+)*)\s*euros?', ['€{0}', '{0} EUR', '{0} euro']),
            (r'(\d+(?:,?\d+)*)\s*dollars?', ['${0}', '{0} USD', '{0} dollar']),
            (r'(\d+(?:,?\d+)*)\s*pounds?', ['£{0}', '{0} GBP', '{0} pound']),
            (r'\$(\d+(?:,?\d+)*)', ['{0} dollars', '{0} USD']),
            (r'€(\d+(?:,?\d+)*)', ['{0} euros', '{0} EUR']),
            (r'£(\d+(?:,?\d+)*)', ['{0} pounds', '{0} GBP']),
        ]

        for pattern, templates in currency_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                num = match.group(1)
                for tmpl in templates:
                    variants.append(tmpl.format(num))
                    variants.append(tmpl.format(num.replace(',', '')))

        # Handle dates
        # "January 1, 2020" -> "Jan 1, 2020", "1/1/2020", "2020-01-01"
        date_patterns = [
            # Full month name to abbreviation
            (r'(January|February|March|April|May|June|July|August|September|October|November|December)',
             {'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
              'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
              'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'}),
        ]

        for pattern, mapping in date_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                month = match.group(1)
                abbrev = mapping.get(month.title(), month)
                variants.append(answer.replace(month, abbrev))

        # Reverse: Abbreviation to full month name (e.g., "Oct" -> "October")
        abbrev_to_full = {
            'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
            'Apr': 'April', 'Jun': 'June', 'Jul': 'July',
            'Aug': 'August', 'Sep': 'September', 'Oct': 'October',
            'Nov': 'November', 'Dec': 'December',
        }
        for abbrev, full in abbrev_to_full.items():
            # Use word boundary to avoid matching "March" when looking for "Mar"
            if re.search(rf'\b{abbrev}\b', answer, re.IGNORECASE):
                variants.append(re.sub(rf'\b{abbrev}\b', full, answer, flags=re.IGNORECASE))

        # Handle ordinals
        # "1st" -> "first", "2nd" -> "second"
        ordinal_map = {
            '1st': 'first', '2nd': 'second', '3rd': 'third',
            '4th': 'fourth', '5th': 'fifth', '6th': 'sixth',
            '7th': 'seventh', '8th': 'eighth', '9th': 'ninth',
            '10th': 'tenth',
        }
        for ordinal, word in ordinal_map.items():
            if ordinal.lower() in answer_lower:
                variants.append(answer_lower.replace(ordinal.lower(), word))

        # Handle names (first name only, last name only)
        # "John Smith" -> "John", "Smith"
        name_parts = answer.split()
        if 2 <= len(name_parts) <= 4:
            # Check if it looks like a name (capitalized words)
            if all(p[0].isupper() for p in name_parts if p):
                variants.extend(name_parts)  # Individual parts
                if len(name_parts) == 2:
                    variants.append(f"{name_parts[1]}, {name_parts[0]}")  # Last, First

        # Strip leading articles and pronouns for matching
        # "The right arm" -> "right arm", "Her right arm" -> "right arm"
        article_stripped = re.sub(
            r'^(the|a|an|his|her|their|its)\s+', '', answer, flags=re.IGNORECASE
        ).strip()
        if article_stripped and article_stripped.lower() != answer_lower:
            variants.append(article_stripped)

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_normalized = self._normalize_for_matching(v)
            if v_normalized and v_normalized not in seen:
                seen.add(v_normalized)
                unique_variants.append(v)

        return unique_variants

    def _normalize_for_matching(self, text: str) -> str:
        """Normalize text for matching."""
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation except for important ones
        text = re.sub(r'[^\w\s€$£]', '', text)
        return text.strip()

    def string_match(self, answer: str, content: str) -> Tuple[bool, Optional[str]]:
        """
        Try to find answer in content using string matching.

        Args:
            answer: Expected answer
            content: KB content to search

        Returns:
            (found, matched_variant)
        """
        variants = self.generate_answer_variants(answer)
        normalized_content = self._normalize_for_matching(content)

        for variant in variants:
            normalized_variant = self._normalize_for_matching(variant)
            if normalized_variant and normalized_variant in normalized_content:
                return True, variant

        return False, None

    def llm_validate(self, answer: str, content: str, question: str = "") -> Tuple[str, str]:
        """
        Use LLM to validate if content contains the answer.

        Args:
            answer: Expected answer
            content: KB content to search
            question: Original question (for context)

        Returns:
            (result: "YES"|"NO"|"PARTIAL", explanation)
        """
        if not self.openai_client:
            return "NO", "LLM validation not available"

        # Truncate content if too long
        # Increased from 4000 to 32000 - answers often appear beyond 4K bytes
        # gpt-4o-mini has 128K context, gpt-4.1 has 1M - no reason for tiny limit
        content_truncated = content[:32000] if len(content) > 32000 else content

        prompt = f"""Determine if the following content contains or implies the expected answer.

Question: {question}
Expected Answer: {answer}

Content to search:
---
{content_truncated}
---

Does the content contain the answer "{answer}" (exactly or semantically equivalent)?

Reply with ONE of:
- YES: The content clearly contains or states the answer
- PARTIAL: The content mentions related information but not the exact answer
- NO: The content does not contain the answer

Then briefly explain your reasoning in 1-2 sentences."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            result_text = response.choices[0].message.content.strip()

            # Parse result
            if result_text.startswith("YES"):
                return "YES", result_text
            elif result_text.startswith("PARTIAL"):
                return "PARTIAL", result_text
            else:
                return "NO", result_text

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return "ERROR", str(e)

    def validate(
        self,
        question_index: int,
        question: str,
        answer: str,
        content: str,
    ) -> ValidationResult:
        """
        Validate if answer appears in content (hybrid approach).

        Args:
            question_index: Index of question
            question: Question text
            answer: Expected answer
            content: KB content to search

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(
            question_index=question_index,
            question=question,
            answer=answer,
            found=False,
            method="not_checked",
            content_preview=content[:200] if content else None,
        )

        # Step 1: Try string matching
        found, variant = self.string_match(answer, content)
        if found:
            result.found = True
            result.method = "string_match"
            result.matched_variant = variant
            return result

        # Step 2: LLM fallback
        if self.use_llm_fallback:
            llm_result, explanation = self.llm_validate(answer, content, question)

            if llm_result == "YES":
                result.found = True
                result.method = "llm_verified"
                result.llm_explanation = explanation
            elif llm_result == "PARTIAL":
                result.found = True
                result.method = "llm_partial"
                result.llm_explanation = explanation
            else:
                result.found = False
                result.method = "not_found"
                result.llm_explanation = explanation
        else:
            result.method = "not_found"

        return result

    def validate_batch(
        self,
        questions: List[Dict],
        contents: List[str],
        progress_callback=None,
    ) -> ValidationReport:
        """
        Validate multiple questions against their content.

        Args:
            questions: List of dicts with 'index', 'question', 'answer' keys
            contents: Corresponding content for each question
            progress_callback: Optional callback(current, total)

        Returns:
            ValidationReport with all results
        """
        report = ValidationReport()
        report.total_questions = len(questions)
        report.by_method = {
            'string_match': 0,
            'llm_verified': 0,
            'llm_partial': 0,
            'not_found': 0,
        }

        for i, (q, content) in enumerate(zip(questions, contents)):
            result = self.validate(
                question_index=q.get('index', i),
                question=q.get('question', ''),
                answer=q.get('answer', ''),
                content=content,
            )
            report.results.append(result)

            if result.found:
                report.answers_found += 1
            else:
                report.answers_not_found += 1

            if result.method in report.by_method:
                report.by_method[result.method] += 1

            if progress_callback:
                progress_callback(i + 1, len(questions))

        return report

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test with example
    validator = AnswerValidator(use_llm_fallback=False)  # Disable LLM for quick test

    # Test variant generation
    test_answers = [
        "120,000 euros",
        "John Smith",
        "January 1, 2020",
        "1st place",
        "$5,000,000",
    ]

    print("Answer Variant Generation Test")
    print("=" * 60)

    for answer in test_answers:
        variants = validator.generate_answer_variants(answer)
        print(f"\nAnswer: {answer}")
        print(f"Variants: {variants}")

    # Test string matching
    print("\n" + "=" * 60)
    print("String Matching Test")
    print("=" * 60)

    test_cases = [
        ("120,000 euros", "The surgeon was ordered to pay €120000 to the family."),
        ("John Smith", "According to Smith, John, the experiment was successful."),
        ("Paris", "The city of Paris is known for the Eiffel Tower."),
        ("Not in content", "This content mentions nothing relevant."),
    ]

    for answer, content in test_cases:
        found, variant = validator.string_match(answer, content)
        print(f"\nAnswer: {answer}")
        print(f"Content: {content[:50]}...")
        print(f"Found: {found}, Variant: {variant}")
