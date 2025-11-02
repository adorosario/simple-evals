# Judge Confidence Score Analysis & Investigation Priorities

**Date:** October 23, 2025
**Run Analyzed:** run_20251023_014936_503
**Provider:** CustomGPT_RAG (191 evaluations)

---

## Executive Summary

Analysis of judge confidence scores reveals they are **semantically correct** and provide valuable signals for prioritizing quality improvement efforts. Contrary to intuition, **lower confidence INCORRECT** cases should be investigated first, as they represent edge cases and potential grading issues rather than clear-cut failures.

---

## What Judge Confidence Means

Judge confidence represents: **"How confident am I in this grading decision?"**

- **High confidence (0.95-1.0):** Clear-cut case with unambiguous evidence
- **Medium confidence (0.90-0.94):** Reasonable decision but some ambiguity exists
- **Lower confidence (0.85-0.89):** Edge case with legitimate uncertainty

**Important:** High confidence on an INCORRECT grade means "I'm very sure this answer is WRONG" (not "I'm unsure if it's wrong").

---

## Empirical Data: CustomGPT_RAG Performance

### Overall Distribution

| Grade | Count | Avg Confidence | Min | Max |
|-------|-------|----------------|-----|-----|
| **CORRECT** | 181 | **98.0%** | 86% | 100% |
| **INCORRECT** | 10 | **94.9%** | 90% | 97% |

### Key Observations

1. **Judge is highly confident overall** (94.9%-98% avg) - appropriate for factual Q&A
2. **INCORRECT cases have slightly lower confidence** (94.9% vs 98%) - expected for borderline failures
3. **Confidence range for CORRECT** is wider (86%-100%) - reflects varying answer quality

---

## Investigation Priority Tiers

### 🔴 **TIER 1 - LOW Confidence INCORRECT (Priority: HIGHEST)**

**Confidence Range:** 0.90 (90%)
**Count:** 3 cases
**Why Investigate First:** Most likely to reveal grading rule issues, false negatives, and evaluation criteria gaps.

#### Examples:

**simpleqa_0007** (Confidence: 0.90)
- **Question:** What day, month, and year did Dylan Sprouse marry Barbara Palvin?
- **Target:** 15 of July of 2023
- **CustomGPT Answer:** "Dylan Sprouse married Barbara Palvin in July 2023... The specific day is not provided in my knowledge base."
- **Judge Reasoning:** "States July 2023 and explicitly says the specific day is not provided. It lacks the day, which is key information. While it doesn't contradict, it is incomplete."
- **Investigation Point:** Is incomplete information wrong? Should partial answers receive partial credit?

**simpleqa_0102** (Confidence: 0.90)
- **Question:** What issue was Black Condor II killed in?
- **Target:** Infinite Crisis #1
- **CustomGPT Answer:** "Black Condor II, Ryan Kendall, was killed at the beginning of the Infinite Crisis storyline. My knowledge base does not specify the exact issue..."
- **Judge Reasoning:** "Gold target specifies the exact issue: Infinite Crisis #1. Predicted answer only says it was at the beginning of the Infinite Crisis storyline."
- **Investigation Point:** Vague but directionally correct vs. specific and precise - how should these be scored?

**simpleqa_0146** (Confidence: 0.90)
- **Question:** In what year did the Pakistani men's national field hockey team not participate in FIH World Cup?
- **Target:** 2014
- **CustomGPT Answer:** "As of 2022, the Pakistani men's national field hockey team did not participate in the FIH World Cup in the years 2014 and 2023."
- **Judge Reasoning:** "Gold target states only 2014. Predicted answer lists 2014 and 2023. Including 2023 adds information not in the gold target."
- **Investigation Point:** Answer includes correct info but adds extra - is this wrong or just more complete?

#### Why This Tier Matters Most:

- **Potential False Negatives:** Answers marked wrong that might deserve credit
- **Grading Rule Refinement:** Edge cases reveal where evaluation criteria need clarification
- **Volume vs Quality Tradeoff:** These cases highlight the tension between strict grading and rewarding directionally correct answers

---

### 🟡 **TIER 2 - LOW Confidence CORRECT (Priority: MEDIUM)**

**Confidence Range:** 0.86-0.96
**Count:** ~20 cases
**Why Investigate:** Potential false positives - answers marked correct that might have issues.

#### Examples:

**simpleqa_0144** (Confidence: 0.86)
- **Target:** 164 acres
- **Answer:** "164-acre property... in 1988"
- **Issue:** Correct acreage but mentions wrong year (1988 vs 1998)
- **Investigation Point:** Should factual errors in supporting details fail an otherwise correct answer?

**simpleqa_0099** (Confidence: 0.90)
- **Target:** "In Relation in Space"
- **Answer:** "Relation in Space"
- **Issue:** Missing the word "In"
- **Investigation Point:** How strict should title matching be?

**simpleqa_0182** (Confidence: 0.90)
- **Target:** "Society for Artists and Performers"
- **Answer:** "Society for Artists and Performers in Hampi"
- **Issue:** Adds location not in target
- **Investigation Point:** Is extra information acceptable or does it indicate possible confusion?

#### Why This Tier Matters:

- **Quality Assurance:** Catches answers that passed but shouldn't have
- **Consistency:** Ensures grading standards are applied uniformly
- **Benchmark Integrity:** Prevents inflated accuracy scores

---

### 🟢 **TIER 3 - HIGH Confidence INCORRECT (Priority: LOWEST)**

**Confidence Range:** 0.97 (97%)
**Count:** 7 cases
**Why Lower Priority:** These are clear-cut failures with unambiguous errors.

#### Examples:

**simpleqa_0091** (Confidence: 0.97)
- **Target:** 1968
- **Answer:** "Seiko released their first 300m diver watch in 1967."
- **Issue:** Off by one year - clear factual error
- **Verdict:** Legitimate failure

**simpleqa_0035** (Confidence: 0.97)
- **Target:** Officer of the Order of the British Empire (OBE)
- **Answer:** "In 2010, the Royal Academy of Arts elected Cornelia Parker as a Royal Academician."
- **Issue:** Completely different honor from different institution
- **Verdict:** Legitimate failure

**simpleqa_0128** (Confidence: 0.97)
- **Target:** Mirza Afzal Beg
- **Answer:** "The first Deputy Chief Minister of Jammu and Kashmir was Bakshi Ghulam Mohammad."
- **Issue:** Wrong person entirely
- **Verdict:** Legitimate failure

**simpleqa_0130** (Confidence: 0.97)
- **Target:** Dušan Lajović
- **Answer:** "Based on my knowledge base, no Serbian player reached the quarterfinals..."
- **Issue:** Direct contradiction - Lajović IS Serbian
- **Verdict:** Legitimate failure

#### Why This Tier is Lower Priority:

- **Clear Failures:** No ambiguity about correctness
- **Limited Learning:** These don't reveal grading issues, just knowledge gaps
- **Not Controversial:** Judge and humans would agree these are wrong

---

## Recommended Investigation Workflow

### Phase 1: Edge Case Analysis (Tier 1)
1. **Review all 3 low-confidence INCORRECT cases**
2. **For each case, determine:**
   - Should partial/incomplete answers receive partial credit?
   - Are grading rules too strict or too lenient?
   - Do these reveal patterns in provider behavior?
3. **Document findings and refine grading criteria**

### Phase 2: Quality Assurance (Tier 2)
1. **Sample 10 low-confidence CORRECT cases**
2. **Verify they truly deserve CORRECT grade**
3. **Identify any systemic issues**
4. **Adjust grading rules if needed**

### Phase 3: Knowledge Gap Analysis (Tier 3)
1. **Review high-confidence INCORRECT cases**
2. **Categorize failure types:**
   - Factual errors
   - Knowledge base gaps
   - Retrieval failures
3. **Prioritize knowledge base improvements**

---

## Key Insights

1. **Judge confidence works as intended:** Lower confidence correctly identifies ambiguous cases

2. **Only 3 edge cases exist** out of 191 evaluations (1.6%) - excellent grading clarity

3. **90% confidence threshold** appears optimal for flagging borderline decisions

4. **The real value is in the low-confidence cases:** They reveal evaluation methodology issues, not just knowledge gaps

---

## Actionable Recommendations

### Immediate Actions:
1. ✅ **Manually review the 3 Tier 1 cases** to establish grading precedents for incomplete/partial answers
2. ✅ **Document decision rationale** for future consistency
3. ✅ **Create edge case guidelines** for partial credit scenarios

### Medium-term Actions:
4. 📊 **Sample Tier 2 cases** to validate grading consistency
5. 🔍 **Pattern analysis** across all providers to see if edge cases cluster around specific question types

### Long-term Actions:
6. 📚 **Use Tier 3 failures** to prioritize knowledge base improvements
7. 🎯 **Establish confidence thresholds** for automatic human review (e.g., <0.92 → flag for review)

---

## Conclusion

The judge confidence system is functioning correctly and provides valuable signals for continuous improvement. **Prioritize investigating low-confidence decisions first**, as these represent opportunities to refine evaluation methodology and improve benchmark quality. High-confidence failures, while important for knowledge base improvement, don't reveal grading system issues.

**Next Step:** Convene a grading standards review session to establish precedents for the 3 Tier 1 edge cases.
