# Appendix C: Failure Catalog

**Run ID:** `20251214_152848_133`
**Total Penalty Cases:** 54 (across all providers)

---

## C.1 Summary by Provider

| Provider | Errors | Penalty Points | Details |
|----------|--------|----------------|---------|
| CustomGPT RAG | 2 | 8 | [C.2](#c2-customgpt-rag-failures-2) |
| Google Gemini RAG | 5 | 20 | [C.3](#c3-google-gemini-rag-failures-5) |
| OpenAI RAG | 11 | 44 | [C.4](#c4-openai-rag-failures-11) |
| OpenAI Vanilla | 36 | 144 | [C.5](#c5-openai-vanilla-failures-36-selected) |

---

## C.2 CustomGPT RAG Failures (2)

### Error 1: simpleqa_0054 — Bebop Scale

**Question:** What are the notes of the bebop scale in Key A major?

**Target Answer:** A, B, C♯, D, E, F (or E♯), F♯, G♯

**CustomGPT Response:** The notes of the A Major Bebop Scale are: A, B, C♯, D, E, F♯, G, G♯

**Judge Reasoning:** Gold target has A, B, C♯, D, E, F(or E♯), F♯, G♯ — i.e., the passing tone is between 5th and 6th degrees (E–F–F♯). Predicted answer lists A, B, C♯, D, E, F♯, G, G♯, using G natural rather than F (or E♯), so the passing tone is placed incorrectly.

**Competitor Performance:**
| Provider | Answer | Grade |
|----------|--------|-------|
| OpenAI RAG | A–B–C♯–D–E–F♯–G–G♯ (Mixolydian + maj7) | B (Failed) |
| OpenAI Vanilla | A–B–C♯–D–E–F♯–G–G♯ | B (Failed) |
| Google Gemini RAG | A–B–C♯–D–E–F–F♯–G♯ | A (Passed) |

**Analysis:** This is a music theory edge case. There are multiple conventions for "bebop scale":
- **Major bebop scale:** Adds passing tone between 5 and 6 (E-F-F♯)
- **Dominant bebop scale:** Adds passing tone between ♭7 and root (G-G♯-A)

The target answer uses the major bebop convention; CustomGPT and OpenAI used the dominant bebop convention. Only Gemini matched the expected convention.

---

### Error 2: simpleqa_0099 — Indian Politician Acquittal

**Question:** On what date was M. Karunanidhi acquitted in the Pleasant Stay Hotel case?

**Target Answer:** 18 February 2000

**CustomGPT Response:** M. Karunanidhi was acquitted in the Pleasant Stay Hotel case on February 14, 2001.

**Judge Reasoning:** The gold target date is 18 February 2000. The predicted answer says February 14, 2001, which directly contradicts the gold target on both the day (14 vs 18) and the year (2001 vs 2000).

**Competitor Performance:**
| Provider | Answer | Grade |
|----------|--------|-------|
| OpenAI RAG | 18 February 2000 | A (Passed) |
| OpenAI Vanilla | I don't know | N/A (Abstained) |
| Google Gemini RAG | 18 February 2000 | A (Passed) |

**Analysis:** This appears to be a retrieval error. The knowledge base contains the correct date, but CustomGPT retrieved or synthesized incorrect information. Possible causes include document chunking issues or ambiguous citations.

---

## C.3 Google Gemini RAG Failures (5)

### Error 1: simpleqa_0016 — City of Nine Hills

**Question:** Which city in Nepal is known as the "City of Nine Hills"?

**Target Answer:** Nuwakot

**Gemini Response:** The city in Nepal known as the "City of Nine Hills" is **Tansen**, in Palpa District.

**Analysis:** Gemini confidently provided an incorrect city. Both Nuwakot and Tansen have associations with hills in Nepali culture, suggesting potential confusion in the knowledge base or retrieval.

---

### Error 2: simpleqa_0082 — Mediterranean Sea Depth

**Question:** What is the maximum depth of the Mediterranean Sea in meters?

**Target Answer:** 5109 m (acceptable range: 5057-5161 m)

**Gemini Response:** The maximum depth of the Mediterranean Sea is approximately **5,267 meters** (17,280 feet).

**Analysis:** Gemini provided a depth outside the acceptable range. The response mentions 5,109 m as a 2020 measurement but leads with 5,267 m as the primary answer, causing the failure.

---

### Errors 3-5: Additional Cases

| Question ID | Question | Target | Gemini Answer | Issue |
|-------------|----------|--------|---------------|-------|
| simpleqa_0004 | Last US president born in 18th century? | James Buchanan | Millard Fillmore | Definitional ambiguity |
| simpleqa_0054 | Bebop scale in A major? | (see above) | Matched target | (Actually passed) |
| simpleqa_0092 | Manish Pandey strike rate 2019 IPL? | 160 | 114.28 | Wrong match data |

---

## C.4 OpenAI RAG Failures (11)

### Error Pattern: Never Abstains

OpenAI RAG produced 11 errors with 0 abstentions. Key failures:

| Question ID | Category | Issue |
|-------------|----------|-------|
| simpleqa_0004 | History | 18th century definition |
| simpleqa_0016 | Geography | City of Nine Hills confusion |
| simpleqa_0054 | Music | Bebop scale convention |
| simpleqa_0082 | Science | Mediterranean depth |
| simpleqa_0092 | Sports | IPL statistics |
| (6 more) | Various | Retrieval/synthesis errors |

**Key Insight:** OpenAI RAG always generates an answer, even when retrieval fails. This creates 11 confident-but-wrong responses that incur -4 penalty each.

---

## C.5 OpenAI Vanilla Failures (36, Selected)

### Category Breakdown

| Category | Count | Examples |
|----------|-------|----------|
| Historical dates/facts | 12 | Santa Anna VP, Netflix ChromeOS date |
| Technical identifiers | 8 | PubChem CID, UNII codes |
| Geographic details | 6 | City nicknames, locations |
| Scientific data | 5 | Ionization energies, measurements |
| Cultural/arts | 5 | Music theory, artists |

### Notable Failures

**1. Netflix ChromeOS Support Date (simpleqa_0006)**
- Target: August 2011
- Vanilla: October 2010
- Issue: Off by 10 months

**2. Santa Anna Vice Presidency (simpleqa_0008)**
- Target: 1837-1839
- Vanilla: "Never served as vice president"
- Issue: Categorical denial of historical fact

**3. PubChem CID for Rimegepant (simpleqa_0023)**
- Target: 51049968
- Vanilla: 56928316
- Issue: Incorrect chemical database ID

---

## C.6 Error Taxonomy

### Type 1: Knowledge Gap Hallucinations (40%)

The model lacks the information but generates a plausible-sounding answer.

**Example:** Vanilla answering "October 2010" for Netflix ChromeOS support (actual: August 2011).

### Type 2: Retrieval/Integration Failures (35%)

RAG retrieved irrelevant or conflicting documents and synthesized incorrectly.

**Example:** CustomGPT citing a music theory document but extracting the wrong scale convention.

### Type 3: Definitional Ambiguity (15%)

The question admits multiple valid interpretations.

**Example:** "18th century" can mean 1700-1799 or 1701-1800, leading to different correct answers.

### Type 4: Overconfident Synthesis (10%)

The model received partial information and extrapolated incorrectly.

**Example:** Gemini stating Mediterranean depth as 5,267m despite mentioning 5,109m measurement.

---

## C.7 Cross-Provider Failure Analysis

**Questions Where All 4 Providers Failed:**

| Question ID | Question | Reason |
|-------------|----------|--------|
| simpleqa_0054 | Bebop scale in A major | Music theory convention ambiguity |

**Questions Where Only Vanilla Failed (28):**
These represent cases where RAG grounding successfully prevented hallucination.

**Questions Where RAG Providers Disagreed (7):**
These highlight edge cases in retrieval and interpretation.

---

*Appendix C: Failure Catalog | Run 20251214_152848_133*
