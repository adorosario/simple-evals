# GoalSpec: SimpleQA-Verified KB Coverage Audit

## Objective (North Star)

Verify the theoretical performance ceiling for RAG systems by auditing whether gold answers actually exist in the corresponding knowledge base files. This establishes the maximum achievable accuracy before retrieval quality becomes the bottleneck.

> A RAG system cannot answer correctly if the answer isn't in the KB. The audit identifies the hard ceiling.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total questions | 1,000 |
| Answers verified in KB | 974 (97.4%) |
| Answers not found | 26 (2.6%) |
| **Theoretical RAG ceiling** | **97.4%** |
| Audit date | 2025-12-08 |
| Status | VERIFIED_FOR_PRODUCTION |

---

## Audit Methodology

### 1:1 File Mapping

Each question maps to exactly ONE knowledge base file via `build_manifest.json`:

```
CSV original_index → KB file
5  → verified_0001.txt
8  → verified_0002.txt
9  → verified_0003.txt
...
```

This ensures the audit checks the **corresponding file**, not just any file in the KB.

### Hybrid Validation (3 Phases)

**Phase 1: String Matching with Answer Variants** (Fast)

Generates normalized answer variants for flexible matching:

| Transformation | Example |
|----------------|---------|
| Currency | "120,000 euros" → "€120,000", "120000 EUR" |
| Numbers | "120,000" → "120000" |
| Dates (bidirectional) | "Jan 15" ↔ "January 15", "2020-01-15" |
| Names | "John Smith" → "John", "Smith", "Smith, John" |
| Articles | "The right arm" → "right arm" |
| Ordinals | "1st" → "first" |

**Result**: 883/1000 found (88.3%)

**Phase 2: LLM Fallback** (GPT-4.1)

For string match failures, uses semantic verification:
- Prompt: "Does content contain answer (exactly or semantically equivalent)?"
- Content: Up to 32KB of KB file
- Model: gpt-4.1

**Result**: 83 YES + 8 PARTIAL = 91 more found (9.1%)

**Phase 3: Not Found**

**26 questions** (2.6%) where neither method found the answer.

---

## The 26 Not-Found Cases

| Category | Count | Examples |
|----------|-------|----------|
| Dataset quality issues | 3 | Q55 asks for latitude but answer is longitude (33.7738) |
| Dynamic content | 5 | JavaScript-rendered census tables, interactive data |
| Inaccessible sources | 15 | Paywalls, 403 errors, removed URLs |
| Validation false negatives | 3 | Q29: "tramp steamer" IS present but validator missed |

### Implications

- **Dataset bugs (3)**: RAG systems SHOULD get these wrong - the "correct" answer is actually incorrect
- **Dynamic content (5)**: Source data exists but couldn't be extracted (JS-rendered)
- **Inaccessible sources (15)**: Content never made it into KB - legitimate abstention cases
- **False negatives (3)**: Answer IS present - theoretical ceiling is actually ~97.7%

---

## Acceptance Function (AF) — "Done?"

AF is true only when **ALL** criteria are satisfied:

### Validation Engine
- [x] **#1** String matching engine with answer variants implemented
- [x] **#2** LLM fallback verification using GPT-4.1
- [x] **#3** 1:1 KB file mapping validated via build_manifest.json
- [x] **#4** Content truncation fixed (4KB → 32KB)
- [x] **#5** Bidirectional date normalization implemented

### Documentation
- [x] **#6** Not-found cases documented with categorized reasons
- [x] **#7** Coverage statistics computed and stored
- [x] **#8** Production certification issued

### Current Status

**Last updated**: 2025-12-08
**Overall**: **MET** (8/8 criteria satisfied)

**Verified AF Criteria**:
- [x] **AF#1-5 Validation Engine**: Hybrid validator operational
- [x] **AF#6-8 Documentation**: Complete audit trail in `knowledge_base_verified/audit/`

---

## Bug Fixes Applied

| Bug | Severity | Impact | Fix |
|-----|----------|--------|-----|
| 4KB content truncation | CRITICAL | ~80 false negatives | Increased to 32KB |
| Answer metadata not stripped | HIGH | ~50 false negatives | Regex to strip "(acceptable range: ...)" |
| One-way date conversion | MEDIUM | ~30 false negatives | Bidirectional date mapping |
| Missing article normalization | MEDIUM | ~10 false negatives | Strip The/A/An/His/Her |
| Weak LLM for semantic matching | LOW | Semantic misses | Upgraded gpt-4o-mini → gpt-4.1 |

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Total KB documents | 1,000 |
| Total words | 27,615,554 |
| Average words/doc | 27,616 |
| Min words | 192 |
| Max words | 96,200 |
| URL validity rate | 68.97% |
| Spot-check verification | 100% (50 samples) |

---

## Audit Artifacts

| File | Purpose |
|------|---------|
| `knowledge_base_verified/audit/final_audit_summary.json` | Production certification |
| `knowledge_base_verified/audit/answer_validation_report.json` | Per-question results |
| `knowledge_base_verified/build_manifest.json` | Question → KB file mapping |
| `scripts/validate_answers_in_kb.py` | Audit script (263 lines) |
| `src/answer_validator.py` | Hybrid validation engine (476 lines) |

---

## Constraints

### Data Integrity
- Questions must be validated against their **corresponding** KB file (1:1 mapping)
- String matching must use normalized variants to handle format differences
- LLM fallback limited to 32KB content per validation

### Performance Implications
- **97.4% ceiling** means perfect RAG retrieval can achieve at most 97.4% accuracy
- The 26 not-found questions SHOULD result in abstentions ("I don't know")
- Abstention on these questions is correct behavior, not failure

---

## Certification

**Status**: APPROVED FOR PRODUCTION

**Criteria Met**:
- Coverage >= 95% (achieved 97.4%)
- Zero empty files
- Spot-check verification 100%
- Known limitations documented

**Approved for**: RAG system evaluation and benchmarking

---

## Notes

- 2025-12-08: Production certification issued by critic_agent
- Q29 false negative identified for future validator improvement
- Dataset bug (Q55) should be reported to SimpleQA maintainers
- Dynamic content sources may need periodic refresh
