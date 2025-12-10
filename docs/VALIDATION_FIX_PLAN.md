# Answer Validation Fix Plan (v2 - Critic-Reviewed)

## Problem Summary

The answer validation script reports **85.8% coverage** but manual verification shows this is **significantly underestimated**. The true coverage is estimated at **93-96%**.

## Root Cause Analysis (Verified)

### Bug #1: Content Truncation (CRITICAL)
**File:** `src/answer_validator.py:261`
```python
content_truncated = content[:4000] if len(content) > 4000 else content
```

**Evidence (byte positions verified):**
- Question 74 ("168 acres"): Answer at byte **10,259** in 337KB file
- Question 32 ("Oct 23, 2018"): Answer at byte **10,572** in 118KB file
- LLM only sees bytes 0-4000, so **answers are NEVER seen**

### Bug #2: Answer Metadata Pollution (HIGH)
**File:** `scripts/validate_answers_in_kb.py:127`

Answers contain metadata that breaks matching:
```
"168 acres (acceptable range: anything between 166 and 170 acres)"
```

### Bug #3: Incomplete Date Variants (MEDIUM)
Date variants only go `January → Jan`, not `Oct → October`.

### Bug #4: No Article Normalization (LOW)
"The right arm" fails to match "Her right arm".

## Fixes (Ordered by Effort/Impact - Critic Approved)

### Fix 1: Increase Truncation Limit (ONE LINE)
```python
# src/answer_validator.py:261
# BEFORE:
content_truncated = content[:4000] if len(content) > 4000 else content
# AFTER:
content_truncated = content[:32000] if len(content) > 32000 else content
```
**Impact:** Fixes ~80 false negatives. gpt-4o-mini has 128K context, no reason for 4K limit.

### Fix 2: Strip Answer Metadata (5 LINES)
```python
# scripts/validate_answers_in_kb.py:127
import re
answer = row['answer']
# Strip metadata like "(acceptable range: ...)"
answer = re.sub(r'\s*\(acceptable range:[^)]+\)', '', answer).strip()
answer = re.sub(r'\s*\(acceptable.*?\)', '', answer, flags=re.IGNORECASE).strip()
```
**Impact:** Fixes ~50 false negatives

### Fix 3: Bidirectional Date Mappings (10 LINES)
```python
# src/answer_validator.py - add after line 177
abbrev_to_full = {
    'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
    'Apr': 'April', 'Jun': 'June', 'Jul': 'July',
    'Aug': 'August', 'Sep': 'September', 'Oct': 'October',
    'Nov': 'November', 'Dec': 'December',
}
for abbrev, full in abbrev_to_full.items():
    if re.search(rf'\b{abbrev}\b', answer, re.IGNORECASE):
        variants.append(re.sub(rf'\b{abbrev}\b', full, answer, flags=re.IGNORECASE))
```
**Impact:** Fixes ~30 false negatives

### Fix 4: Article/Pronoun Normalization (5 LINES)
```python
# src/answer_validator.py - add to generate_answer_variants()
# Strip leading articles/pronouns
article_stripped = re.sub(r'^(the|a|an|his|her|their|its)\s+', '', answer, flags=re.IGNORECASE)
if article_stripped != answer:
    variants.append(article_stripped)
```
**Impact:** Fixes ~10 false negatives

## NOT Recommended (Critic Feedback)

1. ❌ **Chunking approach** - Overengineered. Simple truncation increase is sufficient.
2. ❌ **Upgrade to gpt-4.1** - 13x cost increase. Only if above fixes don't reach target.
3. ❌ **Embedding-based retrieval** - Unnecessary complexity.

## Cost Analysis

| Approach | Cost Impact |
|----------|-------------|
| Increase truncation to 32K | ~1.5x (more input tokens) |
| Upgrade to gpt-4.1 | ~13x per call |
| Chunking (3 chunks) | ~3x more calls |

**Recommendation:** Start with truncation increase only. Model upgrade is last resort.

## Acceptance Criteria

- [ ] Coverage rate increases from 85.8% to **>93%**
- [ ] All 3 manually verified cases (Q74, Q32, Q285) now show as "found"
- [ ] No false positives introduced (spot-check 10 "found" results)

## Testing Strategy (Improved)

1. **Unit tests** for `generate_answer_variants()`:
   - Test date variants both directions
   - Test article stripping
   - Test metadata stripping

2. **Integration test** with known failure cases:
   ```python
   known_failures = [74, 32, 285, 180, 516, 788]  # Questions that should pass after fix
   ```

3. **Before/After comparison**:
   - Run validation before fixes, save report
   - Apply fixes
   - Run validation after fixes, compare

4. **Spot-check remaining failures** to confirm they're true negatives

## Files to Modify

| File | Changes |
|------|---------|
| `src/answer_validator.py:261` | Increase truncation limit |
| `src/answer_validator.py:166-177` | Add reverse date mappings |
| `src/answer_validator.py:193+` | Add article normalization |
| `scripts/validate_answers_in_kb.py:127` | Strip answer metadata |
