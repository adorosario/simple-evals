---
description: Enforce quality gates - reviewer+no-shortcuts+tests+docs+real repro steps. Refuse "done" unless AF is satisfied.
---

# Goal Optimizer: /quality-gate

Enforce all quality gates before accepting work as "done". This is the final checkpoint before claiming completion.

## When to Use

- Before marking a phase as complete
- Before creating a PR
- Before claiming AF criteria are satisfied
- When Alden asks "is this actually done?"

## Quality Gates (All Must Pass)

### Gate 1: Smoke Tests
Run basic checks to catch obvious failures:

```bash
# Lint check
docker compose run --rm dev ruff check .

# Quick tests
docker compose run --rm dev pytest -q --maxfail=3
```

**Requirement**: Both must pass with no errors.

### Gate 2: Reviewer Pass
Invoke the **reviewer** agent to check:

- [ ] Security: No vulnerabilities introduced
- [ ] Performance: No obvious inefficiencies
- [ ] Maintainability: Code is clear and follows patterns
- [ ] Correctness: Logic matches requirements

**Requirement**: No blockers, all major issues addressed.

### Gate 3: "No Shortcuts" Check
Invoke the **critic** agent with shortcut detection:

- [ ] No workarounds masquerading as solutions
- [ ] No missing Manus parity features without justification
- [ ] No "good enough" decisions that should be revisited
- [ ] No technical debt without tracking

**Requirement**: All shortcuts documented and justified.

### Gate 4: QA Scenarios
Invoke the **qa-bach** agent to verify:

- [ ] Real user flows tested (not just unit tests)
- [ ] Edge cases covered
- [ ] Error handling verified
- [ ] E2E scenarios pass

**Requirement**: All critical scenarios pass.

### Gate 5: Documentation Pass
Invoke the **procida** agent to verify:

- [ ] Setup documentation current
- [ ] Usage documentation current
- [ ] Troubleshooting guide current
- [ ] API reference complete (if applicable)

**Requirement**: Docs match implementation.

### Gate 6: Reality Check
Provide exact reproduction steps that anyone can follow:

```markdown
## Reality Check: Manual Verification

### Prerequisites
- [What must be set up]

### Steps to Reproduce
1. `[Command 1]` → Expected: [Result]
2. `[Command 2]` → Expected: [Result]
3. `[Command 3]` → Expected: [Result]

### Expected Final State
- [What should be true when done]
```

**Requirement**: Steps work when followed exactly.

## Output Format

```markdown
## Quality Gate Results

### Summary
| Gate | Status | Notes |
|------|--------|-------|
| Smoke Tests | PASS/FAIL | [Brief note] |
| Reviewer | PASS/FAIL | [Brief note] |
| No Shortcuts | PASS/FAIL | [Brief note] |
| QA Scenarios | PASS/FAIL | [Brief note] |
| Documentation | PASS/FAIL | [Brief note] |
| Reality Check | PASS/FAIL | [Brief note] |

### Overall Verdict
**AF SATISFIED / NOT SATISFIED**

### Gate 1: Smoke Tests
```bash
$ docker compose run --rm dev ruff check .
[output]

$ docker compose run --rm dev pytest -q --maxfail=3
[output]
```
**Result**: PASS / FAIL

### Gate 2: Reviewer Report
[Summary from reviewer agent]

**Blockers**: [List or "None"]
**Major issues**: [List or "None"]

### Gate 3: No Shortcuts Report
[Summary from critic agent]

| Shortcut Found | Justification | Acceptable? |
|----------------|---------------|-------------|
| [Item] | [Reason] | Yes/No |

### Gate 4: QA Report
[Summary from qa-bach agent]

| Scenario | Result | Evidence |
|----------|--------|----------|
| [Scenario 1] | Pass/Fail | [Link/command] |
| [Scenario 2] | Pass/Fail | [Link/command] |

### Gate 5: Documentation Report
[Summary from procida agent]

| Doc Type | Status | Action |
|----------|--------|--------|
| Setup | Current/Stale | [None/Update needed] |
| Usage | Current/Stale | [None/Update needed] |
| Troubleshooting | Current/Stale | [None/Update needed] |
| API Reference | Current/Stale | [None/Update needed] |

### Gate 6: Reality Check

#### Prerequisites
[List]

#### Reproduction Steps
1. `[command]` → [expected]
2. `[command]` → [expected]
3. `[command]` → [expected]

#### Verified By
[Your verification output showing it works]

### AF Status Update

| AF Criterion | Before | After | Evidence |
|--------------|--------|-------|----------|
| #1 | [ ] | [x] | [How verified] |
| #2 | [ ] | [ ] | [Still pending because...] |
| ... | ... | ... | ... |

**Criteria satisfied**: X/14
**Criteria pending**: [List numbers and what's missing]

### Final Verdict

**CAN WE CLAIM DONE?**

[ ] YES — All gates pass, AF criteria satisfied
[ ] NO — Missing: [List what's not done]

### Required Actions (if NO)
1. [Specific action needed]
2. [Specific action needed]
3. [Re-run /quality-gate after fixes]
```

## Rules

1. **All gates must pass** — No exceptions, no "we'll fix it later"
2. **Evidence required** — Show command outputs, not just claims
3. **Be honest** — If something fails, report it clearly
4. **Reproducible** — Reality check must work for anyone
5. **Update AF** — Always show AF status change
