---
description: Shortcut paranoia check — identify any divergence from best-known approach and propose a parity plan.
---

# Goal Optimizer: /no-shortcuts

Scan the current approach and identify any shortcuts, workarounds, or deviations from best practice. This is the "corner-cutting detector" that ensures we're not trading quality for speed.

## When to Use

- When you see something that "looks like a workaround"
- After implementing a fix that might be "good enough"
- When Alden says "I get concerned when you take shortcuts"
- Before declaring a feature complete
- During code review

## What Counts as a Shortcut?

### 1. Workarounds Masquerading as Solutions
- Temporary fixes that became permanent
- "It works but..." implementations
- Hacks that avoid the real problem

### 2. Missing Manus Parity
- Feature Manus has that we don't
- Different approach where Manus has better solution
- Missing tool from the 27-tool set

### 3. Technical Debt Without Tracking
- TODOs without issue references
- Disabled tests without justification
- Known limitations not documented

### 4. "Good Enough" Decisions
- Hardcoded values that should be configurable
- Missing edge case handling
- Skipped error scenarios
- Missing input validation

### 5. Performance/Security Compromises
- Synchronous where async is better
- Missing rate limiting
- Unvalidated inputs
- Exposed internal details

## Analysis Process

### Step 1: Identify Current Approach
For each component/feature under review:
- What approach did we take?
- What alternatives exist?
- What does Manus do?

### Step 2: Compare to Best Practice
- Is this the "right" way to do it?
- What would a senior engineer at a top company do?
- Are we following industry standards?

### Step 3: Assess Impact
- What's the risk of this shortcut?
- Will it cause problems later?
- Does it affect users?

### Step 4: Justify or Plan Fix
- If acceptable: Document why
- If not acceptable: Create plan to fix

## Output Format

```markdown
## No-Shortcuts Analysis

### Summary
- **Components reviewed**: [List]
- **Shortcuts found**: [Count]
- **Critical shortcuts**: [Count]
- **Justified shortcuts**: [Count]
- **Action required**: YES / NO

### Shortcut Inventory

#### Shortcut 1: [Name]
- **Location**: `[file:line]` or [Component]
- **What we did**: [Description of current approach]
- **Best practice**: [What we should do]
- **Manus approach**: [How Manus handles this, if known]
- **Impact**: Low / Medium / High / Critical
- **Risk**: [What could go wrong]
- **Status**: JUSTIFIED / NEEDS FIX

**Justification** (if JUSTIFIED):
[Why this shortcut is acceptable given our constraints]

**Fix plan** (if NEEDS FIX):
1. [Step to fix]
2. [Step to fix]
3. [Verification]

---

#### Shortcut 2: [Name]
[Same format]

---

### Manus Parity Check

| Manus Feature | Our Status | Gap | Priority |
|---------------|------------|-----|----------|
| [Feature 1] | Implemented / Partial / Missing | [What's missing] | P0/P1/P2 |
| [Feature 2] | Implemented / Partial / Missing | [What's missing] | P0/P1/P2 |

### Technical Debt Register

| Item | Location | Type | Tracked? | Issue # |
|------|----------|------|----------|---------|
| [Debt 1] | `[file]` | TODO/FIXME/Skip | Yes/No | [#123] |
| [Debt 2] | `[file]` | TODO/FIXME/Skip | Yes/No | [#124] |

### Parity Plan

If shortcuts need fixing, prioritized plan:

#### Immediate (P0 - Blocking)
1. [Fix 1] — [Why urgent]
2. [Fix 2] — [Why urgent]

#### Short-term (P1 - This sprint)
1. [Fix 1] — [Why soon]
2. [Fix 2] — [Why soon]

#### Backlog (P2 - Track but defer)
1. [Fix 1] — [Why can wait]
2. [Fix 2] — [Why can wait]

### Verdict

**Are we cutting corners?**

[ ] NO — No significant shortcuts, or all shortcuts justified
[ ] YES — Action required before claiming done

**Required actions**:
1. [Action 1]
2. [Action 2]
```

## Common Shortcuts to Watch For

### Code Level
- `# TODO: fix this later` without issue
- `except: pass` (swallowed exceptions)
- `@pytest.mark.skip` without reason
- Magic numbers without constants
- Copy-pasted code blocks
- Disabled linting rules

### Architecture Level
- Single point of failure
- No retry logic
- Missing circuit breakers
- Tight coupling
- No graceful degradation

### Security Level
- Hardcoded credentials (even in tests)
- Missing input validation
- CORS set to `*`
- No rate limiting
- Logging sensitive data

### Testing Level
- No tests for new code
- Tests that don't actually test
- Missing edge cases
- No error path testing
- Flaky tests ignored

## Rules

1. **Be paranoid** — Assume shortcuts exist until proven otherwise
2. **Compare to best** — Not "does it work" but "is it right"
3. **Manus is the bar** — If Manus does it better, we have a gap
4. **Track everything** — No secret shortcuts, all debt documented
5. **Plan to fix** — Every shortcut needs a path to resolution
