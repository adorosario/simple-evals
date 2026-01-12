---
description: Deep research + tradeoffs + concrete plan before touching code.
---

# Goal Optimizer: /ultrathink

Activate deep analysis mode. Do NOT edit code until the plan is agreed or obviously correct.

## When to Use

- Starting a new feature or significant change
- Tackling a complex problem with multiple approaches
- When the "best" solution isn't obvious
- Before making architectural decisions
- When Alden says "ultrathink about this"

## Process

### Phase 1: Deep Research

Conduct thorough research before proposing solutions:

1. **Use MCP WebSearch** (call it as many times as needed)
   - How do industry leaders solve this problem?
   - What does Manus do? (if relevant)
   - What are current best practices?
   - Are there existing libraries/solutions?

2. **Explore the codebase**
   - How do we currently handle similar problems?
   - What patterns exist?
   - What constraints apply?

3. **Document findings**
   - Key insights from research
   - Relevant prior art
   - Applicable patterns

### Phase 2: Multi-Perspective Analysis

Consider at least 3 approaches with tradeoffs:

```markdown
## Approach Analysis

### Approach A: [Name]
**Description**: [How it works]
**Pros**:
- [Advantage 1]
- [Advantage 2]
**Cons**:
- [Disadvantage 1]
- [Disadvantage 2]
**Risk level**: Low / Medium / High
**Effort estimate**: Small / Medium / Large
**Manus parity**: [How does Manus handle this?]

### Approach B: [Name]
[Same format]

### Approach C: [Name]
[Same format]
```

### Phase 3: Recommendation

Pick the best approach and justify:

```markdown
## Recommendation

**Selected approach**: [A/B/C]

**Rationale**:
- [Why this approach is best for our context]
- [How it aligns with constraints]
- [How it serves the GoalSpec objective]

**Tradeoffs accepted**:
- [What we're giving up and why it's acceptable]

**Risks to monitor**:
- [What could go wrong]
- [How we'll detect problems early]
```

### Phase 4: Concrete Plan

Provide step-by-step implementation plan:

```markdown
## Implementation Plan

### Prerequisites
- [ ] [What must be true before we start]

### Steps

#### Step 1: [Name]
- **Action**: [What to do]
- **Expected result**: [What success looks like]
- **Verification**: `[Command to verify]`

#### Step 2: [Name]
[Same format]

#### Step 3: [Name]
[Same format]

### Checkpoints
- After Step 1: [What to verify]
- After Step 2: [What to verify]
- After Step 3: [What to verify]

### Rollback Plan
If things go wrong:
1. [How to undo Step 1]
2. [How to undo Step 2]
3. [How to undo Step 3]
```

## Output Requirements

Your output MUST include:

1. **Research Summary** — Key findings from web search and codebase exploration
2. **3+ Approaches** — Each with pros, cons, risk, effort, and Manus comparison
3. **Recommendation** — Selected approach with rationale
4. **Implementation Plan** — Step-by-step with verification commands
5. **Approval Request** — Ask before proceeding with implementation

## Rules

1. **Research first** — Don't propose solutions without understanding the problem space
2. **Multiple perspectives** — Never present only one approach
3. **Tradeoffs explicit** — Every decision has costs; name them
4. **No code yet** — This is planning mode, not implementation
5. **Seek approval** — Wait for confirmation before implementing

## Example Trigger Phrases

- "ultrathink about this"
- "I need you to deeply research..."
- "What are our options for..."
- "How should we approach..."
- "Before we implement, let's think through..."
