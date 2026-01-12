---
description: Run one Goal Optimizer iteration (explore → plan → score → implement → verify → update TODO).
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(git log:*), Read, Grep, Glob, Edit, Write, Bash
---

# Goal Optimizer: /next

Run one complete iteration of the Goal Optimizer loop.

## Context (Auto-Injected)

### Project State
- **GoalSpec**: @docs/GOALS.md
- **Current TODO**: @todo.md
- **Project rules**: @CLAUDE.md

### Git State
- **Branch**: !`git branch --show-current`
- **Status**: !`git status -sb`
- **Diff stats**: !`git diff --stat`
- **Recent commits**: !`git log --oneline -5`

## Task

Execute the Goal Optimizer loop:

### 1) Explorer Phase
Use the **explorer** agent to produce a minimal context pack:
- Identify relevant files for current work
- Surface test/lint/format commands
- Note any blockers or unknowns

### 2) Planner Phase
Use the **planner** agent to propose 2-4 candidate next actions:
- Each action must have clear description and expected outcome
- Include validation command for each
- Prioritize time-to-signal (steps that quickly validate assumptions)

### 3) Critic Phase
Use the **critic** agent to score candidates with PUF:
- Apply Process Utility Function rubric (Evidence, User Value, Risk, Reversibility, Cost)
- Select the highest-scoring action
- Identify any shortcuts or risks

### 4) Executor Phase
Use the **executor** agent to implement the smallest viable slice:
- Make minimal changes
- Follow existing patterns
- Add tests if appropriate

### 5) Verifier Phase
Use the **verifier** agent to verify reality:
- Run tests/lint/build commands
- Provide exact repro steps
- Update AF status based on evidence

### 6) State Update
Update project state:
- Mark completed items in @todo.md
- Add new items discovered
- Update @docs/GOALS.md AF checklist if criteria satisfied

## Output Requirements

Your output MUST include:

```markdown
## AF Status
- **Overall**: MET / NOT MET (X/14 criteria satisfied)
- **Missing**: [List specific AF items not yet satisfied]

## PUF Scoring Table

| # | Candidate Action | E | U | R | Rev | C | Total |
|---|-----------------|---|---|---|-----|---|-------|
| 1 | [Description] | [0-5] | [0-5] | [0-5] | [0-5] | [0-5] | [/55] |
| 2 | [Description] | [0-5] | [0-5] | [0-5] | [0-5] | [0-5] | [/55] |
| 3 | [Description] | [0-5] | [0-5] | [0-5] | [0-5] | [0-5] | [/55] |

**Selected**: Candidate [N]
**Why**: [Justification]

## What Changed
[Summary of implementation]

## Verification Results
```bash
$ [exact command]
[output]
```
**Result**: PASS / FAIL

## What to Do Next
[Next recommended action based on updated state]

## TODO Update
[Show the new todo.md state]
```

## Rules

1. **Never skip phases** — even if something seems obvious
2. **Evidence required** — don't claim "should work", prove it
3. **Small steps** — implement smallest viable slice
4. **Update state** — always update todo.md and AF status
5. **Be honest** — if something fails, report it clearly
