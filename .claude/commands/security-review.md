---
name: security-review
description: Comprehensive security assessment using STRIDE threat modeling, OWASP checks, and dependency scanning.
---

# Security Review Command

Run a comprehensive security assessment on recent changes using the `security-schneier` agent.

## Context Injection

Automatically gather:
- @docs/GOALS.md (Security AF criteria: #9, #10)
- Recent git diff: `git diff HEAD~5 --name-only`
- Changed files content
- Dependency manifests (requirements.txt, package.json)

## Workflow

### 1) Identify Scope
```bash
# What changed?
git diff HEAD~5 --name-only
git diff HEAD~5 --stat

# What's the current security posture?
# (Read existing security docs if any)
```

### 2) STRIDE Threat Model
For each changed component:
- **S**poofing risks
- **T**ampering risks
- **R**epudiation risks
- **I**nformation disclosure risks
- **D**enial of service risks
- **E**levation of privilege risks

### 3) OWASP Top 10 Check
Verify recent changes don't introduce:
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A04: Insecure Design
- A05: Security Misconfiguration
- A06: Vulnerable Components
- A07: Auth Failures
- A08: Software/Data Integrity
- A09: Logging Failures
- A10: SSRF

### 4) Dependency Scan
```bash
# Python dependencies
docker compose run --rm dev pip-audit 2>/dev/null || echo "pip-audit not installed"
docker compose run --rm dev safety check 2>/dev/null || echo "safety not installed"

# Check for outdated packages
docker compose run --rm dev pip list --outdated
```

### 5) Secret Detection
```bash
# Search for potential hardcoded secrets
grep -rn "password\|secret\|api_key\|token\|credential" --include="*.py" --include="*.js" --include="*.json" . 2>/dev/null | head -20

# Check git history for .env files
git log --all --full-history -- "*.env" 2>/dev/null | head -10
```

### 6) Generate Report
Output using security-schneier format:
- Attack surface analysis
- STRIDE findings with DREAD scores
- OWASP checklist status
- Compliance considerations
- Verdict: APPROVED / CONDITIONAL / REJECTED

## AF Alignment

Map findings to @docs/GOALS.md:

| Finding | Severity | AF Criteria | Impact |
|---------|----------|-------------|--------|
| [Finding] | P0/P1/P2 | #9, #10 | [Impact on AF status] |

## Output Format

```markdown
## Security Review Results

### Scope
- **Files reviewed**: [count]
- **Commits covered**: [range]
- **Date**: [timestamp]

### STRIDE Summary
| Category | Findings | Highest Severity |
|----------|----------|------------------|
| Spoofing | [count] | P0/P1/P2/None |
| Tampering | [count] | P0/P1/P2/None |
| Repudiation | [count] | P0/P1/P2/None |
| Information Disclosure | [count] | P0/P1/P2/None |
| Denial of Service | [count] | P0/P1/P2/None |
| Elevation of Privilege | [count] | P0/P1/P2/None |

### Critical Findings (P0)
[List with DREAD scores and remediation]

### High Findings (P1)
[List with DREAD scores and remediation]

### Dependency Vulnerabilities
[CVE list if any]

### OWASP Status
[Checklist results]

### Verdict
**Status**: APPROVED / CONDITIONAL / REJECTED
**AF #9 Status**: SATISFIED / NOT SATISFIED
**AF #10 Status**: SATISFIED / NOT SATISFIED

### Required Actions
1. [Action with owner and deadline]
```

## Trigger Phrases

- "run security review"
- "security check"
- "threat model this"
- "/security-review"

## Agent Delegation

Primary: `security-schneier`
Supporting: `reviewer` (for code-level security checks)
