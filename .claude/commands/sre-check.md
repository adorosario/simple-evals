---
name: sre-check
description: Production readiness assessment using SLO verification, runbook check, and rollback procedure validation.
---

# SRE Check Command

Run a production readiness assessment using the `sre-allspaw` agent.

## Context Injection

Automatically gather:
- @docs/GOALS.md (Performance constraints, security constraints)
- Current deployment state: `docker compose ps`
- Recent logs: `docker compose logs --tail=50`
- Health check status

## Workflow

### 1) SLO Verification
Check that SLOs are defined and measurable:

```markdown
## SLO Status

| SLI | Target | Current | Measurement Method |
|-----|--------|---------|-------------------|
| Availability | [X%] | [Current] | [How measured] |
| P95 Latency | [Xms] | [Current] | [How measured] |
| Error Rate | [X%] | [Current] | [How measured] |

**Error Budget Status**: [X% remaining]
```

### 2) Observability Check
```bash
# Check if metrics endpoint exists
docker compose exec app curl -s localhost:8080/metrics 2>/dev/null | head -10 || echo "No metrics endpoint"

# Check if health endpoint exists
docker compose exec app curl -s localhost:8080/health 2>/dev/null || echo "No health endpoint"

# Check logging configuration
docker compose logs --tail=10
```

### 3) Runbook Verification
- [ ] Runbook exists for this service
- [ ] Runbook covers common failure scenarios
- [ ] Runbook was updated within last 30 days
- [ ] On-call knows where runbook is

### 4) Rollback Procedure
Verify rollback is documented and tested:

```markdown
## Rollback Readiness

- **Rollback trigger criteria**: [When to rollback]
- **Rollback steps documented**: Yes/No
- **Estimated rollback time**: [X minutes]
- **Last rollback test**: [Date or "Never"]
```

### 5) Alerting Check
- [ ] SLO-based alerts configured
- [ ] Alert routing to on-call verified
- [ ] Runbook linked from alert
- [ ] Alert fatigue assessed (not too noisy)

### 6) Generate Report
Output using sre-allspaw format with production readiness verdict.

## Production Readiness Checklist

### Observability
- [ ] Metrics instrumented (key SLIs)
- [ ] Dashboards created and accessible
- [ ] Alerting configured and tested
- [ ] Logging structured with request IDs
- [ ] Tracing enabled (if distributed)

### Reliability
- [ ] SLOs defined with error budgets
- [ ] Graceful degradation implemented
- [ ] Rate limiting configured
- [ ] Circuit breakers in place
- [ ] Health checks (liveness + readiness)

### Operability
- [ ] Runbook exists and current
- [ ] Rollback procedure documented
- [ ] Rollback tested successfully
- [ ] Backup/restore procedure (if stateful)
- [ ] On-call rotation defined

### Security (cross-check with /security-review)
- [ ] Secrets management verified
- [ ] Network policies applied
- [ ] Audit logging enabled

## AF Alignment

Map to @docs/GOALS.md constraints:

| Constraint | Status | Evidence |
|------------|--------|----------|
| Complex task <5 min latency | [ ] | [Measurement] |
| No credential exposure in logs | [ ] | [Log sample] |
| Immutable audit trail | [ ] | [Audit log location] |

## Output Format

```markdown
## Production Readiness Report

### Service Information
- **Service**: [Name]
- **Version**: [Version]
- **Environment**: [Target env]
- **Assessment date**: [Date]

### SLO Status
| SLI | Target | Current | Error Budget |
|-----|--------|---------|--------------|
| Availability | 99.9% | [X%] | [Y% remaining] |
| P95 Latency | <500ms | [Xms] | [Y% remaining] |
| Error Rate | <1% | [X%] | [Y% remaining] |

### Readiness Score
| Category | Score | Notes |
|----------|-------|-------|
| Observability | X/5 | [Details] |
| Reliability | X/5 | [Details] |
| Operability | X/5 | [Details] |
| Security | X/5 | [Details] |
| **Total** | **X/20** | |

### Verdict
**Status**: READY / NOT READY / CONDITIONAL

**Minimum for READY**: 16/20 with no category below 3

### Blockers (must fix before deploy)
1. [Blocker with remediation]

### Recommendations (should fix soon)
1. [Recommendation]

### Rollback Plan
**Trigger**: [When to rollback]
**Procedure**:
1. [Step 1]
2. [Step 2]
**Verification**: [How to confirm rollback worked]
**Estimated time**: [X minutes]
```

## Trigger Phrases

- "production readiness check"
- "SRE check"
- "ready for deploy?"
- "can we ship this?"
- "/sre-check"

## Agent Delegation

Primary: `sre-allspaw`
Supporting: `verifier` (for running actual checks)

## When to Run

- Before any production deployment
- After major infrastructure changes
- Quarterly production readiness reviews
- After incidents (to verify fixes)
