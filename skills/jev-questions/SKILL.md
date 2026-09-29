---
name: jev-questions
description: >-
  How to write effective TypeSafe Jev questions for coding agents: atomic
  question decomposition, state preparation, confidence gating, and worked
  examples for triage, review gating, and failure classification.
---

# Jev Questions for Coding Agents

Jev is a **System One** decision model: it does not generate text or reason at
length. It evaluates **typed questions** against a **state** and returns
structured answers with calibrated probabilities. Treat it like a fast,
well-calibrated gut you can call from code — not like a chat model.

## The three question types

| Tool | Type | Ask when you want | Returns |
|------|------|-------------------|---------|
| `jev_check` | noul | a yes/no judgment | `yes_probability` 0–1 |
| `jev_route` | choice | exactly one option from a set | chosen option, per-option probabilities, confidence |
| `jev_score` | score | a rating on an ordered rubric | numeric score, level legend, confidence |
| `jev_evaluate` | all three | several judgments on the SAME state in one call | all answers together |

## Rule 1: keep questions atomic

Each question must be a gut-check a senior engineer could answer in seconds
given the context. If it needs extended reasoning or weighs multiple
independent factors, **decompose it**.

Bad: "Rate this pull request" (multi-factor, unpredictable).

Good — decompose into atomic questions against the same diff, then combine in code:

```json
{
  "touches_auth":   {"type": "noul", "instructions": "Does this diff modify authentication or session logic?"},
  "has_tests":      {"type": "noul", "instructions": "Does this diff include or update tests for the changed behavior?"},
  "migration_risk": {"type": "score", "instructions": "How risky is this database migration?", "criteria": ["Trivial", "Review carefully", "Do not ship without a rollback plan"]}
}
```

Then combine: `blocking = (touches_auth > 0.7 and has_tests < 0.5) or migration_risk >= 1.5`

## Rule 2: the state is the evidence

Paste the actual artifact — error output, stack trace, diff, log excerpt,
code — not a vague summary. Jev evaluates what you give it. Trim to the
relevant part; huge states dilute the signal (and cost tokens).

## Rule 3: confidence gates action

Probabilities tell you **what**; confidence tells you **whether to act**.
Suggested gates for coding workflows:

- `yes_probability >= 0.8` (or confidence >= 0.7) → act automatically
- 0.3–0.7 → surface to the human / gather more context, ask again with a sharper state
- `<= 0.2` → act on the negation

Never hard-branch on a bare 0.55.

## Worked patterns

### Failure triage (retry vs escalate)

State: the raw error output. Questions:

```json
{
  "transient":     {"type": "noul", "instructions": "Is this failure transient (retrying could succeed)?"},
  "missing_dep":   {"type": "noul", "instructions": "Is this a missing dependency or version conflict?"},
  "permissions":   {"type": "noul", "instructions": "Is this a permissions or auth error?"}
}
```

Route: transient → retry with backoff; missing_dep → install/fix versions;
permissions → escalate to the user; otherwise → gather more context.

### Change routing

State: task description or bug report. Use `jev_route`:

```json
{"options": {"frontend": "UI/components", "backend": "API/data", "infra": "build/CI/deploy", "unknown": "not enough context"}}
```

### Review gating

State: the diff (or a summary of it). Mix:

```json
{
  "blocks_merge":   {"type": "noul", "instructions": "Would a senior engineer block this diff from merging as-is?"},
  "security_risk":  {"type": "noul", "instructions": "Does this diff introduce a security risk?"},
  "test_coverage":  {"type": "score", "instructions": "How well do the changes test the new behavior?", "criteria": ["No tests", "Partial", "Good", "Excellent"]}
}
```

### Severity scoring

State: a bug report. Use `jev_score` with concrete ordered levels:
`["Cosmetic", "Minor", "Major", "Blocks release"]`

## Anti-patterns

- Asking Jev to "write" or "explain" anything — it returns judgments, not prose.
- Multi-factor questions ("rate the startup pitch") — decompose instead.
- Hard thresholds on low-confidence answers — gate with confidence.
- Vague state ("we have some issues with the build") — paste the real artifact.
- Treating 0.55 as "yes" — that's a coin flip; treat it as uncertainty.

## Response fields

- noul → `yes_probability`
- choice → `choice`, `probabilities`, `confidence`
- score → `score`, `legend`, `confidence`
- `jev_check` adds a convenience `verdict`: yes (≥0.75) / no (≤0.25) / uncertain (between)

## Standing for drafting (adaptation consumer)

When a writing skill is about to draft, call `jev_standing_instruction` (home=`writing_skill`, mode=`legal`|`creative`). It returns mode-filtered resolved standing for the model-facing payload. It is **not** draft generation and **not** `gate_draft`.


