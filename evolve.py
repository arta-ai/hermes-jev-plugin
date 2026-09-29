"""Local correction ledger. Jev labels the miss. Code does not rewrite skills.

A row is a proposal. Nothing here patches a skill, memory, or operator spec.
Promote a label only after it repeats, and only with an explicit apply flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gate import (
    draft_action,
    backend_verb,
    MODES,
    finite_unit,
    authority_from_noul,
    separate_scores,
    reject_sycophancy_signal,
    choice_well_formed,
    score_well_formed,
    validate_answers_against_request,
    request_schema_choice_keys,
)

DEFAULT_ROOT = Path.home() / ".hermes" / "state" / "jev-evolve"
ROOT = DEFAULT_ROOT
LEDGER = ROOT / "ledger.jsonl"
STANDING = ROOT / "standing.json"
SYNTHETIC_PREFIX = "syn-"
HELD_OUT_DIR = ROOT / "held-out-loaw"
PROVE_ISOLATED_DIRNAME = "isolated-prove"
MODEL = "jev-1.13.0"
API_URL = "https://api.typesafe.ai/v1/systemone"


def _sync_paths(root: Path) -> None:
    """Rebind module ledger paths to root. Production default stays DEFAULT_ROOT."""
    global ROOT, LEDGER, STANDING, HELD_OUT_DIR
    ROOT = Path(root)
    LEDGER = ROOT / "ledger.jsonl"
    STANDING = ROOT / "standing.json"
    HELD_OUT_DIR = ROOT / "held-out-loaw"


def isolated_ledger_root(base: Path | None = None, name: str = PROVE_ISOLATED_DIRNAME) -> Path:
    """Return an isolated prove/held-out ledger root (never the production ledger)."""
    parent = Path(base) if base is not None else DEFAULT_ROOT
    return parent / name


class use_isolated_ledger:
    """Context manager: redirect ROOT/LEDGER to an isolated path for prove/held-out.

    Production ~/.hermes/state/jev-evolve/ledger.jsonl is never written while active.
    """

    def __init__(self, root: Path | None = None, *, reset: bool = False):
        self.root = Path(root) if root is not None else isolated_ledger_root()
        self.reset = reset
        self._prior: Path | None = None

    def __enter__(self):
        import shutil
        self._prior = ROOT
        if self.reset and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        _sync_paths(self.root)
        return self.root

    def __exit__(self, *exc):
        if self._prior is not None:
            _sync_paths(self._prior)
        return False

HOMES = {
    "writing_skill": "Voice or structure miss. Propose a writing-skill rule. Do not edit the skill from one row.",
    "turn_router": "Routing miss. Propose a router question. Do not change live routing from one row.",
    "operator_memory": "Standing fact about Arta, not a task log and not client content.",
    "drop": "One-off. Do not store a rule.",
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "too_long": {
        "type": "noul",
        "instructions": "Is the draft longer than the correction asks for?",
        "criteria": {
            "true": "The correction asks for less length, and the draft is still long.",
            "false": "Length is not the miss.",
        },
    },
    "stacked_jargon": {
        "type": "noul",
        "instructions": "Does the draft stack legal or technical jargon where a lawyer would want one plain sentence?",
        "criteria": {
            "true": "A lawyer would not say this out loud in one breath.",
            "false": "The wording is plain enough to say.",
        },
    },
    "mashed_jobs": {
        "type": "noul",
        "instructions": "Does the draft mash two independent jobs into one answer?",
        "criteria": {
            "true": "Two jobs are mixed.",
            "false": "One job only.",
        },
    },
    "invented_fact": {
        "type": "noul",
        "instructions": (
            "Does the draft state a fact that is not supported by the SEPARATE "
            "evidence channel (state.evidence)? The draft itself is NOT evidence. "
            "If state.evidence.available is false, answer is unavailable — do not clear."
        ),
        "criteria": {
            "true": "A fact appears that separate evidence does not support.",
            "false": "Every factual claim is in separate evidence, or the draft makes no factual claim.",
            "unavailable": "No separate evidence channel — do not treat draft-in-state as evidence.",
        },
    },
    "wrong_audience": {
        "type": "noul",
        "instructions": "Is the draft written for the wrong reader, given the correction?",
        "criteria": {
            "true": "The reader named or implied by the correction would not want this voice.",
            "false": "The audience fits.",
        },
    },
    "unsourced_action": {
        "type": "noul",
        "instructions": "Does the draft treat a send, filing, Clio write, or payment as already done when the state does not show that?",
        "criteria": {
            "true": "It claims an effect that has not happened.",
            "false": "It does not claim a completed effect.",
        },
    },
    "hid_option": {
        "type": "noul",
        "instructions": "Does the draft hide a relevant option and show only one path?",
        "criteria": {
            "true": "A relevant option is omitted.",
            "false": "Options are shown, or only one option exists.",
        },
    },
    "correction_is_lock": {
        "type": "noul",
        "instructions": "Is the correction a lock on the last draft, rather than a new task?",
        "criteria": {
            "true": "It corrects the draft. It does not open a new job.",
            "false": "It asks for new work.",
        },
    },
    "private_facts": {
        "type": "noul",
        "instructions": "Does the draft or the correction contain a client name, injury, medical fact, bill, or insurance fact?",
        "criteria": {
            "true": "Client matter facts are present.",
            "false": "No client matter facts are present.",
        },
    },
    "home": {
        "type": "choice",
        "instructions": "Where should a repeated version of this miss be stored? Pick drop if it is a one-off.",
        "criteria": HOMES,
    },
    "speakable": {
        "type": "score",
        "instructions": "How speakable is the draft for a lawyer talking to a person?",
        "criteria": [
            "Cannot be said out loud.",
            "Sayable only after a rewrite.",
            "Sayable with a small cut.",
            "Already speakable.",
        ],
    },
}


def registered_homes() -> frozenset[str]:
    """Trusted home option set from the request schema (HOMES / QUESTIONS['home'])."""
    keys = request_schema_choice_keys(QUESTIONS.get("home"))
    if keys is None:
        keys = list(HOMES.keys())
    return frozenset(keys)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evidence_channel(evidence: dict | None) -> dict[str, Any]:
    """Separate evidence for invented_fact (primer #9 / RO #4).

    Draft-in-state is NOT independent evidence. Missing/empty/draft-sourced
    evidence → unavailable (fail-closed; do not clear invented_fact).
    """
    if not evidence or not isinstance(evidence, dict):
        return {
            "available": False,
            "backend": "clarify",
            "text": None,
            "sha256": None,
            "note": "No separate evidence channel — invented_fact unavailable; draft is not evidence.",
        }
    if evidence.get("is_draft") or evidence.get("source") == "draft":
        return {
            "available": False,
            "backend": "clarify",
            "text": None,
            "sha256": None,
            "note": "Draft-sourced payload rejected — not independent evidence.",
        }
    blob = evidence.get("text") or evidence.get("excerpts")
    if not isinstance(blob, str) or not blob.strip():
        return {
            "available": False,
            "backend": "clarify",
            "text": None,
            "sha256": None,
            "note": "Evidence empty — invented_fact unavailable.",
        }
    return {
        "available": True,
        "backend": "act",
        "text": blob,
        "sha256": _sha(blob),
        "note": "Separate evidence channel present.",
    }


def outbound_privacy_approval(
    draft: str,
    correction: str,
    *,
    allow_outbound: bool = False,
) -> dict[str, Any]:
    """Local privacy/model approval. MUST run BEFORE any outbound call (RO #5).

    Post-call private_facts noul still gates raw text storage, but cannot
    authorize the outbound itself (it arrives too late).
    """
    if not allow_outbound:
        return {
            "ok": False,
            "reason": "allow_outbound_required",
            "backend": "clarify",
            "flagged_hints": False,
            "note": "Explicit allow_outbound must precede model call; privacy noul alone is not sufficient and arrives too late.",
        }
    blob = f"{draft}\n{correction}".lower()
    hints = (
        "ssn",
        "social security",
        "date of birth",
        "medical record",
        " policy number ",
    )
    flagged = any(h in blob for h in hints)
    return {
        "ok": True,
        "reason": "approved",
        "backend": "act",
        "flagged_hints": flagged,
        "note": "Local pre-outbound approval granted; post-call private_facts still gates raw text storage.",
    }


def call_jev(
    state: dict,
    questions: dict | None = None,
    *,
    allow_outbound: bool = False,
    privacy_approval: dict | None = None,
) -> dict:
    """Call TypeSafe Jev. RO #5: refuse unless privacy/model approval already passed.

    HTTP errors never leak provider body. Response answers are validated against
    the trusted request questions before return; unbound/malformed raises RuntimeError.
    """
    approval = privacy_approval
    if approval is None:
        approval = {
            "ok": bool(allow_outbound),
            "reason": "allow_outbound_true" if allow_outbound else "allow_outbound_required",
            "note": "call_jev requires allow_outbound=True (privacy/model approval before outbound).",
        }
    if not approval.get("ok"):
        raise RuntimeError(
            "call_jev blocked: privacy/model approval must run BEFORE outbound call "
            f"({approval.get('reason') or 'denied'})"
        )
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    req_questions = questions or QUESTIONS
    if not isinstance(req_questions, dict) or not req_questions:
        raise RuntimeError("call_jev requires a non-empty trusted questions schema")
    payload = {"model": MODEL, "state": state, "questions": req_questions}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # Sanitize: discard provider body — status only (no leak).
        try:
            exc.read()  # drain without exposing
        except Exception:
            pass
        raise RuntimeError(f"TypeSafe HTTP {exc.code}: request failed") from None
    except urllib.error.URLError:
        raise RuntimeError("TypeSafe unreachable: request failed") from None
    if not isinstance(raw, dict):
        raise RuntimeError("TypeSafe response invalid: not an object")
    answers = raw.get("answers")
    bound = validate_answers_against_request(
        answers if isinstance(answers, dict) else None, req_questions
    )
    if not bound["ok"]:
        raise RuntimeError(
            "TypeSafe response failed request-bound validation: "
            + ",".join(f"{k}={v}" for k, v in sorted(bound["failures"].items())[:8])
        )
    raw = dict(raw)
    raw["_request_bound"] = True
    raw["_validation"] = {"ok": True, "failures": {}}
    return raw

def _load_standing() -> dict:
    if not STANDING.exists():
        return {"labels": {}, "note": "No label is standing until it repeats and apply is explicit."}
    return json.loads(STANDING.read_text())


def record(
    draft: str,
    correction: str,
    allow_text: bool = False,
    mode: str = "legal",
    evidence: dict | None = None,
    allow_outbound: bool = False,
    *,
    synthetic: bool = False,
    append: bool = True,
) -> dict:
    """Label one correction and optionally append a proposal row. Does not apply a rule.

    RO #5: outbound_privacy_approval runs BEFORE call_jev.
    RO #4: invented_fact uses separate evidence channel (draft is not evidence).
    Request-bound validation runs BEFORE any ledger append — unknown home,
    missing answers, or unbound choice/score never write a row.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    approval = outbound_privacy_approval(draft, correction, allow_outbound=allow_outbound)
    if not approval.get("ok"):
        return {
            "ok": False,
            "blocked": True,
            "applied": False,
            "appended": False,
            "backend": "clarify",
            "privacy_approval": approval,
            "draft_sha256": _sha(draft),
            "correction_sha256": _sha(correction),
            "note": "Outbound blocked — privacy/model approval must precede API call.",
        }
    ev = evidence_channel(evidence)
    state = {
        "draft": draft,
        "correction": correction,
        "standing_rules": _load_standing(),
        "mode": mode,
        "evidence": {
            "available": ev["available"],
            "text": ev["text"] if ev["available"] else None,
            "sha256": ev.get("sha256"),
            "note": ev.get("note"),
        },
    }
    questions = dict(QUESTIONS)
    if not ev["available"]:
        questions.pop("invented_fact", None)
    response = call_jev(state, questions, allow_outbound=True, privacy_approval=approval)
    answers = dict(response.get("answers") or {})
    if not ev["available"]:
        answers["invented_fact"] = {
            "type": "noul",
            "noul": None,
            "unavailable": True,
            "backend": "clarify",
            "note": ev.get("note") or "primer #9 — no independent evidence",
        }
    # Validate only request-asked keys. Synthetic invented_fact injection
    # (primer #9 unavailable marker) is added for draft_action and is not an
    # "answer" from the provider — exclude it from extras rejection.
    bound = validate_answers_against_request(
        {k: answers[k] for k in questions if k in answers},
        questions,
    )
    homes = registered_homes()
    home_ans = answers.get("home") if isinstance(answers.get("home"), dict) else {}
    home_choice = home_ans.get("choice")
    home_ok = (
        isinstance(home_choice, str)
        and home_choice in homes
        and choice_well_formed(home_ans, QUESTIONS.get("home"))
    )
    if not bound["ok"] or not home_ok:
        failures = dict(bound.get("failures") or {})
        if not home_ok:
            failures["home"] = "unregistered_or_unbound_home"
        return {
            "ok": False,
            "blocked": True,
            "applied": False,
            "appended": False,
            "backend": "investigate",
            "proposed_home": None,
            "validation_failures": failures,
            "draft_sha256": _sha(draft),
            "correction_sha256": _sha(correction),
            "privacy_approval": approval,
            "note": "Request-bound validation failed — ledger row NOT appended.",
        }
    private = (answers.get("private_facts") or {}).get("noul")
    private_ok = finite_unit(private)
    store_text = bool(allow_text) and private_ok is not None and private_ok < 0.7
    home = home_choice
    gate = draft_action(answers, mode=mode)
    row_id = _sha(_now() + _sha(draft) + _sha(correction))[:16]
    row = {
        "id": row_id,
        "ts": _now(),
        "model": response.get("model"),
        "mode": mode,
        "draft_sha256": _sha(draft),
        "correction_sha256": _sha(correction),
        "draft": draft if store_text else None,
        "correction": correction if store_text else None,
        "text_stored": store_text,
        "labels": answers,
        "draft_gate": gate,
        "backend": gate.get("backend"),
        "proposed_home": home,
        "home_confidence": home_ans.get("confidence"),
        "applied": False,
        "synthetic": bool(synthetic),
        "prefs_eligible": False,
        "evidence_sha256": ev.get("sha256"),
        "evidence_available": ev["available"],
        "privacy_approval": approval,
        "usage": response.get("usage"),
        "ok": True,
        "appended": False,
        "proposal": {
            "version": 1,
            "home": home,
            "mode": mode,
            "scoped_rule": {
                "kind": "correction_proposal",
                "instruction_precedence": "explicit_correction_over_standing",
                "mode_separation": mode,
                "draft_sha256": _sha(draft),
                "correction_sha256": _sha(correction),
                "labels_digest": _sha(json.dumps(answers, sort_keys=True, default=str)),
            },
            "rollback": {
                "action": "rollback_proposal",
                "row_id": row_id,
                "restores": "pre-apply standing snapshot",
            },
        },
    }
    if synthetic:
        row = mark_synthetic(row)
        row["proposal"]["rollback"]["row_id"] = row["id"]
    if append:
        ROOT.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        row["appended"] = True
    return row

def gate_draft(
    draft: str,
    evidence: dict | None = None,
    *,
    allow_outbound: bool = False,
) -> dict:
    """Score a draft against the miss labels. Does not write the ledger.

    RO #5: privacy approval before outbound. RO #4: separate evidence channel.
    """
    approval = outbound_privacy_approval(draft, "", allow_outbound=allow_outbound)
    if not approval.get("ok"):
        return {
            "ok": False,
            "blocked": True,
            "backend": "clarify",
            "privacy_approval": approval,
            "draft_gate": {
                "action": "review",
                "backend": "clarify",
                "revise": [],
                "review": ["outbound_privacy_blocked"],
                "chips": [],
                "malformed": False,
                "missing_required": [],
            },
            "note": "Outbound blocked — privacy/model approval must precede API call.",
        }
    ev = evidence_channel(evidence)
    state = {
        "draft": draft,
        "correction": "",
        "standing_rules": _load_standing(),
        "evidence": {
            "available": ev["available"],
            "text": ev["text"] if ev["available"] else None,
            "sha256": ev.get("sha256"),
            "note": ev.get("note"),
        },
    }
    questions = dict(QUESTIONS)
    if not ev["available"]:
        questions.pop("invented_fact", None)
    response = call_jev(state, questions, allow_outbound=True, privacy_approval=approval)
    answers = dict(response.get("answers") or {})
    if not ev["available"]:
        answers["invented_fact"] = {
            "type": "noul",
            "noul": None,
            "unavailable": True,
            "backend": "clarify",
            "note": ev.get("note"),
        }
    return {
        "model": response.get("model"),
        "draft_gate": draft_action(answers),
        "speakable": answers.get("speakable"),
        "evidence_available": ev["available"],
        "privacy_approval": approval,
        "usage": response.get("usage"),
    }


def _read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows: list[dict] = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_row(row_id: str) -> dict | None:
    for row in _read_ledger():
        if row.get("id") == row_id:
            return row
    return None


def replay_row(row_id: str, mode: str | None = None) -> dict:
    """Replay a stored correction proposal without applying a rule.

    Recomputes draft_gate from stored labels (no new API call). Exact
    draft_sha256 / correction_sha256 must match the ledger row.
    """
    row = load_row(row_id)
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    use_mode = mode or row.get("mode") or "legal"
    if use_mode not in MODES:
        raise ValueError(f"unknown mode: {use_mode}")
    recomputed = draft_action(row.get("labels") or {}, mode=use_mode)
    return {
        "id": row["id"],
        "draft_sha256": row["draft_sha256"],
        "correction_sha256": row["correction_sha256"],
        "applied": row.get("applied", False),
        "mode": use_mode,
        "draft_gate": recomputed,
        "backend": recomputed.get("backend"),
        "proposed_home": row.get("proposed_home"),
        "text_stored": row.get("text_stored"),
        "mutated": False,
    }


def prefs_snapshot() -> dict:
    """Reversible standing prefs. Empty until an explicit apply promotes a label."""
    standing = _load_standing()
    return {
        "path": str(STANDING),
        "labels": standing.get("labels") or {},
        "note": standing.get("note") or "",
        "reversible": True,
        "apply_default": False,
    }


def prefs_clear() -> dict:
    """Drop standing labels (reversible prefs). Does not touch skills or memory."""
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": {},
        "note": "Cleared. No label is standing until it repeats and apply is explicit.",
        "cleared_at": _now(),
    }
    STANDING.write_text(json.dumps(payload, indent=2) + "\n")
    return prefs_snapshot()


def mark_synthetic(row: dict) -> dict:
    """Tag synthetic proof rows — never eligible for preference learning."""
    row = dict(row)
    row["synthetic"] = True
    row["prefs_eligible"] = False
    rid = str(row.get("id") or "")
    if not rid.startswith(SYNTHETIC_PREFIX):
        row["id"] = SYNTHETIC_PREFIX + rid
    return row


def verify_claims_against_evidence(claims: list[str], evidence: dict | None) -> dict:
    """Lexical occurrence check only — NOT factual/semantic verification.

    Substring presence proves occurrence-only. Semantic support is unavailable
    unless a separate assessment is supplied. Negating text that contains the
    claim words must NOT authorize (e.g. "...claim is false" still matches
    lexically but semantic_support=unavailable).
    """
    if not evidence or not isinstance(evidence, dict):
        return {
            "ok": False,
            "backend": "clarify",
            "unsupported": list(claims),
            "occurrence_only": [],
            "semantic_support": "unavailable",
            "note": "No separate evidence supplied — do not authorize.",
        }
    blob = evidence.get("text") or evidence.get("excerpts") or ""
    if not isinstance(blob, str) or not blob.strip():
        return {
            "ok": False,
            "backend": "clarify",
            "unsupported": list(claims),
            "occurrence_only": [],
            "semantic_support": "unavailable",
            "note": "Evidence empty — do not authorize.",
        }
    blob_l = blob.lower()
    occurrence_only = []
    unsupported = []
    for claim in claims:
        if not claim or not isinstance(claim, str):
            unsupported.append(claim)
            continue
        if claim.lower() in blob_l:
            occurrence_only.append(claim)
        else:
            unsupported.append(claim)
    # Lexical hits are NEVER sufficient for ok/act authorization.
    semantic = evidence.get("semantic_support")
    if semantic in (True, "supported", "yes"):
        # Only an explicit separate semantic assessment may authorize.
        ok = len(unsupported) == 0
        return {
            "ok": ok,
            "backend": "act" if ok else "investigate",
            "unsupported": unsupported,
            "occurrence_only": occurrence_only,
            "semantic_support": "supported",
            "evidence_sha256": _sha(blob),
            "note": "Separate semantic_support provided on evidence channel.",
        }
    return {
        "ok": False,
        "backend": "investigate",
        "unsupported": unsupported,
        "occurrence_only": occurrence_only,
        "semantic_support": "unavailable",
        "evidence_sha256": _sha(blob),
        "note": "Lexical occurrence only — substring presence is not factual verification.",
    }

def _standing_snapshot() -> dict:
    return dict(_load_standing())


def propose_correction_rule(row_id: str) -> dict:
    """Build a usable scoped proposal from a ledger row (no apply).

    Preserves explicit instruction precedence and legal/creative mode separation.
    """
    row = load_row(row_id)
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    if row.get("synthetic") or str(row.get("id", "")).startswith(SYNTHETIC_PREFIX):
        return {
            "ok": False,
            "reason": "synthetic_excluded",
            "note": "Synthetic proof rows stay out of preference learning.",
        }
    home = row.get("proposed_home") or "drop"
    if home not in registered_homes() or home == "drop":
        return {"ok": False, "reason": "home_not_promotable", "home": home}
    mode = row.get("mode") or "legal"
    if mode not in MODES:
        return {"ok": False, "reason": "unknown_mode", "mode": mode}
    proposal = row.get("proposal") or {}
    scoped = proposal.get("scoped_rule") or {
        "kind": "correction_proposal",
        "instruction_precedence": "explicit_correction_over_standing",
        "mode_separation": mode,
        "draft_sha256": row.get("draft_sha256"),
        "correction_sha256": row.get("correction_sha256"),
    }
    version = int(proposal.get("version") or 1)
    return {
        "ok": True,
        "row_id": row["id"],
        "version": version,
        "home": home,
        "mode": mode,
        "scoped_rule": scoped,
        "instruction_precedence": "explicit_correction_over_standing",
        "rollback": {
            "action": "rollback_proposal",
            "row_id": row["id"],
            "version": version,
        },
        "applied": False,
        "note": "Proposal only — explicit apply required; rollback available.",
    }


def replay_proposal(row_id: str) -> dict:
    """Replay a stored proposal without mutating standing or skills."""
    proposal = propose_correction_rule(row_id)
    if not proposal.get("ok"):
        return proposal
    again = replay_row(row_id)
    return {
        "ok": True,
        "row_id": row_id,
        "proposal": proposal,
        "replay": again,
        "mutated": False,
        "standing_unchanged": True,
        "instruction_precedence": "explicit_correction_over_standing",
        "mode_separation": proposal.get("mode"),
    }


def rollback_proposal(row_id: str) -> dict:
    """Roll back an applied proposal for row_id using the standing snapshot.

    If the row was never applied, standing is left unchanged. Always reversible.
    """
    row = load_row(row_id)
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    standing = _load_standing()
    labels = standing.setdefault("labels", {})
    removed = None
    for home, meta in list(labels.items()):
        if isinstance(meta, dict) and meta.get("from_row") == row_id:
            removed = {home: labels.pop(home)}
            break
    history = standing.setdefault("rollback_history", [])
    history.append({
        "row_id": row_id,
        "removed": removed,
        "rolled_back_at": _now(),
        "version": (row.get("proposal") or {}).get("version"),
    })
    ROOT.mkdir(parents=True, exist_ok=True)
    STANDING.write_text(json.dumps(standing, indent=2) + "\n")
    return {
        "ok": True,
        "row_id": row_id,
        "removed": removed,
        "standing": prefs_snapshot(),
        "note": "Rollback applied; explicit instruction precedence preserved.",
    }


def promote_to_standing(row_id: str, apply: bool = False) -> dict:
    """Promote a scoped proposal only with explicit apply. Synthetic rows rejected.

    Stores a usable scoped rule + version + rollback handle (not hashes/home alone).
    Preserves explicit instruction precedence and legal/creative mode separation.
    """
    row = load_row(row_id)
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    if row.get("synthetic") or str(row.get("id", "")).startswith(SYNTHETIC_PREFIX):
        return {
            "ok": False,
            "reason": "synthetic_excluded",
            "note": "Synthetic proof rows stay out of preference learning.",
        }
    proposal = propose_correction_rule(row_id)
    if not proposal.get("ok"):
        return proposal
    if not apply:
        return {
            "ok": False,
            "reason": "apply_false",
            "proposal": proposal,
            "note": "Explicit apply required. Proposal/replay/rollback available without apply.",
        }
    if not row.get("prefs_eligible"):
        return {"ok": False, "reason": "not_prefs_eligible", "note": "Row not marked prefs_eligible."}
    standing = _load_standing()
    # snapshot for rollback
    pre = {
        "labels": dict(standing.get("labels") or {}),
        "snapshotted_at": _now(),
        "for_row": row_id,
    }
    snaps = standing.setdefault("apply_snapshots", {})
    snaps[row_id] = pre
    labels = standing.setdefault("labels", {})
    home = proposal["home"]
    mode = proposal["mode"]
    version = proposal["version"]
    labels[home] = {
        "from_row": row["id"],
        "correction_sha256": row.get("correction_sha256"),
        "draft_sha256": row.get("draft_sha256"),
        "promoted_at": _now(),
        "version": version,
        "mode": mode,
        "scoped_rule": proposal["scoped_rule"],
        "instruction_precedence": "explicit_correction_over_standing",
        "rollback": proposal["rollback"],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    STANDING.write_text(json.dumps(standing, indent=2) + "\n")
    return {
        "ok": True,
        "standing": prefs_snapshot(),
        "proposal": proposal,
        "rollback": proposal["rollback"],
        "version": version,
        "mode": mode,
    }

def held_out_loaw_calibrate() -> dict:
    """Offline gate-plumbing check on held-out LOAW-shaped fixtures (no LIVE API).

    Feeds prewritten model labels through draft_action / separate_scores /
    authority guards. This proves GATE PLUMBING only — it is NOT held-out model
    calibration and must not be reported as model calibration.
    Preference / grounding / success scored separately. Synthetic; not prefs learning.
    """
    tasks = [
        {
            "id": "loaw-held-demand-length",
            "draft_labels": {"too_long": {"type": "noul", "noul": 0.88}, "invented_fact": {"type": "noul", "noul": 0.12}},
            "preference_fit": 0.4,
            "factual_grounding": 0.85,
            "task_success": 0.55,
            "agreement": 0.95,
        },
        {
            "id": "loaw-held-clio-unsourced",
            "draft_labels": {"unsourced_action": {"type": "noul", "noul": 0.81}, "hid_option": {"type": "noul", "noul": 0.2}},
            "preference_fit": 0.6,
            "factual_grounding": 0.7,
            "task_success": 0.3,
            "agreement": 0.9,
        },
        {
            "id": "loaw-held-malformed-noul",
            "draft_labels": {"too_long": {"type": "noul", "noul": True}, "stacked_jargon": {"noul": float("nan")}},
            "preference_fit": 1.5,
            "factual_grounding": "yes",
            "task_success": False,
            "agreement": 1.0,
        },
    ]
    results = []
    for task in tasks:
        gate = draft_action(task["draft_labels"], mode="legal")
        scores = separate_scores(task["preference_fit"], task["factual_grounding"], task["task_success"])
        syc = reject_sycophancy_signal(task["agreement"])
        auth = authority_from_noul(task["draft_labels"].get("unsourced_action") or task["draft_labels"].get("too_long") or {})
        results.append(
            {
                "id": task["id"],
                "backend": gate["backend"],
                "malformed": gate.get("malformed", False),
                "scores": scores,
                "sycophancy": syc,
                "noul_authority": auth,
                "prefs_eligible": False,
                "synthetic": True,
            }
        )
    # evidence check sample
    evidence = verify_claims_against_evidence(
        ["demand mailed 2026-09-01"],
        {"text": "Matter note: LOR sent; demand not yet mailed."},
    )
    return {
        "ok": all(r["backend"] in ("act", "investigate", "clarify") for r in results)
        and results[-1]["malformed"] is True
        and results[-1]["scores"]["usable"] is False
        and all(r["sycophancy"]["reward_allowed"] is False for r in results)
        and evidence["ok"] is False
        and evidence.get("semantic_support") == "unavailable"
        and auth_check_ok(results),
        "kind": "gate_plumbing_offline",
        "model_calibration": False,
        "tasks": results,
        "evidence_sample": evidence,
        "note": "Gate plumbing only (prewritten labels) — NOT held-out model calibration. Synthetic; not prefs learning.",
    }


def auth_check_ok(results: list) -> bool:
    return all(r["noul_authority"]["authority"] is False for r in results)


def held_out_check(row_id: str | None = None) -> dict:
    """Offline held-out: ledger row persists SHAs; replay matches; applied stays false."""
    rows = _read_ledger()
    if not rows:
        raise RuntimeError("ledger empty — run prove first")
    row = load_row(row_id) if row_id else rows[-1]
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    again = replay_row(row["id"])
    ok = (
        again["draft_sha256"] == row["draft_sha256"]
        and again["correction_sha256"] == row["correction_sha256"]
        and again["applied"] is False
        and row.get("applied") is False
        and again["backend"] in ("act", "investigate", "clarify")
        and again["mutated"] is False
    )
    return {
        "ok": ok,
        "id": row["id"],
        "draft_sha256": row["draft_sha256"],
        "correction_sha256": row["correction_sha256"],
        "backend": again["backend"],
        "mode": again["mode"],
        "applied": again["applied"],
        "prefs": prefs_snapshot(),
    }


def _offline_checks() -> None:
    from gate import decide_answer, MISS_LABELS, choice_well_formed, score_well_formed

    # High miss → revise even if other labels missing (revise outranks missing)
    assert draft_action({"too_long": {"type": "noul", "noul": 0.91}})["action"] == "revise"
    assert draft_action({"too_long": {"type": "noul", "noul": 0.91}})["backend"] == "clarify"
    assert draft_action({"too_long": {"type": "noul", "noul": 0.5}})["action"] == "review"
    assert draft_action({"too_long": {"type": "noul", "noul": 0.5}})["backend"] == "investigate"
    # RO #3: partial label set with low noul → investigate, NEVER act/pass
    partial = draft_action({"too_long": {"type": "noul", "noul": 0.01}})
    assert partial["action"] == "review" and partial["backend"] == "investigate"
    assert "missing_required_labels" in partial["review"]
    assert "too_long" not in partial["missing_required"]
    # Full clean set → pass
    full_clean = {name: {"type": "noul", "noul": 0.05} for name in MISS_LABELS}
    assert draft_action(full_clean)["action"] == "pass"
    assert draft_action(full_clean)["backend"] == "act"
    # creative mode: stacked_jargon alone does not force revise (full set required for pass)
    full_creative = {name: {"type": "noul", "noul": 0.05} for name in MISS_LABELS}
    full_creative["stacked_jargon"] = {"type": "noul", "noul": 0.91}
    creative = draft_action(full_creative, mode="creative")
    assert creative["action"] == "pass" and creative["backend"] == "act"
    legal = draft_action({"stacked_jargon": {"type": "noul", "noul": 0.91}}, mode="legal")
    assert legal["action"] == "revise" and legal["backend"] == "clarify"

    assert decide_answer({"type": "noul", "noul": 0.99}, "effect") == "approve"
    assert backend_verb({"type": "noul", "noul": 0.99}, "effect") == "clarify"
    # RO #1: unknown type high confidence → investigate (not act)
    assert backend_verb({"type": "unknown", "confidence": 0.99}, "ordinary") == "investigate"
    assert decide_answer({"type": "unknown", "confidence": 0.99}, "effect") == "stop"
    # RO #2: choice without payload / without trusted request → investigate
    assert backend_verb({"type": "choice", "confidence": 0.99}, "ordinary") == "investigate"
    choice_req = {"type": "choice", "criteria": {"drop": "d", "keep": "k"}}
    well = {
        "type": "choice",
        "choice": "drop",
        "confidence": 0.9,
        "probabilities": {"drop": 0.9, "keep": 0.1},
    }
    # Without trusted request schema → fail-closed (never trust answer keys)
    assert choice_well_formed(well) is False
    assert backend_verb(well, "ordinary") == "investigate"
    assert choice_well_formed(well, choice_req) is True
    assert backend_verb(well, "ordinary", choice_req) == "act"
    # Adversary answer self-supplying request_options / option keys → rejected
    evil = {
        "type": "choice",
        "choice": "attacker",
        "confidence": 0.99,
        "probabilities": {"attacker": 1.0},
        "request_options": ["attacker"],
        "expected_options": ["attacker"],
    }
    assert choice_well_formed(evil) is False
    assert choice_well_formed(evil, choice_req) is False
    assert backend_verb(evil, "ordinary", choice_req) == "investigate"
    # RO dist/EV: negative/unnormalized probs + OOB score must NOT act
    bad_neg = {"type": "choice", "choice": "x", "probabilities": {"x": -8}, "confidence": 0.99}
    assert choice_well_formed(bad_neg, {"type": "choice", "criteria": {"x": "x"}}) is False
    assert backend_verb(bad_neg, "ordinary", {"type": "choice", "criteria": {"x": "x"}}) == "investigate"
    bad_score = {"type": "score", "score": 999, "confidence": 0.99}
    score_req = {"type": "score", "criteria": ["low", "mid", "high"]}
    assert score_well_formed(bad_score) is False
    assert score_well_formed(bad_score, score_req) is False
    assert backend_verb(bad_score, "ordinary", score_req) == "investigate"
    well_score = {
        "type": "score",
        "score": 2.0,
        "confidence": 0.95,
        "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
    }
    # Answer-embedded criteria alone is NOT trusted
    assert score_well_formed({**well_score, "criteria": ["low", "mid", "high"]}) is False
    assert score_well_formed(well_score, score_req) is True
    assert backend_verb(well_score, "ordinary", score_req) == "act"
    # selected must be argmax; missing option vs request set fails closed
    not_argmax = {"type": "choice", "choice": "keep", "confidence": 0.99, "probabilities": {"drop": 0.9, "keep": 0.1}}
    assert choice_well_formed(not_argmax, choice_req) is False
    assert backend_verb(not_argmax, "ordinary", choice_req) == "investigate"
    assert backend_verb({"type": "choice", "confidence": 0.2}, "ordinary") == "investigate"
    assert decide_answer({"type": "choice", "confidence": 0.2}, "effect") == "stop"
    assert decide_answer({"type": "choice", "confidence": 0.95}, "effect") == "stop"  # no payload
    assert decide_answer({**well, "confidence": 0.95}, "effect", choice_req) == "approve"
    assert decide_answer({"type": "choice", "confidence": 0.2}, "harmless") == "review"  # no payload
    assert finite_unit(True) is None
    assert finite_unit(float("nan")) is None
    assert finite_unit(1.5) is None
    assert finite_unit(0.5) == 0.5
    assert decide_answer({}, "effect") == "stop"
    assert decide_answer({"type": "noul", "noul": True}, "ordinary") == "review"
    assert authority_from_noul({"type": "noul", "noul": 0.99})["authority"] is False
    sep = separate_scores(0.9, 0.2, 0.8)
    assert sep["combined_forbidden"] is True and sep["authorize"] is False
    assert reject_sycophancy_signal(0.99)["reward_allowed"] is False
    bad = draft_action({"too_long": {"noul": True}})
    assert bad["backend"] == "investigate" and bad["malformed"] is True
    # Occurrence-only ≠ factual verification
    v = verify_claims_against_evidence(
        ["package shipped"],
        {"text": "The package shipped claim is false. It has not shipped."},
    )
    assert v["ok"] is False and v["semantic_support"] == "unavailable"
    assert "package shipped" in v["occurrence_only"]
    assert verify_claims_against_evidence(["x"], None)["ok"] is False
    # RO #4: evidence_channel rejects draft-as-evidence / missing
    assert evidence_channel(None)["available"] is False
    assert evidence_channel({"source": "draft", "text": "x"})["available"] is False
    assert evidence_channel({"text": "LOR sent 2026-09-01"})["available"] is True
    # RO #5: outbound blocked without allow_outbound
    blocked = outbound_privacy_approval("draft", "shorter", allow_outbound=False)
    assert blocked["ok"] is False and blocked["backend"] == "clarify"
    ok_out = outbound_privacy_approval("draft", "shorter", allow_outbound=True)
    assert ok_out["ok"] is True
    try:
        call_jev({"draft": "x"}, allow_outbound=False)
        raise AssertionError("call_jev should block without allow_outbound")
    except RuntimeError as exc:
        assert "BEFORE outbound" in str(exc) or "allow_outbound" in str(exc)
    # HTTP error sanitize: provider body must not appear
    import io, urllib.error
    class _FakeHTTP(urllib.error.HTTPError):
        def __init__(self):
            urllib.error.HTTPError.__init__(
                self, "https://api.typesafe.ai/v1/systemone", 500,
                "Internal", hdrs=None, fp=io.BytesIO(b"SECRET_PROVIDER_BODY_LEAK")
            )
    real_urlopen = urllib.request.urlopen
    def _boom(*a, **k):
        raise _FakeHTTP()
    urllib.request.urlopen = _boom
    os.environ["TYPESAFE_API_KEY"] = os.environ.get("TYPESAFE_API_KEY") or "test-key-not-real"
    try:
        call_jev({"draft": "x"}, {"q": {"type": "noul", "instructions": "y?"}}, allow_outbound=True,
                 privacy_approval={"ok": True, "reason": "test"})
        raise AssertionError("expected HTTP error")
    except RuntimeError as exc:
        msg = str(exc)
        assert "SECRET_PROVIDER_BODY_LEAK" not in msg
        assert "HTTP 500" in msg and "request failed" in msg
    finally:
        urllib.request.urlopen = real_urlopen
    # held-out-loaw is gate plumbing, not model calibration
    cal = held_out_loaw_calibrate()
    assert cal["ok"] is True and cal.get("model_calibration") is False
    assert "NOT held-out model calibration" in cal["note"]
    # proposal / replay / rollback contract shape (no real pref write on production)
    proposal_shape = {
        "version": 1,
        "scoped_rule": {"kind": "correction_proposal", "instruction_precedence": "explicit_correction_over_standing"},
        "rollback": {"action": "rollback_proposal"},
    }
    assert proposal_shape["scoped_rule"]["instruction_precedence"] == "explicit_correction_over_standing"

def main(argv: list[str]) -> int:
    _offline_checks()
    command = argv[1] if len(argv) > 1 else "prove"
    if command == "prove":
        # Isolated ledger only — never write production ~/.hermes/state/jev-evolve/ledger.jsonl
        draft = (
            "It is axiomatic that the aforementioned liability posture, viewed through "
            "the lens of comparative fault and the totality of the evidentiary matrix, "
            "clearly establishes a compelling narrative for immediate escalation."
        )
        correction = "shorter"
        with use_isolated_ledger(reset=True) as iso_root:
            # synthetic tagged BEFORE append — no rewrite race on production
            row = record(
                draft, correction,
                allow_text=True, allow_outbound=True, synthetic=True, append=True,
            )
            public = {
                "id": row.get("id"),
                "ok": row.get("ok"),
                "appended": row.get("appended"),
                "model": row.get("model"),
                "proposed_home": row.get("proposed_home"),
                "home_confidence": row.get("home_confidence"),
                "applied": row.get("applied"),
                "synthetic": row.get("synthetic"),
                "prefs_eligible": row.get("prefs_eligible"),
                "draft_gate": row.get("draft_gate"),
                "speakable": ((row.get("labels") or {}).get("speakable") or {}).get("score"),
                "speakable_confidence": ((row.get("labels") or {}).get("speakable") or {}).get("confidence"),
                "correction_is_lock": ((row.get("labels") or {}).get("correction_is_lock") or {}).get("noul"),
                "private_facts": ((row.get("labels") or {}).get("private_facts") or {}).get("noul"),
                "usage": row.get("usage"),
                "ledger": str(LEDGER),
                "isolated_root": str(iso_root),
                "production_ledger_untouched": str(DEFAULT_ROOT / "ledger.jsonl"),
                "proposal": row.get("proposal"),
                "validation_failures": row.get("validation_failures"),
                "note": row.get("note"),
            }
            print(json.dumps(public, indent=2))
            return 0 if row.get("ok") and row.get("appended") else 1
    if command == "gate":
        print(json.dumps(gate_draft(argv[2]), indent=2))
        return 0
    if command == "replay":
        if len(argv) < 3:
            print("use: replay <row_id>", file=sys.stderr)
            return 2
        print(json.dumps(replay_row(argv[2]), indent=2))
        return 0
    if command == "held-out-loaw":
        result = held_out_loaw_calibrate()
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    if command == "held-out":
        # Read/prove against isolated ledger only — zero production preference writes
        row_id = argv[2] if len(argv) > 2 else None
        with use_isolated_ledger(reset=False):
            result = held_out_check(row_id)
            result["isolated_ledger"] = str(LEDGER)
            result["production_ledger_untouched"] = str(DEFAULT_ROOT / "ledger.jsonl")
            print(json.dumps(result, indent=2))
            return 0 if result.get("ok") else 1
    if command == "prefs":
        sub = argv[2] if len(argv) > 2 else "show"
        if sub == "clear":
            print(json.dumps(prefs_clear(), indent=2))
        else:
            print(json.dumps(prefs_snapshot(), indent=2))
        return 0
    if command == "propose":
        if len(argv) < 3:
            print("use: propose <row_id>", file=sys.stderr)
            return 2
        print(json.dumps(propose_correction_rule(argv[2]), indent=2))
        return 0
    if command == "replay-proposal":
        if len(argv) < 3:
            print("use: replay-proposal <row_id>", file=sys.stderr)
            return 2
        print(json.dumps(replay_proposal(argv[2]), indent=2))
        return 0
    if command == "rollback":
        if len(argv) < 3:
            print("use: rollback <row_id>", file=sys.stderr)
            return 2
        print(json.dumps(rollback_proposal(argv[2]), indent=2))
        return 0
    print(
        "use: prove | gate <draft> | replay <row_id> | held-out [row_id] | held-out-loaw | "
        "prefs [show|clear] | propose <row_id> | replay-proposal <row_id> | rollback <row_id>",
        file=sys.stderr,
    )
    return 2

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
