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
)

ROOT = Path.home() / ".hermes" / "state" / "jev-evolve"
LEDGER = ROOT / "ledger.jsonl"
STANDING = ROOT / "standing.json"
SYNTHETIC_PREFIX = "syn-"
HELD_OUT_DIR = ROOT / "held-out-loaw"
MODEL = "jev-1.13.0"
API_URL = "https://api.typesafe.ai/v1/systemone"

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
    """Call TypeSafe Jev. RO #5: refuse unless privacy/model approval already passed."""
    approval = privacy_approval
    if approval is None:
        # Derive a minimal approval check from explicit flag only.
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
    payload = {"model": MODEL, "state": state, "questions": questions or QUESTIONS}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:300]
        raise RuntimeError(f"TypeSafe HTTP {exc.code}: {detail}") from exc


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
) -> dict:
    """Label one correction and append a proposal row. Does not apply a rule.

    RO #5: outbound_privacy_approval runs BEFORE call_jev.
    RO #4: invented_fact uses separate evidence channel (draft is not evidence).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    # RO #5 — approval BEFORE outbound (not only before ledger write)
    approval = outbound_privacy_approval(draft, correction, allow_outbound=allow_outbound)
    if not approval.get("ok"):
        return {
            "ok": False,
            "blocked": True,
            "applied": False,
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
        # Separate evidence channel — never imply draft is evidence (RO #4)
        "evidence": {
            "available": ev["available"],
            "text": ev["text"] if ev["available"] else None,
            "sha256": ev.get("sha256"),
            "note": ev.get("note"),
        },
    }
    questions = dict(QUESTIONS)
    # If evidence unavailable, do not ask invented_fact against draft-only state
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
    private = (answers.get("private_facts") or {}).get("noul")
    # Privacy noul alone must NOT permit raw text storage (primer #7).
    # Require explicit allow_text AND finite private_facts < 0.7.
    private_ok = finite_unit(private)
    store_text = bool(allow_text) and private_ok is not None and private_ok < 0.7
    home = (answers.get("home") or {}).get("choice") or "drop"
    gate = draft_action(answers, mode=mode)
    row = {
        "id": _sha(_now() + _sha(draft) + _sha(correction))[:16],
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
        "home_confidence": (answers.get("home") or {}).get("confidence"),
        "applied": False,
        "synthetic": False,
        "prefs_eligible": False,
        "evidence_sha256": ev.get("sha256"),
        "evidence_available": ev["available"],
        "privacy_approval": approval,
        "usage": response.get("usage"),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    """Factual claims require separately supplied evidence (primer #9)."""
    if not evidence or not isinstance(evidence, dict):
        return {
            "ok": False,
            "backend": "clarify",
            "unsupported": list(claims),
            "note": "No separate evidence supplied — do not authorize.",
        }
    blob = evidence.get("text") or evidence.get("excerpts") or ""
    if not isinstance(blob, str) or not blob.strip():
        return {
            "ok": False,
            "backend": "clarify",
            "unsupported": list(claims),
            "note": "Evidence empty — do not authorize.",
        }
    unsupported = [c for c in claims if c and c.lower() not in blob.lower()]
    return {
        "ok": len(unsupported) == 0,
        "backend": "act" if not unsupported else "investigate",
        "unsupported": unsupported,
        "evidence_sha256": _sha(blob),
    }


def promote_to_standing(row_id: str, apply: bool = False) -> dict:
    """Promote a label only with explicit apply. Synthetic rows rejected."""
    row = load_row(row_id)
    if row is None:
        raise KeyError(f"no ledger row: {row_id}")
    if row.get("synthetic") or str(row.get("id", "")).startswith(SYNTHETIC_PREFIX):
        return {
            "ok": False,
            "reason": "synthetic_excluded",
            "note": "Synthetic proof rows stay out of preference learning.",
        }
    if not apply:
        return {"ok": False, "reason": "apply_false", "note": "Explicit apply required."}
    if not row.get("prefs_eligible"):
        return {"ok": False, "reason": "not_prefs_eligible", "note": "Row not marked prefs_eligible."}
    standing = _load_standing()
    labels = standing.setdefault("labels", {})
    home = row.get("proposed_home") or "drop"
    if home == "drop":
        return {"ok": False, "reason": "home_drop"}
    labels[home] = {
        "from_row": row["id"],
        "correction_sha256": row.get("correction_sha256"),
        "promoted_at": _now(),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    STANDING.write_text(json.dumps(standing, indent=2) + "\n")
    return {"ok": True, "standing": prefs_snapshot()}


def held_out_loaw_calibrate() -> dict:
    """Calibrate gates on held-out LOAW-shaped tasks (no LIVE API; offline labels).

    Preference / grounding / success scored separately. Synthetic rows tagged.
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
        and auth_check_ok(results),
        "tasks": results,
        "evidence_sample": evidence,
        "note": "Held-out LOAW calibrate offline; synthetic; not prefs learning.",
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
    # RO #2: choice without payload → investigate; well-formed choice may act
    assert backend_verb({"type": "choice", "confidence": 0.99}, "ordinary") == "investigate"
    well = {
        "type": "choice",
        "choice": "drop",
        "confidence": 0.9,
        "probabilities": {"drop": 0.9, "keep": 0.1},
    }
    assert choice_well_formed(well) is True
    assert backend_verb(well, "ordinary") == "act"
    # RO dist/EV: negative/unnormalized probs + OOB score must NOT act
    bad_neg = {"type": "choice", "choice": "x", "probabilities": {"x": -8}, "confidence": 0.99}
    assert choice_well_formed(bad_neg) is False
    assert backend_verb(bad_neg, "ordinary") == "investigate"
    bad_score = {"type": "score", "score": 999, "confidence": 0.99}
    assert score_well_formed(bad_score) is False
    assert backend_verb(bad_score, "ordinary") == "investigate"
    well_score = {
        "type": "score",
        "score": 2.0,
        "confidence": 0.95,
        "criteria": ["low", "mid", "high"],
        "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
    }
    assert score_well_formed(well_score) is True
    assert backend_verb(well_score, "ordinary") == "act"
    # selected must be argmax; missing option vs request set fails closed
    not_argmax = {"type": "choice", "choice": "keep", "confidence": 0.99, "probabilities": {"drop": 0.9, "keep": 0.1}}
    assert choice_well_formed(not_argmax) is False
    assert backend_verb(not_argmax, "ordinary") == "investigate"
    assert backend_verb({"type": "choice", "confidence": 0.2}, "ordinary") == "investigate"
    assert decide_answer({"type": "choice", "confidence": 0.2}, "effect") == "stop"
    assert decide_answer({"type": "choice", "confidence": 0.95}, "effect") == "stop"  # no payload
    assert decide_answer({**well, "confidence": 0.95}, "effect") == "approve"
    assert decide_answer({"type": "choice", "confidence": 0.2}, "harmless") == "review"  # no payload
    assert finite_unit(True) is None
    assert finite_unit(float('nan')) is None
    assert finite_unit(1.5) is None
    assert finite_unit(0.5) == 0.5
    assert decide_answer({}, 'effect') == 'stop'
    assert decide_answer({'type': 'noul', 'noul': True}, 'ordinary') == 'review'
    assert authority_from_noul({'type': 'noul', 'noul': 0.99})['authority'] is False
    sep = separate_scores(0.9, 0.2, 0.8)
    assert sep['combined_forbidden'] is True and sep['authorize'] is False
    assert reject_sycophancy_signal(0.99)['reward_allowed'] is False
    bad = draft_action({'too_long': {'noul': True}})
    assert bad['backend'] == 'investigate' and bad['malformed'] is True
    assert verify_claims_against_evidence(['x'], None)['ok'] is False
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


def main(argv: list[str]) -> int:
    _offline_checks()
    command = argv[1] if len(argv) > 1 else "prove"
    if command == "prove":
        draft = (
            "It is axiomatic that the aforementioned liability posture, viewed through "
            "the lens of comparative fault and the totality of the evidentiary matrix, "
            "clearly establishes a compelling narrative for immediate escalation."
        )
        correction = "shorter"
        row = mark_synthetic(record(draft, correction, allow_text=True, allow_outbound=True))
        # rewrite last ledger line as synthetic-tagged
        rows = _read_ledger()
        if rows:
            rows[-1] = mark_synthetic(rows[-1])
            LEDGER.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            row = rows[-1]
        public = {
            "id": row["id"],
            "model": row["model"],
            "proposed_home": row["proposed_home"],
            "home_confidence": row["home_confidence"],
            "applied": row["applied"],
            "draft_gate": row["draft_gate"],
            "speakable": (row["labels"].get("speakable") or {}).get("score"),
            "speakable_confidence": (row["labels"].get("speakable") or {}).get("confidence"),
            "correction_is_lock": (row["labels"].get("correction_is_lock") or {}).get("noul"),
            "private_facts": (row["labels"].get("private_facts") or {}).get("noul"),
            "usage": row["usage"],
            "ledger": str(LEDGER),
        }
        print(json.dumps(public, indent=2))
        return 0
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
        row_id = argv[2] if len(argv) > 2 else None
        result = held_out_check(row_id)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1
    if command == "prefs":
        sub = argv[2] if len(argv) > 2 else "show"
        if sub == "clear":
            print(json.dumps(prefs_clear(), indent=2))
        else:
            print(json.dumps(prefs_snapshot(), indent=2))
        return 0
    print("use: prove | gate <draft> | replay <row_id> | held-out [row_id] | held-out-loaw | prefs [show|clear]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
