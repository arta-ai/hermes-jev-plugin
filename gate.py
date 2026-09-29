"""Confidence gates for TypeSafe Choice, Score, and Noul answers.

Noul answers have no confidence field. Gate those on the yes-probability.
Choice and Score confidence is how peaked the option spread is. It is not
permission to act. Effect stakes never auto-act.

Backend verbs (Hermes OWN WIL-914): act | investigate | clarify.
These alias the internal gate outcomes; they do not grant mutate rights.

AI-primer / review-before-deploy guards (WIL-914): malformed or non-finite
inputs never authorize; Noul never supplies authority; preference fit is
scored apart from grounding and task success.

RO harden (WIL-914 side branch): unknown types fail-closed; choice/score
require full shape before act; missing required miss-labels ≠ clean pass.
Choice/score option keys and score legends bind ONLY to a trusted request
schema passed separately — never to answer-supplied keys. Choice act needs
full finite normalized distribution + selected=argmax over that request set.
Score act needs 2..10 legend/index bounds + EV-consistent score. Malformed
never throws.
"""

from __future__ import annotations

import math
from typing import Any

STAKES = ("harmless", "ordinary", "draft", "effect")
MODES = ("legal", "creative")
BACKEND_VERBS = ("act", "investigate", "clarify")
KNOWN_ANSWER_TYPES = ("noul", "choice", "score")
MISS_LABELS = (
    "too_long",
    "stacked_jargon",
    "mashed_jobs",
    "invented_fact",
    "wrong_audience",
    "unsourced_action",
    "hid_option",
)
# Required miss-label set for draft_action: absence ≠ clean pass (RO #3).
REQUIRED_MISS_LABELS = MISS_LABELS

_BACKEND_MAP = {
    "act": "act",
    "review": "investigate",
    "stop": "clarify",
    "approve": "clarify",
}

_DRAFT_BACKEND = {
    "pass": "act",
    "review": "investigate",
    "revise": "clarify",
}


def finite_unit(value: Any) -> float | None:
    """Accept only finite floats in [0, 1]. Reject bool, NaN, Inf, out-of-range."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number < 0.0 or number > 1.0:
            return None
        return number
    return None


# Bridge gap-close name (WIL-914) — same fail-closed unit interval.
_finite_unit_interval = finite_unit


def _confidence(answer: dict) -> float | None:
    if not isinstance(answer, dict):
        return None
    return finite_unit(answer.get("confidence"))


def _noul(answer: dict) -> float | None:
    if not isinstance(answer, dict):
        return None
    return finite_unit(answer.get("noul", answer.get("yes_probability")))


def _fail_closed(stakes: str) -> str:
    """Unknown / malformed / incomplete → never act."""
    return "stop" if stakes == "effect" else "review"


_NORM_TOL = 0.01
_EV_TOL = 0.02
SCORE_LEGEND_MIN = 2
SCORE_LEGEND_MAX = 10


def _finite_prob(value: Any) -> float | None:
    """Accept only finite floats in [0, 1]. Reject bool, NaN, Inf, out-of-range."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number < 0.0 or number > 1.0:
            return None
        return number
    return None


def _options_map(answer: dict) -> dict | None:
    """Return a well-formed options/criteria/probabilities map from an answer, or None.

    Used only as a distribution source (values). Option KEYS must still come
    from a trusted request schema — never from this map alone.
    """
    if not isinstance(answer, dict):
        return None
    for key in ("probabilities", "options", "choices"):
        value = answer.get(key)
        if isinstance(value, dict) and len(value) >= 1:
            if any(not isinstance(k, str) or not k.strip() for k in value.keys()):
                return None
            return value
    return None


def request_schema_choice_keys(request_schema: dict | None) -> list[str] | None:
    """Extract choice option keys ONLY from a trusted request schema.

    Never reads the answer. Accepts criteria dict, options dict/list, or
    explicit request_options / expected_options on the *request* object.
    """
    if not isinstance(request_schema, dict):
        return None
    for key in ("request_options", "expected_options"):
        raw = request_schema.get(key)
        if isinstance(raw, (list, tuple)):
            keys = [k for k in raw if isinstance(k, str) and k.strip()]
            if len(keys) == len(raw) and len(keys) >= 1:
                return list(keys)
        if isinstance(raw, dict) and raw:
            keys = [k for k in raw.keys() if isinstance(k, str) and k.strip()]
            if len(keys) == len(raw) and len(keys) >= 1:
                return keys
    criteria = request_schema.get("criteria")
    if isinstance(criteria, dict) and criteria:
        keys = [k for k in criteria.keys() if isinstance(k, str) and k.strip()]
        if len(keys) == len(criteria) and len(keys) >= 1:
            return keys
    options = request_schema.get("options")
    if isinstance(options, (list, tuple)):
        keys = [k for k in options if isinstance(k, str) and k.strip()]
        if len(keys) == len(options) and len(keys) >= 1:
            return list(keys)
    if isinstance(options, dict) and options:
        keys = [k for k in options.keys() if isinstance(k, str) and k.strip()]
        if len(keys) == len(options) and len(keys) >= 1:
            return keys
    return None


def request_schema_score_levels(request_schema: dict | None) -> list[str] | None:
    """Extract ordered score legend (len 2..10) ONLY from trusted request schema."""
    if not isinstance(request_schema, dict):
        return None
    criteria = request_schema.get("criteria")
    if criteria is None:
        criteria = request_schema.get("levels")
    if isinstance(criteria, list):
        if not (SCORE_LEGEND_MIN <= len(criteria) <= SCORE_LEGEND_MAX):
            return None
        labels: list[str] = []
        for item in criteria:
            if isinstance(item, bool) or item is None:
                return None
            if isinstance(item, str) and item.strip():
                labels.append(item.strip())
            else:
                return None
        return labels
    legend = request_schema.get("legend")
    if isinstance(legend, dict) and legend:
        n = len(legend)
        if not (SCORE_LEGEND_MIN <= n <= SCORE_LEGEND_MAX):
            return None
        labels = []
        for i in range(n):
            key = str(i)
            if key not in legend:
                return None
            val = legend[key]
            if not isinstance(val, str) or not val.strip():
                return None
            labels.append(val.strip())
        if any(k not in {str(i) for i in range(n)} for k in legend.keys()):
            return None
        return labels
    return None


# Back-compat aliases — both require a trusted request; answer-only paths removed.
def _expected_choice_keys(request_schema: dict | None) -> list[str] | None:
    return request_schema_choice_keys(request_schema)


def _score_legend_levels(request_schema: dict | None) -> list[str] | None:
    return request_schema_score_levels(request_schema)


def _normalized_distribution(raw: Any, expected_keys: list[str]) -> dict[str, float] | None:
    """Full finite normalized distribution over exactly expected_keys, or None."""
    if not isinstance(raw, dict) or not expected_keys:
        return None
    if len(raw) != len(expected_keys):
        return None
    if any(k not in raw for k in expected_keys):
        return None
    if any(k not in expected_keys for k in raw.keys()):
        return None
    out: dict[str, float] = {}
    total = 0.0
    for key in expected_keys:
        prob = _finite_prob(raw[key])
        if prob is None:
            return None
        out[key] = prob
        total += prob
    if abs(total - 1.0) > _NORM_TOL:
        return None
    return out


def choice_well_formed(answer: dict, request_schema: dict | None = None) -> bool:
    """Choice requires selected = argmax over a full finite normalized distribution.

    Option keys bind ONLY to the trusted request schema. Answer-supplied
    request_options / probabilities keys are NEVER the option set (RO bind).
    Without a trusted request schema → False (fail-closed). Malformed never throws.
    """
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return False
    expected = request_schema_choice_keys(request_schema)
    if expected is None:
        return False
    choice = answer.get("choice")
    if choice is None or isinstance(choice, (bool, list, tuple, dict)):
        return False
    if not isinstance(choice, str) or not choice.strip():
        if not isinstance(choice, (int, float)) or isinstance(choice, bool):
            return False
        choice = str(choice)
    dist_raw = answer.get("probabilities")
    if not isinstance(dist_raw, dict):
        for key in ("options", "choices"):
            candidate = answer.get(key)
            if isinstance(candidate, dict):
                dist_raw = candidate
                break
    dist = _normalized_distribution(dist_raw, expected)
    if dist is None:
        return False
    if choice not in dist:
        return False
    peak = max(dist.values())
    if dist[choice] < peak:
        return False
    return True


def score_well_formed(answer: dict, request_schema: dict | None = None) -> bool:
    """Score requires 2..10 legend/index bounds + EV-consistent finite score.

    Legend/levels bind ONLY to the trusted request schema — never answer
    criteria/legend alone. Mere presence of a finite score is NOT enough.
    Without a trusted request schema → False.
    """
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return False
    levels = request_schema_score_levels(request_schema)
    if levels is None:
        return False
    n = len(levels)
    index_keys = [str(i) for i in range(n)]
    probs_raw = answer.get("probabilities")
    dist = _normalized_distribution(probs_raw, index_keys)
    if dist is None:
        return False
    score = answer.get("score")
    if isinstance(score, bool) or score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    score_f = float(score)
    if not math.isfinite(score_f):
        return False
    if score_f < 0.0 or score_f > float(n - 1):
        return False
    expected = sum(int(k) * dist[k] for k in index_keys)
    if abs(score_f - expected) > _EV_TOL:
        return False
    return True


def decide_answer(
    answer: dict,
    stakes: str = "ordinary",
    request_schema: dict | None = None,
) -> str:
    """Return act, review, stop, or approve for one answer.

    Unavailable / malformed / unknown-type answers → review (ordinary) or
    stop (effect). Never silently authorize. High confidence on an unknown
    type is NOT permission to act (RO #1). Choice/score require full shape
    bound to trusted request schema before confidence can authorize (RO #2).
    """
    if stakes not in STAKES:
        raise ValueError(f"unknown stakes: {stakes}")
    if not isinstance(answer, dict) or not answer:
        return _fail_closed(stakes)
    kind = answer.get("type")
    if kind == "noul":
        probability = _noul(answer)
        if probability is None:
            return _fail_closed(stakes)
        if stakes == "effect":
            return "approve"
        if stakes == "harmless":
            return "act"
        if probability <= 0.3 or probability >= 0.7:
            return "act"
        return "review"

    if kind not in ("choice", "score"):
        return _fail_closed(stakes)

    if kind == "choice" and not choice_well_formed(answer, request_schema):
        return _fail_closed(stakes)
    if kind == "score" and not score_well_formed(answer, request_schema):
        return _fail_closed(stakes)

    confidence = _confidence(answer)
    if stakes == "harmless":
        return "act" if confidence is not None else "review"
    if stakes == "effect":
        if confidence is None or confidence < 0.5:
            return "stop"
        return "approve"
    if confidence is None or confidence < 0.5:
        return "review"
    return "act"


def backend_verb(
    answer: dict,
    stakes: str = "ordinary",
    request_schema: dict | None = None,
) -> str:
    """Map one answer to act | investigate | clarify. Never auto-mutates."""
    return _BACKEND_MAP[decide_answer(answer, stakes, request_schema)]


def validate_answer_against_request(
    answer: dict | None,
    request_schema: dict,
) -> dict[str, Any]:
    """Request-bound validation result for one answer. Never throws."""
    if not isinstance(request_schema, dict):
        return {"ok": False, "reason": "missing_request_schema", "backend": "investigate"}
    qtype = request_schema.get("type")
    if qtype not in KNOWN_ANSWER_TYPES:
        return {"ok": False, "reason": "unknown_request_type", "backend": "investigate"}
    if not isinstance(answer, dict):
        return {"ok": False, "reason": "missing_answer", "backend": "investigate"}
    if answer.get("type") != qtype:
        return {"ok": False, "reason": "type_mismatch", "backend": "investigate"}
    if qtype == "noul":
        if _noul(answer) is None and not answer.get("unavailable"):
            return {"ok": False, "reason": "malformed_noul", "backend": "investigate"}
        return {"ok": True, "reason": "noul_ok", "backend": backend_verb(answer, "ordinary", request_schema)}
    if qtype == "choice":
        if not choice_well_formed(answer, request_schema):
            return {"ok": False, "reason": "choice_not_request_bound", "backend": "investigate"}
        return {"ok": True, "reason": "choice_ok", "backend": backend_verb(answer, "ordinary", request_schema)}
    if not score_well_formed(answer, request_schema):
        return {"ok": False, "reason": "score_not_request_bound", "backend": "investigate"}
    return {"ok": True, "reason": "score_ok", "backend": backend_verb(answer, "ordinary", request_schema)}


def validate_answers_against_request(
    answers: dict | None,
    questions: dict,
) -> dict[str, Any]:
    """Validate a full answers map against trusted request questions.

    Returns {ok, failures, backends}. Does not throw. Extra answer keys that
    were not in the request are recorded as failures (not silently trusted).
    """
    if not isinstance(questions, dict) or not questions:
        return {"ok": False, "failures": {"_request": "empty_questions"}, "backends": {}}
    if not isinstance(answers, dict):
        return {
            "ok": False,
            "failures": {name: "missing_answer" for name in questions},
            "backends": {name: "investigate" for name in questions},
        }
    failures: dict[str, str] = {}
    backends: dict[str, str] = {}
    for name, schema in questions.items():
        result = validate_answer_against_request(answers.get(name), schema)
        backends[name] = result["backend"]
        if not result["ok"]:
            failures[name] = result["reason"]
    extras = [k for k in answers.keys() if k not in questions]
    for extra in extras:
        failures[extra] = "unexpected_answer_key"
        backends[extra] = "investigate"
    return {"ok": len(failures) == 0, "failures": failures, "backends": backends}


def authority_from_noul(answer: dict) -> dict[str, Any]:
    """Noul never grants authority. At most: suggest_clarify."""
    probability = _noul(answer) if isinstance(answer, dict) else None
    return {
        "authority": False,
        "backend": "clarify" if probability is not None else "investigate",
        "noul": probability,
        "note": "Noul cannot supply approval authority.",
    }


def draft_action(answers: dict, mode: str = "legal") -> dict[str, Any]:
    """Turn miss-label Nouls into pass, review, or revise + backend verb.

    mode=legal: stacked_jargon and unsourced_action stay in the chip set.
    mode=creative: stacked_jargon ignored for revise; invented_fact /
    unsourced_action still count.
    Malformed chips are skipped and force investigate (not act).

    RO #3: REQUIRED_MISS_LABELS must all be present and well-formed for a
    clean pass. Missing labels ≠ clean pass — fail-closed to review/investigate.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if not isinstance(answers, dict):
        return {
            "action": "review",
            "backend": "investigate",
            "mode": mode,
            "revise": [],
            "review": ["malformed_answers"],
            "chips": [],
            "malformed": True,
            "missing_required": list(REQUIRED_MISS_LABELS),
        }
    reasons: list[str] = []
    uncertain: list[str] = []
    chips: list[dict[str, Any]] = []
    malformed = False
    missing_required: list[str] = []
    skip_revise = {"stacked_jargon"} if mode == "creative" else set()
    for name in MISS_LABELS:
        answer = answers.get(name)
        if answer is None:
            missing_required.append(name)
            continue
        if not isinstance(answer, dict):
            malformed = True
            missing_required.append(name)  # present but unusable counts against completeness
            continue
        # Explicit unavailable marker (primer #9 / invented_fact) — not a clear pass
        if answer.get("unavailable") is True:
            malformed = True
            chips.append(
                {
                    "id": name,
                    "noul": None,
                    "gate": "review",
                    "backend": "investigate",
                    "unavailable": True,
                }
            )
            continue
        probability = _noul(answer)
        if probability is None:
            # bool / non-finite / OOR / missing → do not authorize
            malformed = True
            chips.append({"id": name, "noul": None, "gate": "review", "backend": "investigate", "rejected": True})
            continue
        gate = decide_answer({"type": "noul", "noul": probability})
        chips.append(
            {
                "id": name,
                "noul": round(probability, 4),
                "gate": gate,
                "backend": _BACKEND_MAP[gate],
            }
        )
        if name in skip_revise:
            continue
        if probability >= 0.7:
            reasons.append(name)
        elif probability > 0.3:
            uncertain.append(name)

    review_tags = list(uncertain)
    if malformed:
        review_tags.append("malformed_inputs")
    if missing_required:
        review_tags.append("missing_required_labels")

    if reasons:
        action = "revise"
    elif uncertain or malformed or missing_required:
        # RO #3: missing required labels block pass/act
        action = "review"
    elif chips and not missing_required:
        action = "pass"
    else:
        action = "review"
    return {
        "action": action,
        "backend": _DRAFT_BACKEND[action],
        "mode": mode,
        "revise": reasons,
        "review": review_tags,
        "chips": chips,
        "malformed": malformed,
        "missing_required": missing_required,
    }


def separate_scores(
    preference_fit: Any,
    factual_grounding: Any,
    task_success: Any,
) -> dict[str, Any]:
    """Score preference fit SEPARATELY from grounding and task success.

    Does not average them into one reward. Sycophancy / agreement is not a
    substitute for any of the three.
    """
    pref = finite_unit(preference_fit)
    ground = finite_unit(factual_grounding)
    success = finite_unit(task_success)
    return {
        "preference_fit": pref,
        "factual_grounding": ground,
        "task_success": success,
        "combined_forbidden": True,
        "authorize": False,
        "usable": all(v is not None for v in (pref, ground, success)),
    }


def reject_sycophancy_signal(agreement_with_user: Any) -> dict[str, Any]:
    """Do not reward agreement / sycophancy as preference learning signal."""
    value = finite_unit(agreement_with_user)
    return {
        "agreement": value,
        "reward_allowed": False,
        "note": "Agreement alone must not update prefs or authorize.",
    }
