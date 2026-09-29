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
Choice act requires full finite normalized distribution + selected=argmax
bound to request option set. Score act requires 2..10 legend/index bounds
+ EV-consistent score (not mere finite presence). Malformed never throws.
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
    """Return a well-formed options/criteria/probabilities map, or None."""
    for key in ("options", "choices", "criteria", "probabilities"):
        value = answer.get(key)
        if isinstance(value, dict) and len(value) >= 1:
            # reject empty-string keys / non-sensible maps
            if any(not isinstance(k, str) or not k.strip() for k in value.keys()):
                return None
            return value
    return None


def _expected_choice_keys(answer: dict) -> list[str] | None:
    """Bind choice validation to the request option set when present.

    Prefer explicit request_options / options-as-list / criteria-dict keys.
    Fall back to probabilities/options map keys. Never invent options.
    """
    if not isinstance(answer, dict):
        return None
    for key in ("request_options", "expected_options"):
        raw = answer.get(key)
        if isinstance(raw, (list, tuple)):
            keys = [k for k in raw if isinstance(k, str) and k.strip()]
            if len(keys) == len(raw) and len(keys) >= 1:
                return list(keys)
        if isinstance(raw, dict) and raw:
            keys = [k for k in raw.keys() if isinstance(k, str) and k.strip()]
            if len(keys) == len(raw):
                return keys
    options = answer.get("options")
    if isinstance(options, (list, tuple)):
        keys = [k for k in options if isinstance(k, str) and k.strip()]
        if len(keys) == len(options) and len(keys) >= 1:
            return list(keys)
    criteria = answer.get("criteria")
    if isinstance(criteria, dict) and criteria:
        keys = [k for k in criteria.keys() if isinstance(k, str) and k.strip()]
        if len(keys) == len(criteria):
            return keys
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        keys = [k for k in probs.keys() if isinstance(k, str) and k.strip()]
        if len(keys) == len(probs):
            return keys
    # last resort: options/choices dict maps
    for key in ("options", "choices"):
        value = answer.get(key)
        if isinstance(value, dict) and value:
            keys = [k for k in value.keys() if isinstance(k, str) and k.strip()]
            if len(keys) == len(value):
                return keys
    return None


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


def choice_well_formed(answer: dict) -> bool:
    """Choice requires selected = argmax over a full finite normalized distribution.

    Option keys must match the expected request option set when supplied.
    Malformed list/non-string choice → False (never throw).
    """
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return False
    choice = answer.get("choice")
    # reject missing / blank / non-scalar choice (list/dict/bool)
    if choice is None or isinstance(choice, (bool, list, tuple, dict)):
        return False
    if not isinstance(choice, str) or not choice.strip():
        # allow non-str scalars only if they appear as option keys after str()? no — keys are str
        if not isinstance(choice, (int, float)) or isinstance(choice, bool):
            return False
        choice = str(choice)
    expected = _expected_choice_keys(answer)
    if expected is None:
        return False
    # probabilities preferred; else options/choices numeric maps
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
    # selected must be an argmax (ties allowed if selected is among max)
    if dist[choice] < peak:
        return False
    return True


def _score_legend_levels(answer: dict) -> list[str] | None:
    """Return ordered legend labels (len 2..10) or None.

    Accepts criteria list (request shape) or legend map keyed 0..n-1.
    Does not invent levels.
    """
    if not isinstance(answer, dict):
        return None
    criteria = answer.get("criteria")
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
    legend = answer.get("legend")
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


def score_well_formed(answer: dict) -> bool:
    """Score requires 2..10 legend/index bounds + EV-consistent finite score.

    Mere presence of a finite score is NOT enough (score=999 must fail-closed).
    Probabilities must be a full finite normalized distribution over 0..n-1.
    """
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return False
    levels = _score_legend_levels(answer)
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
    # index bounds: 0 .. n-1 (TypeSafe score = weighted mean of level indices)
    if score_f < 0.0 or score_f > float(n - 1):
        return False
    expected = sum(int(k) * dist[k] for k in index_keys)
    if abs(score_f - expected) > _EV_TOL:
        return False
    return True


def decide_answer(answer: dict, stakes: str = "ordinary") -> str:
    """Return act, review, stop, or approve for one answer.

    Unavailable / malformed / unknown-type answers → review (ordinary) or
    stop (effect). Never silently authorize. High confidence on an unknown
    type is NOT permission to act (RO #1). Choice/score require full shape
    before confidence can authorize (RO #2).
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
            # A Noul can only flag that approval may be needed — never authorize.
            return "approve"
        if stakes == "harmless":
            return "act"
        if probability <= 0.3 or probability >= 0.7:
            return "act"
        return "review"

    # RO #1: unknown / missing type → fail-closed (confidence irrelevant)
    if kind not in ("choice", "score"):
        return _fail_closed(stakes)

    # RO #2: full type/shape validation before act
    if kind == "choice" and not choice_well_formed(answer):
        return _fail_closed(stakes)
    if kind == "score" and not score_well_formed(answer):
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


def backend_verb(answer: dict, stakes: str = "ordinary") -> str:
    """Map one answer to act | investigate | clarify. Never auto-mutates."""
    return _BACKEND_MAP[decide_answer(answer, stakes)]


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
