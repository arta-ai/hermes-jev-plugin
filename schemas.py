"""Tool schemas — what the LLM sees."""

EVALUATE = {
    "name": "jev_evaluate",
    "description": (
        "Ask TypeSafe Jev (a 'System One' decision model, NOT a text model) one or "
        "more typed questions against a state, and get structured answers with "
        "calibrated probabilities. Question types: 'noul' (yes/no -> yes_probability), "
        "'choice' (pick one option -> choice + probabilities), 'score' (rate on an "
        "ordered rubric -> numeric score + legend). All questions are evaluated in "
        "parallel against the same state in ONE call; asking several is cheap. Keep "
        "each question atomic (a gut-check judgment, not multi-step reasoning) — "
        "decompose complex judgments into several atomic questions and combine the "
        "answers yourself. Use for triage, routing, failure classification, review "
        "gating, and go/no-go checks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The context to evaluate: pasted text, error output, "
                "diff summary, code snippet, or a compact description of the situation.",
            },
            "questions": {
                "type": "object",
                "description": "Map of question_name -> spec. Spec fields: type "
                "('noul'|'choice'|'score'), instructions (the atomic question), and "
                "type-specific criteria. Examples: "
                '{"transient":{"type":"noul","instructions":"Is this failure transient '
                '(retrying could help)?"}} ; '
                '{"team":{"type":"choice","instructions":"Which team should own this",'
                '"criteria":{"backend":"...","frontend":"...","infra":"..."}}} ; '
                '{"severity":{"type":"score","instructions":"How severe is this bug",'
                '"criteria":["Cosmetic","Minor","Major","Blocks release"]}}',
            },
            "model": {
                "type": "string",
                "description": "Optional TypeSafe model id (default jev-latest).",
            },
        },
        "required": ["state", "questions"],
    },
}

CHECK = {
    "name": "jev_check",
    "description": (
        "Ask Jev a single yes/no (noul) question about a state and get the "
        "probability that the answer is yes (0-1). Use for atomic binary judgments "
        "about code, errors, logs, or plans: 'Is this stack trace a missing-dependency "
        "error?', 'Is this failure transient (a retry could fix it)?', 'Does this diff "
        "touch authentication logic?', 'Is this migration backwards-compatible?'. One "
        "question per call — make several calls (or use jev_evaluate) for multiple "
        "judgments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The context to evaluate.",
            },
            "question": {
                "type": "string",
                "description": "A single, atomic yes/no question.",
            },
            "criterion": {
                "type": "object",
                "description": "Optional structured criteria the model should check "
                "(JSON object of named requirements).",
            },
            "model": {
                "type": "string",
                "description": "Optional TypeSafe model id (default jev-latest).",
            },
        },
        "required": ["state", "question"],
    },
}

ROUTE = {
    "name": "jev_route",
    "description": (
        "Ask Jev to choose exactly one option from a defined set for a given state "
        "(a Choice question). Use for routing decisions: which subsystem a bug "
        "belongs to, which agent should handle a task, which fix strategy to apply, "
        "which test suite to run first. Returns the chosen option, per-option "
        "probabilities, and confidence. Options must be mutually exclusive and "
        "exhaustive enough to cover the input."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The context to evaluate.",
            },
            "question": {
                "type": "string",
                "description": "What to choose, e.g. 'Which component owns this failure?'",
            },
            "options": {
                "type": "object",
                "description": "Map of option_key -> short description. Example: "
                '{"auth":"login/session issues","db":"queries or migrations",'
                '"network":"timeouts and connectivity"}',
            },
            "model": {
                "type": "string",
                "description": "Optional TypeSafe model id (default jev-latest).",
            },
        },
        "required": ["state", "question", "options"],
    },
}

SCORE = {
    "name": "jev_score",
    "description": (
        "Ask Jev to rate a state against an ordered rubric (a Score question) and "
        "get a numeric score with per-level legend and confidence. Use for "
        "graduated judgments: severity of a review finding, code-review blocking-ness, "
        "task complexity, riskiness of a migration, confidence in a plan. Levels are "
        "ordered lowest to highest; 3-5 concrete levels work best."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The context to evaluate.",
            },
            "question": {
                "type": "string",
                "description": "What to score, e.g. 'How severe is this review finding?'",
            },
            "levels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered level descriptions, lowest first. Example: "
                '["Cosmetic","Minor","Major","Blocks release"]',
            },
            "model": {
                "type": "string",
                "description": "Optional TypeSafe model id (default jev-latest).",
            },
        },
        "required": ["state", "question", "levels"],
    },
}

SCHEMAS = [EVALUATE, CHECK, ROUTE, SCORE]
