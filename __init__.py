"""Jev plugin — TypeSafe System One questions for Hermes Agent.

Registers TypeSafe Jev evaluator tools plus a read-only drafting standing consumer:
  jev_evaluate — many mixed questions in one call
  jev_check    — single yes/no (noul)
  jev_route    — pick one option (choice)
  jev_score    — ordered rubric rating (score)
  jev_standing_instruction — read-only mode-filtered standing for drafting (not gate_draft)
"""

from __future__ import annotations

import logging

from . import client, tools
from .schemas import CHECK, EVALUATE, ROUTE, SCORE, STANDING_INSTRUCTION

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Called exactly once by Hermes at startup."""
    common = {"toolset": "jev"}
    ctx.register_tool(name="jev_evaluate", schema=EVALUATE,
                      handler=lambda args, **kw: tools.evaluate(args, ctx=ctx, **kw), **common)
    ctx.register_tool(name="jev_check", schema=CHECK,
                      handler=lambda args, **kw: tools.check(args, ctx=ctx, **kw), **common)
    ctx.register_tool(name="jev_route", schema=ROUTE,
                      handler=lambda args, **kw: tools.route(args, ctx=ctx, **kw), **common)
    ctx.register_tool(name="jev_score", schema=SCORE,
                      handler=lambda args, **kw: tools.score(args, ctx=ctx, **kw), **common)
    ctx.register_tool(name="jev_standing_instruction", schema=STANDING_INSTRUCTION,
                      handler=lambda args, **kw: tools.standing_instruction(args, ctx=ctx, **kw), **common)

    # Bundle the usage skill so the agent can load it on demand.
    try:
        from pathlib import Path

        skills_dir = Path(__file__).parent / "skills"
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md)
    except Exception:  # noqa: BLE001 — skill bundling must never break plugin load
        logger.debug("jev: skill bundling skipped", exc_info=True)

    logger.info("jev: registered 5 tools (evaluate, check, route, score, standing_instruction)")
