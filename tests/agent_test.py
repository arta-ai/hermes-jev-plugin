#!/usr/bin/env python3
"""Agent-path integration tests for the jev Hermes plugin.

Drives real `hermes -z` sessions (fresh agent, jev toolset only) and asserts
on the JSON each tool returns through the full path a coding agent uses:
model -> tool registry -> plugin handler -> TypeSafe API.

Coverage:
  1. clear-cut noul    -> confident 'yes' (p >= 0.70)
  2. calibration noul  -> no extreme confidence on an uninformative state
  3. route             -> picks the owning subsystem, with probabilities
  4. evaluate (mixed)  -> noul + choice + score in ONE call, using the
                          'levels' alias for score questions
  5. malformed call    -> graceful error JSON, hermes exits 0 (loop intact)

Requirements: `hermes` on PATH, jev plugin enabled, TYPESAFE_API_KEY in
~/.hermes/.env (Hermes loads the key itself at runtime).

Stdlib only — run with any python3. Expect ~2-4 min (one agent session per
test). Exit codes: 0 all pass, 1 failures, 2 preflight failed. Assertions
allow small margins because jev's probabilities vary slightly run-to-run
(observed ±0.02 on identical input).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERMES = shutil.which("hermes") or "hermes"
ENV_FILE = Path.home() / ".hermes" / ".env"
PER_TEST_TIMEOUT = 240

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        failures.append(name)


def preflight() -> bool:
    ok = True
    r = subprocess.run([HERMES, "plugins", "list"], capture_output=True, text=True, timeout=60)
    listing = r.stdout + r.stderr
    line = next((l for l in listing.splitlines() if "jev" in l), "")
    enabled = "enabled" in line.lower()
    check("preflight: jev plugin enabled", enabled, line.strip()[:120])
    ok &= enabled

    key_line = False
    if ENV_FILE.exists():
        key_line = any(l.strip().startswith("TYPESAFE_API_KEY=") for l in ENV_FILE.read_text().splitlines())
    check("preflight: TYPESAFE_API_KEY in ~/.hermes/.env", key_line)
    ok &= key_line
    return ok


def json_candidates(text: str) -> list:
    """All balanced top-level-ish JSON objects found in agent output."""
    text = re.sub(r"```(?:json)?", "", text)
    found, i, n = [], 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start == -1:
            break
        depth, end = 0, None
        for j in range(start, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            i = start + 1
            continue
        try:
            found.append(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
        i = end + 1
    return found


def pick(cands: list, want_keys: tuple) -> dict | None:
    for c in cands:
        if isinstance(c, dict) and any(k in c for k in want_keys):
            return c
    return cands[0] if cands else None


def run_agent(prompt: str) -> tuple[int, str]:
    """Run a fresh hermes -z session restricted to the jev toolset."""
    result = None
    for toolset in ("jev", None):  # retry without -t if the flag is rejected
        args = [HERMES, "-z", prompt]
        if toolset:
            args += ["-t", toolset]
        result = subprocess.run(args, capture_output=True, text=True, timeout=PER_TEST_TIMEOUT)
        out = result.stdout + result.stderr
        if result.returncode == 0 and "unrecognized arguments" not in out:
            return result.returncode, out
    return result.returncode, result.stdout + result.stderr


def agent_test(name: str, prompt: str, want_keys: tuple, asserts) -> None:
    rc, raw = run_agent(prompt)
    cands = json_candidates(raw)
    data = pick(cands, want_keys)
    if data is None:
        check(f"{name}: agent returned tool JSON", False, (raw[-400:] if raw else "no output"))
        return
    asserts(name, data, rc, raw)


def main() -> int:
    t0 = time.time()
    print("== jev plugin: agent-path integration tests ==\n", flush=True)
    if not preflight():
        print("\nRESULT: preflight failed — fix the above and re-run.", flush=True)
        return 2
    print(flush=True)

    # 1) Clear-cut noul -> confident yes
    def t_clear(name, data, rc, raw):
        p = data.get("yes_probability")
        print(f"    observed: yes_probability={p} verdict={data.get('verdict')}", flush=True)
        check(f"{name}: numeric yes_probability", isinstance(p, (int, float)))
        check(f"{name}: confident yes (p >= 0.70)", isinstance(p, (int, float)) and p >= 0.70)
        check(f"{name}: verdict 'yes'", data.get("verdict") == "yes")

    agent_test(
        "clear-cut check",
        "Call the jev_check tool with state='The build failed with exit code 1 after a network "
        "timeout during dependency download' and question='Is this failure transient (a retry "
        "could succeed)?'. After the tool returns, output ONLY the JSON the tool returned, "
        "verbatim — no other text.",
        ("yes_probability",),
        t_clear,
    )

    # 2) Calibration -> no extreme confidence on uninformative state
    def t_calib(name, data, rc, raw):
        p = data.get("yes_probability")
        print(f"    observed: yes_probability={p} verdict={data.get('verdict')}", flush=True)
        check(
            f"{name}: not overconfident on uninformative state (0.15 < p < 0.85)",
            isinstance(p, (int, float)) and 0.15 < p < 0.85,
            f"p={p}",
        )

    agent_test(
        "calibration check",
        "Call the jev_check tool with state='The deploy went out at 14:00 on Tuesday. Nothing "
        "else is known about the service.' and question='Will the service stay up this "
        "weekend?'. After the tool returns, output ONLY the JSON the tool returned, verbatim "
        "— no other text.",
        ("yes_probability",),
        t_calib,
    )

    # 3) Route -> picks the owning subsystem
    def t_route(name, data, rc, raw):
        choice = data.get("choice")
        probs = data.get("probabilities") or {}
        conf = data.get("confidence")
        print(f"    observed: choice={choice} confidence={conf}", flush=True)
        check(f"{name}: chose db_layer", choice == "db_layer", f"got {choice!r}")
        check(f"{name}: db_layer in probabilities", "db_layer" in probs)
        check(f"{name}: confidence in [0,1]", isinstance(conf, (int, float)) and 0 <= conf <= 1)

    agent_test(
        "route",
        "Call the jev_route tool with state=\"ImportError: cannot import name 'Session' from "
        "'app.db.session' right after commit a1b2c3 renamed app/db/session.py to "
        "app/db/database.py; no other modules changed.\" question='Which component owns this "
        "failure?' options={'db_layer':'app/db module, session factory and imports',"
        "'auth':'login and token handling','deploy':'build, CI or container configuration'}. "
        "After the tool returns, output ONLY the JSON the tool returned, verbatim — no other text.",
        ("choice",),
        t_route,
    )

    # 4) Mixed evaluate in one call (score uses the 'levels' alias)
    questions = {
        "transient": {"type": "noul", "instructions": "Is this failure transient (retrying could succeed)?"},
        "owner": {
            "type": "choice",
            "instructions": "Which subsystem owns this failure?",
            "criteria": {
                "payments": "payment gateway or billing code",
                "db": "database queries or migrations",
                "infra": "deploy or config",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this failure?",
            "levels": ["Cosmetic", "Minor", "Major", "Blocks release"],
        },
    }

    def t_eval(name, data, rc, raw):
        ans = data.get("answers") or {}
        sev = (ans.get("severity") or {}).get("score")
        print(f"    observed: answers={sorted(ans)} severity={sev}", flush=True)
        check(f"{name}: all three answered", {"transient", "owner", "severity"} <= set(ans), f"got {sorted(ans)}")
        check(
            f"{name}: noul has yes_probability",
            isinstance((ans.get("transient") or {}).get("yes_probability"), (int, float)),
        )
        check(f"{name}: choice answered", bool((ans.get("owner") or {}).get("choice")))
        check(
            f"{name}: score within rubric range [0, 3]",
            isinstance(sev, (int, float)) and 0 <= sev <= 3,
            f"score={sev}",
        )

    agent_test(
        "mixed evaluate",
        "Call the jev_evaluate tool with state='Checkout returns HTTP 500 for card payments "
        "only; traceback shows TypeError inside the payment gateway client. A recent diff "
        "touched the retry decorator shared by all clients.' and questions="
        f"{json.dumps(questions)}. After the tool returns, output ONLY the JSON the tool "
        "returned, verbatim — no other text.",
        ("answers",),
        t_eval,
    )

    # 5) Malformed call -> graceful error JSON, loop intact
    def t_malformed(name, data, rc, raw):
        print(f"    observed: rc={rc} keys={sorted(data)}", flush=True)
        check(f"{name}: error JSON returned", "error" in data, str(data)[:200])
        check(f"{name}: hermes exited 0 (loop intact)", rc == 0, f"rc={rc}")

    agent_test(
        "malformed route",
        "Call the jev_route tool with state='x', question='Which one?', and options="
        "{'only':'a single option'}. After the tool returns, output ONLY the JSON the tool "
        "returned, verbatim — no other text.",
        ("error",),
        t_malformed,
    )

    total = len(failures)
    print(
        f"\nRESULT: {'ALL PASS' if total == 0 else f'{total} FAILURE(S): {failures}'}"
        f" — {time.time() - t0:.0f}s",
        flush=True,
    )
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
