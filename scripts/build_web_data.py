#!/usr/bin/env python3
"""Generate web/data.json: every note run through the loop, ready to display.

The page is static and has no server, so the run happens here and the result is
baked in. Everything the page shows comes from the same code paths eval.py
measures, so the screen cannot show a number the evaluation disagrees with.

    python scripts/build_web_data.py
    python scripts/build_web_data.py --simplifier lossy   demo the repair round
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.diff import aggregate  # noqa: E402
from core.loop import run  # noqa: E402
from core.simplify import SIMPLIFIERS  # noqa: E402
from core.text import note_id, strip_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "data" / "notes"
OUT = ROOT / "web" / "data.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplifier", choices=tuple(SIMPLIFIERS), default="glossary")
    parser.add_argument("--target-grade", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    simplifier = SIMPLIFIERS[args.simplifier]()
    notes, results = [], []
    for path in sorted(NOTES.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        nid = note_id(text) or path.stem
        outcome = run(text, simplifier, note_id=nid, target_grade=args.target_grade)
        best = outcome.best
        results.append(best.result)
        notes.append({
            "id": nid,
            "original": strip_metadata(text).strip(),
            "rewrite": strip_metadata(best.text).strip(),
            "rounds": [
                {"round": r.index, "fidelity": round(r.result.fidelity, 4),
                 "grade": r.readability.grade,
                 "unexplained": r.readability.unexplained_count,
                 "errors": len(r.result.errors),
                 "constraints_given": r.constraints_given}
                for r in outcome.rounds],
            "best_round": best.index,
            "converged": outcome.converged,
            "facts": [f.as_dict() for f in outcome.original_facts.facts()],
            "rewrite_facts": [f.as_dict() for f in best.facts.facts()],
            "diff": best.result.as_dict(),
            "before": outcome.original_readability.as_dict(),
            "after": best.readability.as_dict(),
        })

    payload = {"simplifier": args.simplifier, "target_grade": args.target_grade,
               "summary": aggregate(results), "notes": notes}
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    size = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size:.0f} KB, {len(notes)} notes, "
          f"{payload['summary']['facts']} facts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
