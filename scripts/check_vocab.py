#!/usr/bin/env python3
"""Check that simplifying a term does not change what the extractor thinks it is.

This catches the subtlest bug in the project, and the one that cost the most
time to find by hand.

The extractor reads the original and the rewrite with the same code. If the
glossary rewrites a phrase the extractor recognises into a phrase it does not,
the fidelity score reports an information loss that did not happen. Three real
examples, each of which cost a debugging session:

  "seek immediate medical attention" -> "get medical help right away"
      the plain form was in no action table, so seven warning signs vanished
  "follow up with cardiology"        -> "see cardiology"
      the follow-up cue required "see your", so an appointment vanished
  "emesis"                           -> "vomiting"
      "emesis" was in no symptom table, so the rewrite looked like it had
      invented a warning sign

The invariant: for every substitution the glossary can make, if the source
phrase resolves to a canonical value in one of the extractor's tables, the
replacement must resolve to the same value. Not to no value, and not to a
different one.

    python scripts/check_vocab.py          report, exit 1 on any mismatch
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.extract import (_CONDITIONAL, _FOLLOWUP_CUE, _MED_CUE,  # noqa: E402
                          _NEGATION)
from core.glossary import (ABBREVIATIONS, ABBREVIATIONS_UPPER, GLOSSES,  # noqa: E402
                           PHRASES, WORDS)
from core.extract import _drop_subsumed  # noqa: E402
from core.vocab import (ACTIONS, ACTIVITIES, FOLLOWUP_ROLES, SYMPTOMS,  # noqa: E402
                        matching_keys)

TABLES = {
    "symptom": SYMPTOMS,
    "action": ACTIONS,
    "activity": ACTIVITIES,
    "follow-up role": FOLLOWUP_ROLES,
}

# The extractor's gating patterns, which decide whether a sentence is read as
# an instruction at all. These were outside the check at first, and the gap let
# one through: the glossary rewrites "indicated" as "needed", the follow-up cue
# matched only "is indicated in", and a scheduled chest x-ray vanished from
# every rewrite of the pneumonia note.
CUES = {
    "follow-up cue": _FOLLOWUP_CUE,
    "red-flag conditional": _CONDITIONAL,
    "medication cue": _MED_CUE,
    "negation": _NEGATION,
}

# Substitutions where source and replacement deliberately resolve differently,
# each with the reason it is acceptable.
ALLOWED = {
    # "deep breathing exercises" trips the `exercise` activity on the word
    # "exercises". An activity only becomes a restriction in a negated or
    # limited sentence, and no note restricts breathing exercises, so losing
    # the match cannot change a fact. Narrowing the `exercise` aliases to
    # exclude it would cost more real matches than it saves.
    ("deep breathing exercises", "slow deep breaths"),
}


def _resolve(phrase: str) -> dict[str, tuple[str, ...]]:
    """What the extractor's tables make of this phrase.

    Symptoms go through the same subsumption the extractor applies, so this
    checks what is actually read rather than what the raw table would match.
    """
    probe = f"please {phrase} now."
    out = {}
    for name, table in TABLES.items():
        keys = matching_keys(table, probe)
        out[name] = tuple(_drop_subsumed(keys) if name == "symptom" else keys)
    return out


def main() -> int:
    pairs: list[tuple[str, str, str]] = []
    for label, table in (("ABBREVIATIONS", ABBREVIATIONS),
                         ("ABBREVIATIONS_UPPER", ABBREVIATIONS_UPPER),
                         ("PHRASES", PHRASES),
                         ("WORDS", WORDS)):
        for source, replacement in table.items():
            if replacement:
                pairs.append((label, source, replacement))

    problems: list[str] = []
    checked = 0
    for label, source, replacement in pairs:
        if (source.lower(), replacement.lower()) in ALLOWED:
            continue
        before, after = _resolve(source), _resolve(replacement)
        for family in TABLES:
            was, now = before[family], after[family]
            if not was:
                continue                 # the source means nothing to the extractor
            checked += 1
            if set(was) == set(now):
                continue
            lost = set(was) - set(now)
            gained = set(now) - set(was)
            detail = []
            if lost:
                detail.append(f"loses {sorted(lost)}")
            if gained:
                detail.append(f"gains {sorted(gained)}")
            problems.append(
                f"  {label}: {source!r} -> {replacement!r}\n"
                f"    as a {family}: {', '.join(detail)}")

    # Same invariant for the gating patterns: a substitution inside a sentence
    # must not stop the extractor recognising the sentence as an instruction.
    for label, source, replacement in pairs:
        if (source.lower(), replacement.lower()) in ALLOWED:
            continue
        for family, pattern in CUES.items():
            before = bool(pattern.search(f"please {source} now."))
            after = bool(pattern.search(f"please {replacement} now."))
            if before and not after:
                checked += 1
                problems.append(
                    f"  {label}: {source!r} -> {replacement!r}\n"
                    f"    the {family} no longer fires, so the sentence stops "
                    f"being read as an instruction")
            elif before:
                checked += 1

    # The glosses are additive, so the only risk is a gloss that introduces a
    # canonical value the original text did not carry.
    for term, gloss in GLOSSES.items():
        for family, table in TABLES.items():
            gained = matching_keys(table, f"please {gloss} now.")
            if gained:
                problems.append(
                    f"  GLOSSES: {term!r} glossed as {gloss!r}\n"
                    f"    the gloss itself reads as a {family}: {gained}")

    print(f"checked {len(pairs)} substitutions "
          f"({checked} of them resolve to a canonical value)")
    if problems:
        print(f"\n{len(problems)} substitution(s) change what the extractor reads:\n")
        print("\n".join(problems))
        print("\nEach of these would be reported as an information loss that did"
              "\nnot happen. Fix by adding the replacement's wording to the"
              "\nrelevant table in core/vocab.py, or by changing the glossary.")
        return 1
    print("every substitution preserves the extractor's reading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
