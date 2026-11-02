#!/usr/bin/env python3
"""Everything this project claims, measured.

Three studies, in the order that matters:

1. `extraction`: the extractor against hand-labelled gold facts. This comes
   first because the other two are meaningless if the instrument is wrong. A
   fidelity score computed by a bad extractor is a number about the extractor.

2. `sensitivity`: the detector against deliberately injected errors. Ground
   truth here is known by construction: the mutation declares what it broke, so
   this measures detection rather than agreement. It also counts what the
   detector reported that nothing broke, because a detector that flags
   everything scores perfectly on detection alone.

3. `loop`: the full generate-verify-repair loop over the corpus: fidelity,
   reading level, unexplained-term count, and how many notes needed review.

Everything runs offline with no API key. `--simplifier claude` swaps the
offline substitution baseline for the model-backed one and is the only path
that needs credentials.

    python eval.py                      all three studies, offline
    python eval.py --study sensitivity  one study
    python eval.py --simplifier claude  model-backed loop study
    python eval.py --json results.json  machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from core.diff import aggregate, diff
from core.extract import extract
from core.loop import run
from core.mutate import MUTATION_NAMES, mutations_for
from core.readability import HAVE_WORDFREQ, assess, threshold_report
from core.schema import FactSet
from core.simplify import DEFAULT_MODEL, SIMPLIFIERS

# Options for the model-backed simplifier, set from the command line. The
# offline simplifiers take none.
SIMPLIFIER_OPTIONS: dict = {}


def build_simplifier(name: str):
    factory = SIMPLIFIERS[name]
    return factory(**SIMPLIFIER_OPTIONS) if name == "claude" else factory()
from core.text import note_id

ROOT = Path(__file__).resolve().parent
NOTES = ROOT / "data" / "notes"
GOLD = ROOT / "data" / "gold" / "gold.json"


def load_notes() -> list[tuple[str, str]]:
    out = []
    for path in sorted(NOTES.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        out.append((note_id(text) or path.stem, text))
    return out


def rule(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


# ---------------------------------------------------------------------------
# 1. extraction accuracy

def _gold_factset(payload: dict) -> FactSet:
    """Build a FactSet from gold labels, assigning follow-up ordinals the same
    way the extractor does so the two are comparable."""
    from core.extract import assign_ordinals
    clean = {}
    for family in ("medications", "red_flags", "restrictions"):
        rows = []
        for row in payload.get(family, []):
            rows.append({k: v for k, v in row.items()
                         if k not in ("comment", "span", "ordinal")})
        clean[family] = rows
    facts = FactSet.from_dict(clean)
    facts.follow_ups = assign_ordinals(
        [(row["who"], row.get("within_days"), "")
         for row in payload.get("follow_ups", [])])
    return facts


def _prf(true_positive: int, false_positive: int, false_negative: int) -> tuple:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def study_extraction(notes: list[tuple[str, str]], path: Path = GOLD,
                     label: str = "") -> dict:
    if not path.exists():
        print(rule("1. EXTRACTION ACCURACY"))
        print(f"{path} is missing, so this study cannot run.")
        return {"available": False}

    gold = json.loads(path.read_text(encoding="utf-8"))
    identity = Counter()        # tp / fp / fn on identity alone
    exact = Counter()           # of matched identities, how many values agree
    per_family: dict[str, Counter] = {}
    field_errors: Counter = Counter()
    examples: list[str] = []
    unclassifiable: list[str] = []

    for nid, text in notes:
        if nid not in gold:
            continue
        unclassifiable += [f"{nid}: {item}" for item in gold[nid].get("unclassifiable", [])]
        want = _gold_factset(gold[nid]).by_key()
        got = extract(text).by_key()
        for key in set(want) | set(got):
            family = key[0]
            counter = per_family.setdefault(family, Counter())
            if key in want and key in got:
                identity["tp"] += 1
                counter["tp"] += 1
                if want[key].value == got[key].value:
                    exact["match"] += 1
                    counter["value_match"] += 1
                else:
                    exact["differ"] += 1
                    for index, (a, b) in enumerate(zip(want[key].value, got[key].value)):
                        if a != b:
                            field_errors[f"{family}[{index}]"] += 1
                    if len(examples) < 14:
                        examples.append(
                            f"  {nid} {family} {key[1]}\n"
                            f"    gold      {want[key].value}\n"
                            f"    extracted {got[key].value}")
            elif key in want:
                identity["fn"] += 1
                counter["fn"] += 1
                if len(examples) < 14:
                    examples.append(f"  {nid} MISSED {family} {key[1]}")
            else:
                identity["fp"] += 1
                counter["fp"] += 1
                if len(examples) < 14:
                    examples.append(f"  {nid} SPURIOUS {family} {key[1]}")

    print(rule(f"1. EXTRACTION ACCURACY{label}"))
    print("Identity = did the extractor find a fact about this thing at all.")
    print("Value    = of the facts it found, how many agree field for field.\n")
    print(f"{'family':<14}{'gold':>6}{'found':>7}{'tp':>5}{'fp':>5}{'fn':>5}"
          f"{'prec':>7}{'rec':>7}{'F1':>7}{'exact':>8}")
    overall_rows = []
    for family in sorted(per_family):
        counter = per_family[family]
        precision, recall, f1 = _prf(counter["tp"], counter["fp"], counter["fn"])
        exact_rate = (counter["value_match"] / counter["tp"]) if counter["tp"] else 0.0
        print(f"{family:<14}{counter['tp'] + counter['fn']:>6}"
              f"{counter['tp'] + counter['fp']:>7}{counter['tp']:>5}"
              f"{counter['fp']:>5}{counter['fn']:>5}"
              f"{precision:>7.3f}{recall:>7.3f}{f1:>7.3f}{exact_rate:>8.3f}")
        overall_rows.append({"family": family, "precision": precision,
                             "recall": recall, "f1": f1,
                             "exact_value_rate": round(exact_rate, 4),
                             "tp": counter["tp"], "fp": counter["fp"],
                             "fn": counter["fn"]})
    precision, recall, f1 = _prf(identity["tp"], identity["fp"], identity["fn"])
    exact_rate = exact["match"] / identity["tp"] if identity["tp"] else 0.0
    print(f"{'ALL':<14}{identity['tp'] + identity['fn']:>6}"
          f"{identity['tp'] + identity['fp']:>7}{identity['tp']:>5}"
          f"{identity['fp']:>5}{identity['fn']:>5}"
          f"{precision:>7.3f}{recall:>7.3f}{f1:>7.3f}{exact_rate:>8.3f}")

    if field_errors:
        print("\nWhich fields disagree most (family[index]):")
        for name, count in field_errors.most_common(8):
            print(f"  {name:<20}{count:>4}")
    if examples:
        print("\nDisagreements:")
        for line in examples:
            print(line)
    if unclassifiable:
        print(f"\nInstructions the four-family schema cannot hold ({len(unclassifiable)}):")
        for line in unclassifiable[:14]:
            print(f"  {line}")
        if len(unclassifiable) > 14:
            print(f"  ... and {len(unclassifiable) - 14} more")
        print("\nThese are real instructions in the notes that the schema drops.")
        print("They are excluded from every fidelity score in this project, so a")
        print("rewrite could lose all of them and still score 1.000. That is the")
        print("most important limitation of the measurement.")

    return {"available": True, "families": overall_rows,
            "overall": {"precision": precision, "recall": recall, "f1": f1,
                        "exact_value_rate": round(exact_rate, 4),
                        "tp": identity["tp"], "fp": identity["fp"],
                        "fn": identity["fn"]},
            "unclassifiable_count": len(unclassifiable),
            "unclassifiable": unclassifiable}


# ---------------------------------------------------------------------------
# 2. detector sensitivity

def study_sensitivity(notes: list[tuple[str, str]]) -> dict:
    found = Counter()
    total = Counter()
    unexpected = Counter()
    misses: list[str] = []
    noise: list[str] = []

    for nid, text in notes:
        base = extract(text)
        for mutation in mutations_for(nid, text):
            total[mutation.name] += 1
            result = diff(base, extract(mutation.text))
            if any(mutation.expectation.matches(item) for item in result.errors):
                found[mutation.name] += 1
            else:
                misses.append(f"  {nid} {mutation.name}: {mutation.detail}\n"
                              f"    expected {mutation.expectation.describe()}\n"
                              f"    reported {[i.message() for i in result.errors][:2]}")
            extra = mutation.unexpected(result.errors)
            unexpected[mutation.name] += len(extra)
            if extra and len(noise) < 8:
                noise.append(f"  {nid} {mutation.name}: "
                             f"{[i.message() for i in extra][:2]}")

    print(rule("2. DETECTOR SENSITIVITY (injected errors with known ground truth)"))
    print("One edit per case, declared in advance. 'found' is detection of that")
    print("edit; 'extra' is anything else the detector reported.\n")
    print(f"{'mutation':<28}{'cases':>7}{'found':>7}{'rate':>8}{'extra':>7}")
    rows = []
    for name in MUTATION_NAMES:
        count = total[name]
        if not count:
            continue
        rate = found[name] / count
        print(f"{name:<28}{count:>7}{found[name]:>7}{rate:>8.3f}{unexpected[name]:>7}")
        rows.append({"mutation": name, "cases": count, "found": found[name],
                     "rate": round(rate, 4), "unexpected": unexpected[name]})
    cases = sum(total.values())
    hits = sum(found.values())
    extras = sum(unexpected.values())
    rate = hits / cases if cases else 0.0
    print(f"{'ALL':<28}{cases:>7}{hits:>7}{rate:>8.3f}{extras:>7}")
    if misses:
        print("\nUndetected:")
        for line in misses[:8]:
            print(line)
    if noise:
        print("\nReported but not planted:")
        for line in noise:
            print(line)
    print("\nRead this number with its limit in mind: the detector and this")
    print("harness were built together, and four early false positives were")
    print("resolved by fixing the extractor (a daily ceiling carrying to the")
    print("wrong drug, an appointment invented from 'return to the emergency")
    print("department', one sentence's threshold applied to every symptom in")
    print("it) and two by fixing the harness. So this measures that the")
    print("detector handles these twelve known error families, not that it")
    print("would handle a thirteenth.")

    return {"mutations": rows, "cases": cases, "found": hits,
            "rate": round(rate, 4), "unexpected": extras}


# ---------------------------------------------------------------------------
# 3. the loop

def study_loop(notes: list[tuple[str, str]], simplifier_name: str,
               target_grade: float, max_rounds: int) -> dict:
    simplifier = build_simplifier(simplifier_name)
    results = []
    rows = []

    print(rule(f"3. LOOP (simplifier: {simplifier.name}, "
               f"target grade {target_grade}, cap {max_rounds} rounds)"))
    print(f"{'note':<17}{'facts':>6}{'grade':>13}{'jargon':>10}"
          f"{'fid':>7}{'wfid':>7}{'rnd':>5}  status")
    for nid, text in notes:
        outcome = run(text, simplifier, note_id=nid,
                      target_grade=target_grade, max_rounds=max_rounds)
        best = outcome.best
        results.append(best.result)
        before, after = outcome.original_readability, best.readability
        print(f"{nid:<17}{best.result.original_count:>6}"
              f"{before.grade:>6.1f}{'->':^3}{after.grade:<4.1f}"
              f"{before.unexplained_count:>5}{'->':^2}{after.unexplained_count:<3}"
              f"{best.result.fidelity:>7.3f}{best.result.weighted_fidelity:>7.3f}"
              f"{len(outcome.rounds):>5}  "
              f"{'ok' if outcome.converged else 'NEEDS REVIEW'}")
        rows.append({"note": nid, "facts": best.result.original_count,
                     "grade_before": before.grade, "grade_after": after.grade,
                     "jargon_before": before.unexplained_count,
                     "jargon_after": after.unexplained_count,
                     "fidelity": round(best.result.fidelity, 4),
                     "weighted_fidelity": round(best.result.weighted_fidelity, 4),
                     "rounds": len(outcome.rounds),
                     "converged": outcome.converged,
                     "omissions": len(best.result.omissions),
                     "corruptions": len(best.result.corruptions),
                     "fabrications": len(best.result.fabrications)})

    summary = aggregate(results)
    grade_before = sum(r["grade_before"] for r in rows) / len(rows)
    grade_after = sum(r["grade_after"] for r in rows) / len(rows)
    jargon_before = sum(r["jargon_before"] for r in rows)
    jargon_after = sum(r["jargon_after"] for r in rows)
    summary.update({
        "simplifier": simplifier.name,
        "mean_grade_before": round(grade_before, 2),
        "mean_grade_after": round(grade_after, 2),
        "mean_grade_drop": round(grade_before - grade_after, 2),
        "jargon_before": jargon_before,
        "jargon_after": jargon_after,
        "notes_meeting_target": sum(1 for r in rows if r["grade_after"] <= target_grade),
        "rows": rows,
    })

    print(f"\n{summary['notes']} notes, {summary['facts']} facts")
    print(f"mean reading grade   {summary['mean_grade_before']} -> "
          f"{summary['mean_grade_after']}  "
          f"({summary['notes_meeting_target']}/{summary['notes']} at or under "
          f"grade {target_grade})")
    print(f"unexplained terms    {jargon_before} -> {jargon_after}")
    print(f"pooled fidelity      {summary['pooled_fidelity']}  "
          f"(weighted {summary['pooled_weighted_fidelity']})")
    print(f"per-note mean        {summary['mean_note_fidelity']}")
    print(f"omitted {summary['omissions']}, corrupted {summary['corruptions']}, "
          f"fabricated {summary['fabrications']}")
    print(f"notes needing review {summary['notes'] - summary['clean_notes']}"
          f"/{summary['notes']}")
    if simplifier_name == "glossary":
        print("\nThe offline baseline substitutes words and cannot restructure a")
        print("sentence, so a high fidelity score here is close to the floor")
        print("rather than an achievement. It is the control. The number worth")
        print("comparing it against is --simplifier claude.")
    return summary


# ---------------------------------------------------------------------------

def study_headline(notes: list[tuple[str, str]], simplifier_name: str,
                   target_grade: float, max_rounds: int) -> dict:
    """One-shot rewrite against the same rewrite with the loop around it.

    The spec's headline experiment. The comparison is honest only because the
    one-shot column is literally round 1 of the same run: same simplifier, same
    prompt, same note, no second sampling. Any difference is the loop.
    """
    simplifier = build_simplifier(simplifier_name)
    one_shot, looped, rows = [], [], []

    print(rule(f"4. ONE-SHOT vs LOOP (simplifier: {simplifier.name})"))
    print(f"{'note':<17}{'facts':>6}{'one-shot':>10}{'looped':>9}{'rnd':>5}"
          f"{'grade one-shot':>16}{'grade looped':>14}")
    for nid, text in notes:
        outcome = run(text, simplifier, note_id=nid,
                      target_grade=target_grade, max_rounds=max_rounds)
        first, best = outcome.rounds[0], outcome.best
        one_shot.append(first.result)
        looped.append(best.result)
        print(f"{nid:<17}{first.result.original_count:>6}"
              f"{first.result.fidelity:>10.3f}{best.result.fidelity:>9.3f}"
              f"{len(outcome.rounds):>5}{first.readability.grade:>16.1f}"
              f"{best.readability.grade:>14.1f}")
        rows.append({"note": nid,
                     "one_shot_fidelity": round(first.result.fidelity, 4),
                     "looped_fidelity": round(best.result.fidelity, 4),
                     "one_shot_grade": first.readability.grade,
                     "looped_grade": best.readability.grade,
                     "rounds": len(outcome.rounds)})

    before, after = aggregate(one_shot), aggregate(looped)
    repaired = sum(1 for a, b in zip(one_shot, looped) if not a.clean and b.clean)
    print(f"\n{'':<17}{'one-shot':>10}{'looped':>9}")
    for label, key in (("pooled fidelity", "pooled_fidelity"),
                       ("weighted", "pooled_weighted_fidelity"),
                       ("clean notes", "clean_notes"),
                       ("omissions", "omissions"),
                       ("corruptions", "corruptions"),
                       ("fabrications", "fabrications")):
        print(f"{label:<17}{str(before[key]):>10}{str(after[key]):>9}")
    print(f"\n{repaired} note(s) the loop repaired, "
          f"{after['notes'] - after['clean_notes']} still needing review.")
    if before["pooled_fidelity"] == after["pooled_fidelity"]:
        print("No difference, because this simplifier loses nothing on the first")
        print("pass. That is a fact about the simplifier, not about the loop.")
        print("Run --simplifier lossy to see the repair round work, or")
        print("--simplifier claude for the result that matters.")
    return {"simplifier": simplifier.name, "one_shot": before, "looped": after,
            "repaired": repaired, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=("all", "extraction", "sensitivity",
                                            "loop", "headline", "threshold"),
                        default="all")
    parser.add_argument("--simplifier", choices=tuple(SIMPLIFIERS), default="glossary")
    parser.add_argument("--target-grade", type=float, default=6.0)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write results here")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model for --simplifier claude (default: %(default)s)")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"),
                        help="effort for --simplifier claude; omit for the model default")
    args = parser.parse_args(argv)
    SIMPLIFIER_OPTIONS.update(model=args.model, effort=args.effort)

    notes = load_notes()
    if not notes:
        print(f"no notes found in {NOTES}", file=sys.stderr)
        return 1
    if not HAVE_WORDFREQ:
        print("WARNING: wordfreq is not installed. Unexplained-term counts are "
              "meaningless without it. pip install wordfreq\n", file=sys.stderr)

    out: dict = {"notes": len(notes)}
    if args.study in ("all", "threshold"):
        print(rule("0. THE RARE-TERM THRESHOLD"))
        print(threshold_report())
    if args.study in ("all", "extraction"):
        # Both rounds, labelled. The round-1 number is the uncontaminated one:
        # it was measured before the annotator's findings changed the schema.
        # The round-2 number is after those changes and is not independent of
        # them, which is why both are printed rather than only the better one.
        out["extraction"] = study_extraction(
            notes, GOLD, ": round 1 gold, scored before the schema changed")
        second = GOLD.with_name("gold.v2.json")
        if second.exists():
            out["extraction_v2"] = study_extraction(
                notes, second, ": round 2 gold, after the schema changed")
    if args.study in ("all", "sensitivity"):
        out["sensitivity"] = study_sensitivity(notes)
    if args.study in ("all", "loop"):
        out["loop"] = study_loop(notes, args.simplifier, args.target_grade,
                                 args.max_rounds)
    if args.study in ("all", "headline"):
        out["headline"] = study_headline(notes, args.simplifier,
                                         args.target_grade, args.max_rounds)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
