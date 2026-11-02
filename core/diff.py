"""Compare two fact sets and classify every discrepancy.

Three classes, from the spec, in increasing order of how much they should worry
you:

  omission     the rewrite dropped an instruction. The person goes back and asks.
  fabrication  the rewrite invented an instruction. The person follows it.
  corruption   the rewrite kept the instruction and changed what it says. The
               person follows it, confidently, and it is wrong.

The fidelity score is the spec's:

    F = 1 - (omissions + corruptions + fabrications) / facts_in_original

It can go below zero, because fabrication has no upper bound while the other
two are capped by the size of the original. That is left uncapped rather than
clamped, because a rewrite that invents more than it preserves deserves a
number that says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import CRITICAL_VALUE_INDEX, VALUE_LABELS, Fact, FactSet
from .vocab import ACTION_SEVERITY

CORRUPTION_WEIGHT = 2.0


@dataclass(frozen=True)
class Discrepancy:
    kind: str                 # omission | corruption | fabrication | softening
    fact_kind: str
    identity: tuple
    field: str = ""
    was: object = None
    now: object = None
    note: str = ""

    def message(self) -> str:
        subject = " ".join(str(part) for part in self.identity)
        if self.kind == "omission":
            return f"The {self.fact_kind.replace('_', ' ')} for {subject} is missing from the rewrite."
        if self.kind == "fabrication":
            return (f"The rewrite introduces a {self.fact_kind.replace('_', ' ')} for "
                    f"{subject} that the original does not contain.")
        if self.kind == "corruption":
            return (f"The {self.field} for {subject} changed from "
                    f"{_show(self.was)} to {_show(self.now)}.{(' ' + self.note) if self.note else ''}")
        return (f"The {self.field} for {subject} was dropped "
                f"({_show(self.was)} is no longer stated).")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "fact_kind": self.fact_kind,
                "identity": list(self.identity), "field": self.field,
                "was": self.was, "now": self.now, "message": self.message()}


def _show(value: object) -> str:
    if value in (None, ""):
        return "unstated"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class DiffResult:
    original_count: int
    rewrite_count: int
    omissions: list[Discrepancy] = field(default_factory=list)
    corruptions: list[Discrepancy] = field(default_factory=list)
    fabrications: list[Discrepancy] = field(default_factory=list)
    softenings: list[Discrepancy] = field(default_factory=list)

    @property
    def errors(self) -> list[Discrepancy]:
        return self.omissions + self.corruptions + self.fabrications

    @property
    def fidelity(self) -> float:
        if self.original_count == 0:
            return 1.0 if self.rewrite_count == 0 else 0.0
        return 1.0 - len(self.errors) / self.original_count

    @property
    def weighted_fidelity(self) -> float:
        """Corruptions counted at `CORRUPTION_WEIGHT`, the rest at 1.

        A missing instruction sends someone back to ask. A wrong one does not.
        """
        if self.original_count == 0:
            return 1.0 if self.rewrite_count == 0 else 0.0
        penalty = (len(self.omissions) + len(self.fabrications)
                   + CORRUPTION_WEIGHT * len(self.corruptions))
        return 1.0 - penalty / self.original_count

    @property
    def clean(self) -> bool:
        return not self.errors

    def constraints(self) -> list[str]:
        """The discrepancy list as instructions, for the repair round."""
        return [d.message() for d in
                self.corruptions + self.fabrications + self.omissions]

    def summary(self) -> str:
        return (f"{self.original_count} facts, fidelity {self.fidelity:.3f} "
                f"(weighted {self.weighted_fidelity:.3f}): "
                f"{len(self.omissions)} omitted, {len(self.corruptions)} corrupted, "
                f"{len(self.fabrications)} fabricated")

    def as_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "rewrite_count": self.rewrite_count,
            "fidelity": round(self.fidelity, 4),
            "weighted_fidelity": round(self.weighted_fidelity, 4),
            "omissions": [d.as_dict() for d in self.omissions],
            "corruptions": [d.as_dict() for d in self.corruptions],
            "fabrications": [d.as_dict() for d in self.fabrications],
            "softenings": [d.as_dict() for d in self.softenings],
        }


def _severity_note(fact_kind: str, field_name: str, was: object, now: object) -> str:
    if fact_kind != "red_flag" or field_name != "action":
        return ""
    before = ACTION_SEVERITY.get(str(was), 0)
    after = ACTION_SEVERITY.get(str(now), 0)
    if after < before:
        return "This tells the person to do less than the original did."
    if after > before:
        return "This escalates what the original asked for."
    return ""


def compare_facts(original: Fact, rewrite: Fact) -> list[Discrepancy]:
    """Field-by-field, splitting safety-critical changes from wording changes."""
    out: list[Discrepancy] = []
    critical = CRITICAL_VALUE_INDEX.get(original.kind, ())
    labels = VALUE_LABELS.get(original.kind, ())
    for index, (was, now) in enumerate(zip(original.value, rewrite.value)):
        if was == now:
            continue
        label = labels[index] if index < len(labels) else f"field {index}"
        if index not in critical:
            out.append(Discrepancy("softening", original.kind, original.identity,
                                   label, was, now))
        elif now in (None, "") and was not in (None, ""):
            # The value was stated and is now absent. That is a dropped
            # constraint, not a rewording, so it counts as a corruption.
            out.append(Discrepancy("corruption", original.kind, original.identity,
                                   label, was, now,
                                   "The original states it and the rewrite does not."))
        else:
            out.append(Discrepancy("corruption", original.kind, original.identity,
                                   label, was, now,
                                   _severity_note(original.kind, label, was, now)))
    return out


def diff(original: FactSet, rewrite: FactSet) -> DiffResult:
    before, after = original.by_key(), rewrite.by_key()
    result = DiffResult(original_count=len(original), rewrite_count=len(rewrite))

    for key, fact in before.items():
        if key not in after:
            result.omissions.append(
                Discrepancy("omission", fact.kind, fact.identity, note=fact.span))
            continue
        for item in compare_facts(fact, after[key]):
            (result.softenings if item.kind == "softening"
             else result.corruptions).append(item)

    for key, fact in after.items():
        if key not in before:
            result.fabrications.append(
                Discrepancy("fabrication", fact.kind, fact.identity, note=fact.span))

    return result


def aggregate(results: list[DiffResult]) -> dict:
    """Corpus-level numbers. Facts are pooled rather than averaged over notes,
    so a long note counts for more than a short one, which is the behaviour you
    want when the question is 'how often does this lose a fact'."""
    facts = sum(r.original_count for r in results)
    omissions = sum(len(r.omissions) for r in results)
    corruptions = sum(len(r.corruptions) for r in results)
    fabrications = sum(len(r.fabrications) for r in results)
    errors = omissions + corruptions + fabrications
    clean = sum(1 for r in results if r.clean)
    return {
        "notes": len(results),
        "facts": facts,
        "omissions": omissions,
        "corruptions": corruptions,
        "fabrications": fabrications,
        "pooled_fidelity": round(1.0 - errors / facts, 4) if facts else None,
        "pooled_weighted_fidelity": round(
            1.0 - (omissions + fabrications + CORRUPTION_WEIGHT * corruptions) / facts, 4
        ) if facts else None,
        "mean_note_fidelity": round(sum(r.fidelity for r in results) / len(results), 4)
        if results else None,
        "clean_notes": clean,
        "clean_note_rate": round(clean / len(results), 4) if results else None,
    }
