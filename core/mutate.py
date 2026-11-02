"""Known mutations, for measuring whether the detector actually detects.

The fidelity score is only worth reading if the thing computing it can be shown
to notice a real error. Measuring that against a simplifier's output is
circular, because nobody knows the right answer for a given rewrite. Measuring
it against a deliberate edit is not, because the right answer is whatever was
edited.

Each mutation here takes a note, makes one specific change to its text, and
declares exactly which discrepancy the diff ought to report. `eval.py` then
asks two questions per mutation:

  did it find the planted error?          sensitivity
  did it report anything else as well?    false positives

The second question is the one that keeps the first honest. A detector that
flags everything finds every planted error and is useless.

Mutations edit prose, never the fact set, so the whole extract-diff path is
under test rather than just the comparison.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import extract
from .schema import FactSet


@dataclass(frozen=True)
class Expectation:
    kind: str                 # omission | corruption | fabrication
    fact_kind: str
    identity: tuple
    field: str = ""           # for corruptions

    def matches(self, discrepancy) -> bool:
        return (discrepancy.kind == self.kind
                and discrepancy.fact_kind == self.fact_kind
                and discrepancy.identity == self.identity
                and (not self.field or discrepancy.field == self.field))

    def describe(self) -> str:
        subject = " ".join(str(part) for part in self.identity)
        return f"{self.kind} of {self.fact_kind} {subject}" + (
            f" ({self.field})" if self.field else "")


@dataclass(frozen=True)
class Mutation:
    name: str
    note_id: str
    text: str
    expectation: Expectation
    detail: str = ""
    collateral: tuple[Expectation, ...] = ()

    @property
    def expected(self) -> tuple[Expectation, ...]:
        return (self.expectation,) + self.collateral

    def unexpected(self, discrepancies) -> list:
        """Reported errors that no declared expectation accounts for.

        `collateral` is worked out from the edit itself, never from the
        detector's output. Letting the diff define what counts as expected
        would make the false-positive count unfalsifiable: anything it said
        would be correct by definition.
        """
        return [item for item in discrepancies
                if not any(e.matches(item) for e in self.expected)]


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _replace_in_span(text: str, span: str, new_span: str) -> str | None:
    if span not in text:
        return None
    return text.replace(span, new_span, 1)


def _drop_span(text: str, span: str) -> str | None:
    """Remove one instruction, at sentence granularity.

    Earlier this deleted the whole line the span sat on. Several of these notes
    put three instructions in one prose line, so a "drop one medication"
    mutation silently deleted three and the false-positive count blamed the
    detector for noticing.
    """
    span = span.strip()
    if not span:
        return None
    out = []
    dropped = False
    for line in _lines(text):
        if not dropped and span in line:
            remainder = line.replace(span, "", 1)
            dropped = True
            remainder = re.sub(r"\s{2,}", " ", remainder).strip()
            if remainder and remainder not in "-*•:.":
                out.append(remainder)
            continue
        out.append(line)
    return "\n".join(out) if dropped else None


# ---------------------------------------------------------------------------
# the mutations

def _mentions(text: str, needle: str) -> int:
    return sum(1 for line in _lines(text)
               if re.search(rf"\b{re.escape(needle)}\b", line, re.I))


def _drop_medication(text: str, facts: FactSet):
    out = []
    for med in facts.medications:
        head = med.name.split()[0]
        if _mentions(text, med.name) != 1 or _mentions(text, head) != 1:
            # The drug is named elsewhere in the note, so deleting this line
            # leaves a partial instruction behind. That is a corruption, not an
            # omission, and a mutation whose expected result is ambiguous
            # measures nothing.
            continue
        changed = _drop_span(text, med.span)
        if not changed:
            continue
        others = tuple(
            Expectation("omission", "medication", (other.name,))
            for other in facts.medications
            if other.name != med.name and other.span == med.span)
        out.append(("drop_medication", changed,
                    Expectation("omission", "medication", (med.name,)),
                    f"removed the line instructing {med.name}", others))
    return out


def _halve_dose(text: str, facts: FactSet) -> list[tuple[str, str, Expectation, str]]:
    out = []
    for med in facts.medications:
        if med.dose is None or not med.span:
            continue
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|mL|ml|unit|units|mEq)\b",
                          med.span)
        if not match:
            continue
        halved = float(match.group(1)) / 2
        shown = str(int(halved)) if halved.is_integer() else str(halved)
        new_span = med.span[:match.start(1)] + shown + med.span[match.end(1):]
        changed = _replace_in_span(text, med.span, new_span)
        if changed:
            out.append(("halve_dose", changed,
                        Expectation("corruption", "medication", (med.name,), "dose"),
                        f"{med.name} {match.group(1)} -> {shown} {match.group(2)}"))
    return out


_PRN_PHRASES = (
    " PRN", " prn", " as needed", " if you need it", " if needed", " when needed",
    " only if needed", " PRN ", " as required",
)


def _drop_prn(text: str, facts: FactSet) -> list[tuple[str, str, Expectation, str]]:
    out = []
    for med in facts.medications:
        if "as needed" not in med.frequency or not med.span:
            continue
        new_span = med.span
        for phrase in _PRN_PHRASES:
            if phrase in new_span:
                new_span = new_span.replace(phrase, " " if phrase.endswith(" ") else "", 1)
                break
        else:
            continue
        changed = _replace_in_span(text, med.span, new_span)
        if changed:
            out.append(("drop_prn", changed,
                        Expectation("corruption", "medication", (med.name,), "frequency"),
                        f"{med.name}: dropped 'as needed', making a PRN drug scheduled"))
    return out


def _drop_duration(text: str, facts: FactSet) -> list[tuple[str, str, Expectation, str]]:
    out = []
    for med in facts.medications:
        if med.duration_days is None or not med.span:
            continue
        new_span = re.sub(
            r"\s*(?:for|x)\s+\d+\s*(?:day|week|month)s?\b", "", med.span, count=1,
            flags=re.I)
        if new_span == med.span:
            continue
        changed = _replace_in_span(text, med.span, new_span)
        if changed:
            out.append(("drop_duration", changed,
                        Expectation("corruption", "medication", (med.name,),
                                    "duration (days)"),
                        f"{med.name}: removed the {med.duration_days}-day course length"))
    return out


def _drop_ceiling(text: str, facts: FactSet) -> list[tuple[str, str, Expectation, str]]:
    out = []
    for med in facts.medications:
        if not med.max_per_day:
            continue
        span = next((line for line in _lines(text)
                     if re.search(r"exceed|more than|maximum|max |limit", line, re.I)
                     and med.name.split()[0] in line.lower()), "")
        target = span or med.span
        new_span = re.sub(
            r"\s*(?:Do not exceed|Not more than|No more than|Maximum(?: of)?|Max(?: of)?"
            r"|Limit(?: yourself)? to)[^.;]*[.;]?", "", target, count=1, flags=re.I)
        if new_span == target:
            continue
        changed = _replace_in_span(text, target, new_span)
        if changed:
            out.append(("drop_ceiling", changed,
                        Expectation("corruption", "medication", (med.name,),
                                    "daily ceiling"),
                        f"{med.name}: removed the {med.max_per_day} daily ceiling"))
    return out


_DOWNGRADE = [
    (r"Call 911 or go to the nearest emergency department", "Call the office"),
    (r"CALL 911 OR GO TO THE NEAREST EMERGENCY DEPARTMENT", "CALL THE OFFICE"),
    (r"Call 911", "Call the office"),
    (r"CALL 911", "CALL THE OFFICE"),
    (r"Return to the emergency department immediately", "Call the office"),
    (r"RETURN TO THE EMERGENCY DEPARTMENT IMMEDIATELY", "CALL THE OFFICE"),
    (r"Present to the emergency department", "Call the office"),
    (r"go to the nearest emergency department", "call the office"),
    (r"Return to the emergency department", "Call the office"),
]


def _downgrade_action(text: str, facts: FactSet):
    emergencies = [flag for flag in facts.red_flags if flag.action == "emergency"]
    if not emergencies:
        return []
    for pattern, replacement in _DOWNGRADE:
        match = re.search(pattern, text)
        if not match:
            continue
        changed = re.sub(pattern, replacement, text, count=1)
        # Every emergency flag announced by the edited sentence is downgraded by
        # this one edit. They are enumerated from the spans, which is a property
        # of the note, rather than from what the diff happens to report.
        edited = match.group(0)
        affected = [flag for flag in emergencies
                    if flag.span and (edited in flag.span
                                      or flag.span.startswith(edited[:30]))]
        if not affected:
            affected = [flag for flag in emergencies
                        if flag.span and flag.span not in changed]
        if not affected:
            continue
        primary, *rest = affected
        return [("downgrade_action", changed,
                 Expectation("corruption", "red_flag", (primary.symptom,), "action"),
                 f"an emergency instruction softened to '{replacement}', "
                 f"affecting {len(affected)} warning sign(s)",
                 tuple(Expectation("corruption", "red_flag", (flag.symptom,), "action")
                       for flag in rest))]
    return []


# A careless simplifier does not delete "temperature above 101 F" and leave
# "temperature". It writes "a fever". The number vanishes and the warning sign
# survives, which is the dangerous version and the one worth measuring. These
# are the plain phrasings that keep each symptom recognisable without its
# threshold.
_PLAIN_SYMPTOM = {
    "fever": "a fever",
    "high blood sugar": "high blood sugar",
    "low blood sugar": "low blood sugar",
    "weight gain": "weight gain",
    "bleeding": "bleeding that will not stop",
    "swelling": "swelling",
    "severe pain": "severe pain",
    "headache": "a bad headache",
    "low urine output": "passing less urine",
}

_THRESHOLD_PHRASE = re.compile(
    r"(?:temperature|temp|blood glucose|blood sugar|glucose|weight gain|fever)?"
    r"\s*(?:over|above|higher than|more than|greater than|exceeds|below|under)\s*"
    r"\d+(?:\.\d+)?\s*(?:°\s*)?(?:F\b|C\b|degrees?|mg/dL|pounds?|lbs?)?", re.I)


def _drop_threshold(text: str, facts: FactSet):
    out = []
    for flag in facts.red_flags:
        if not flag.threshold or not flag.span:
            continue
        plain = _PLAIN_SYMPTOM.get(flag.symptom)
        if not plain:
            continue
        if len(_THRESHOLD_PHRASE.findall(flag.span)) != 1:
            # One sentence, several thresholds: replacing "the first one" edits
            # a different symptom's number than the one this case claims to
            # test. In the DKA note that rewrote the hyperglycaemia rule while
            # declaring it had rewritten the hypoglycaemia rule.
            continue
        new_span = _THRESHOLD_PHRASE.sub(plain, flag.span, count=1)
        if new_span == flag.span:
            continue
        changed = _replace_in_span(text, flag.span, new_span)
        if not changed:
            continue
        # The mutation only means what it claims if the symptom survives it.
        after = {f.symptom for f in extract(changed).red_flags}
        if flag.symptom not in after:
            continue
        out.append(("drop_threshold", changed,
                    Expectation("corruption", "red_flag", (flag.symptom,),
                                "threshold"),
                    f"{flag.symptom}: '{flag.threshold}' replaced by '{plain}'", ()))
    return out


def _drop_follow_up(text: str, facts: FactSet):
    out = []
    counts = {}
    for item in facts.follow_ups:
        counts[item.who] = counts.get(item.who, 0) + 1
    for item in facts.follow_ups:
        if counts[item.who] != 1 or item.who == "clinic":
            # Removing one of two same-specialty appointments shifts the
            # ordinals of the rest, so the expected result is not a clean
            # omission. The "clinic" bucket is the unnamed-specialty catch-all
            # and holds several appointments by design. Both cases are measured
            # under stretch_window instead, where the identity is stable.
            continue
        changed = _drop_span(text, item.span)
        if not changed:
            continue
        others = tuple(
            Expectation("omission", "follow_up", (other.who, other.ordinal))
            for other in facts.follow_ups
            if other.who != item.who and other.span == item.span)
        out.append(("drop_follow_up", changed,
                    Expectation("omission", "follow_up", (item.who, item.ordinal)),
                    f"removed the {item.who} appointment", others))
    return out


def _stretch_window(text: str, facts: FactSet):
    out = []
    for item in facts.follow_ups:
        if item.within_days is None or not item.span:
            continue
        match = re.search(r"\b(\d+)\s*(day|week|month)s?\b", item.span, re.I)
        if not match:
            continue
        stretched = int(match.group(1)) * 3
        scale = {"day": 1, "week": 7, "month": 30}[match.group(2).lower()]
        if any(other.who == item.who and other.ordinal != item.ordinal
               and other.within_days == stretched * scale
               for other in facts.follow_ups):
            # Tripling this window would land it on a sibling appointment's
            # date, merging two facts into one. The expected result would then
            # be a corruption plus an omission, which is not what this mutation
            # claims to test.
            continue
        new_span = (item.span[:match.start(1)] + str(stretched)
                    + item.span[match.end(1):])
        changed = _replace_in_span(text, item.span, new_span)
        if not changed:
            continue
        others = tuple(
            Expectation("corruption", "follow_up", (other.who, other.ordinal),
                        "window (days)")
            for other in facts.follow_ups
            if not (other.who == item.who and other.ordinal == item.ordinal)
            and other.span == item.span)
        out.append(("stretch_window", changed,
                    Expectation("corruption", "follow_up", (item.who, item.ordinal),
                                "window (days)"),
                    f"{item.who}: window tripled to {stretched} {match.group(2)}s",
                    others))
    return out


def _drop_restriction_duration(text: str, facts: FactSet):
    out = []
    for item in facts.restrictions:
        if item.duration_days is None or not item.span:
            continue
        new_span = re.sub(r"\s*for\s+\d+\s*(?:day|week|month)s?\b", "", item.span,
                          count=1, flags=re.I)
        if new_span == item.span:
            continue
        changed = _replace_in_span(text, item.span, new_span)
        if not changed:
            continue
        # "Do not drive for 5 days, and not at all while taking oxycodone" is
        # two restrictions in one sentence; removing the window affects both.
        others = tuple(
            Expectation("corruption", "restriction", (other.activity,),
                        "duration (days)")
            for other in facts.restrictions
            if other.activity != item.activity and other.span == item.span)
        out.append(("drop_restriction_duration", changed,
                    Expectation("corruption", "restriction", (item.activity,),
                                "duration (days)"),
                    f"{item.activity}: removed the {item.duration_days}-day limit",
                    others))
    return out


def _drop_weight_limit(text: str, facts: FactSet) -> list[tuple[str, str, Expectation, str]]:
    out = []
    for item in facts.restrictions:
        if not item.limit or not item.span:
            continue
        new_span = re.sub(
            r"\s*(?:greater than|more than|over|heavier than|above)?\s*"
            r"\d+\s*(?:pounds?|lbs?|kilograms?|kgs?)\b",
            "", item.span, count=1, flags=re.I)
        if new_span == item.span:
            continue
        changed = _replace_in_span(text, item.span, new_span)
        if changed:
            out.append(("drop_weight_limit", changed,
                        Expectation("corruption", "restriction", (item.activity,),
                                    "limit"),
                        f"{item.activity}: removed the {item.limit} limit"))
    return out


ADDED_LINE = "- Amiodarone 200 mg PO QD for 30 days."


def _add_medication(text: str, facts: FactSet):
    if any(med.name == "amiodarone" for med in facts.medications):
        return []
    # Not just any anchor: inserting a line between a drug and the sentence that
    # states its daily ceiling reassigns that ceiling to the inserted drug,
    # because the extractor carries a ceiling to the most recent drug. An
    # anchor whose own line holds no ceiling avoids a mutation that changes two
    # things while claiming to change one.
    for med in facts.medications:
        if not med.span or med.max_per_day:
            continue
        line = next((row for row in _lines(text) if med.span in row), "")
        if re.search(r"exceed|more than|maximum|max |limit", line, re.I):
            continue
        changed = _replace_in_span(text, line or med.span,
                                   (line or med.span) + "\n" + ADDED_LINE)
        if changed:
            return [("add_medication", changed,
                     Expectation("fabrication", "medication", ("amiodarone",)),
                     "inserted a drug the original never mentions", ())]
    return []


MUTATORS = (
    _drop_medication, _halve_dose, _drop_prn, _drop_duration, _drop_ceiling,
    _downgrade_action, _drop_threshold, _drop_follow_up, _stretch_window,
    _drop_restriction_duration, _drop_weight_limit, _add_medication,
)

MUTATION_NAMES = (
    "drop_medication", "halve_dose", "drop_prn", "drop_duration", "drop_ceiling",
    "downgrade_action", "drop_threshold", "drop_follow_up", "stretch_window",
    "drop_restriction_duration", "drop_weight_limit", "add_medication",
)


def mutations_for(note_id: str, text: str, *, limit_per_kind: int = 2) -> list[Mutation]:
    """Every mutation applicable to this note.

    `limit_per_kind` keeps one long note from dominating the aggregate: a note
    with nine drugs would otherwise contribute nine times as many
    drop_medication cases as a note with one.
    """
    facts = extract(text)
    out: list[Mutation] = []
    for mutator in MUTATORS:
        produced = mutator(text, facts)
        for row in produced[:limit_per_kind]:
            name, changed, expectation, detail = row[:4]
            collateral = row[4] if len(row) > 4 else ()
            out.append(Mutation(name=name, note_id=note_id, text=changed,
                                expectation=expectation, detail=detail,
                                collateral=tuple(collateral)))
    return out
