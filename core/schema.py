"""The four families of clinical fact, and what counts as the same fact.

A discharge note carries instructions a person has to act on. This module says
what those instructions look like as data, which is what makes it possible to
ask whether a rewrite still says the same thing.

The design decision that matters: every fact has an **identity** and a **value**.
Two facts are the same fact when their identities match, whatever their values
say. That is what separates the three ways a rewrite can go wrong:

  omission     an identity in the original, missing from the rewrite
  corruption   the same identity in both, different value. The dangerous one
  fabrication  an identity in the rewrite that was not in the original

Without an identity distinct from the value, a changed dose looks like one
omission plus one fabrication, and the most dangerous failure mode disappears
into the least dangerous counting.

The field sets follow the spec's four families. Two refinements: `dose_max`
exists because "1 to 2 tablets" collapsed to "2 tablets" doubles the floor, and
`duration_note` exists because "until the bottle is finished" is an antibiotic
instruction that no integer represents.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Fact:
    kind: str
    identity: tuple           # what this fact is about
    value: tuple              # what it says about it
    span: str = ""            # the source sentence, kept so every fact is checkable

    @property
    def key(self) -> tuple:
        return (self.kind, self.identity)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "identity": list(self.identity),
                "value": list(self.value), "span": self.span}


@dataclass(frozen=True)
class Medication:
    name: str
    dose: float | None = None
    dose_max: float | None = None
    unit: str = ""
    route: str = ""
    frequency: str = ""
    duration_days: int | None = None
    duration_note: str = ""
    max_per_day: str = ""     # "6 tablets": the ceiling on a PRN drug
    # A second phase: "apixaban 10 mg BID for 7 days, then 5 mg BID for at
    # least 6 months". Without these four fields the schema could hold only the
    # first phase, and the fact it produced said to stop a blood thinner on day
    # 7 after a pulmonary embolism. Three notes in the corpus prescribe a
    # phased regimen and in all three the second phase is the actual plan.
    then_dose: float | None = None
    then_frequency: str = ""
    then_duration_days: int | None = None
    then_duration_note: str = ""
    # When to take it anyway and when to skip it. The DKA note prescribes a
    # basal insulin that "must be taken every night even if you are not eating"
    # and a bolus insulin to "omit if you skip the meal". Those are opposite
    # instructions, and without this field the two facts were identical apart
    # from the drug name and the dose.
    hold_rule: str = ""
    span: str = ""

    def to_fact(self) -> Fact:
        # Identity is the drug. Everything else is its value, so changing a dose
        # is one corruption rather than an omission plus a fabrication.
        return Fact("medication", (self.name,),
                    (self.dose, self.dose_max, self.unit, self.route,
                     self.frequency, self.duration_days, self.duration_note,
                     self.max_per_day, self.then_dose, self.then_frequency,
                     self.then_duration_days, self.then_duration_note,
                     self.hold_rule),
                    self.span)


@dataclass(frozen=True)
class FollowUp:
    who: str
    within_days: int | None = None
    ordinal: int = 1
    span: str = ""

    def to_fact(self) -> Fact:
        # The ordinal exists because one note can hold two appointments with the
        # same specialty: a two-week incision check and a six-week postpartum
        # visit are both obstetrics. Keyed on the role alone they collapsed into
        # one fact and a rewrite could delete either for free.
        #
        # Ordinals are assigned in window order, so they are reproducible from
        # the text and do not depend on where in the note the line sits. Making
        # the window part of the identity instead would have been simpler, and
        # would have turned every stretched appointment window into an omission
        # plus a fabrication rather than the corruption it is.
        return Fact("follow_up", (self.who, self.ordinal), (self.within_days,),
                    self.span)


@dataclass(frozen=True)
class RedFlag:
    symptom: str
    action: str = ""
    # The threshold carries its direction and its unit, not just a number.
    # "over 101 F" and "below 70 mg/dL" are a ceiling and a floor, and stored
    # as the bare numbers "101" and "70" the comparison direction was lost: a
    # hypoglycaemia rule read as a hyperglycaemia rule.
    threshold: str = ""       # "over 101 F", "below 70 mg/dL", "over 3 pounds"
    span: str = ""

    def to_fact(self) -> Fact:
        return Fact("red_flag", (self.symptom,), (self.action, self.threshold), self.span)


@dataclass(frozen=True)
class Restriction:
    activity: str
    duration_days: int | None = None
    duration_note: str = ""
    limit: str = ""           # "10 pounds"; without it the restriction is a different one
    # "Do not drive for 5 days, and not at all while taking oxycodone." Four
    # notes carry that second clause and without this field every one of them
    # produced a fact clearing the patient to drive on day 6 while still taking
    # an opioid.
    condition: str = ""       # "while taking oxycodone", "unassisted"
    span: str = ""

    def to_fact(self) -> Fact:
        return Fact("restriction", (self.activity,),
                    (self.duration_days, self.duration_note, self.limit,
                     self.condition), self.span)


_ORDER = ("medications", "follow_ups", "red_flags", "restrictions")


@dataclass
class FactSet:
    medications: list[Medication] = field(default_factory=list)
    follow_ups: list[FollowUp] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    restrictions: list[Restriction] = field(default_factory=list)

    def facts(self) -> list[Fact]:
        out: list[Fact] = []
        for name in _ORDER:
            for item in getattr(self, name):
                out.append(item.to_fact())
        return out

    def __len__(self) -> int:
        return sum(len(getattr(self, name)) for name in _ORDER)

    def by_key(self) -> dict[tuple, Fact]:
        return {f.key: f for f in self.facts()}

    def as_dict(self) -> dict:
        return {name: [asdict(item) for item in getattr(self, name)] for name in _ORDER}

    @staticmethod
    def from_dict(payload: dict) -> "FactSet":
        kinds = {"medications": Medication, "follow_ups": FollowUp,
                 "red_flags": RedFlag, "restrictions": Restriction}
        return FactSet(**{name: [cls(**row) for row in payload.get(name, [])]
                          for name, cls in kinds.items()})


# Which positions in a fact's value tuple are safety-critical. A missing route
# is a wording difference. A changed dose is not. The diff reports both but
# only the critical ones count as corruptions.
#
# Index 3 of a medication value is the route, the one field treated as wording
# rather than safety: "by mouth" dropped from a tablet instruction is a style
# loss, and every other field is not.
CRITICAL_VALUE_INDEX: dict[str, tuple[int, ...]] = {
    "medication": (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "follow_up": (0,),
    "red_flag": (0, 1),
    "restriction": (0, 1, 2, 3),
}

VALUE_LABELS: dict[str, tuple[str, ...]] = {
    "medication": ("dose", "dose maximum", "unit", "route", "frequency",
                   "duration (days)", "duration", "daily ceiling",
                   "second-phase dose", "second-phase frequency",
                   "second-phase duration (days)", "second-phase duration",
                   "hold rule"),
    "follow_up": ("window (days)",),
    "red_flag": ("action", "threshold"),
    "restriction": ("duration (days)", "duration", "limit", "condition"),
}
