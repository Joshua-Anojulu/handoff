"""Deterministic extraction of clinical facts from discharge prose.

Why deterministic and not a model call: this extractor runs on the original
note and on the rewrite, and the diff between the two results is the project's
measurement. If the extractor were a language model, a language model would be
grading a language model, the measurement would inherit the grader's failures,
and nobody could reproduce the number. Regex and a vocabulary are worse at
language and far better at being an instrument. The model's job here is the
rewrite, which is the part that needs language.

One code path. There is no flag anywhere in this module that says whether the
text being read is an original or a simplification, because the moment such a
flag exists the comparison stops meaning anything.

The extractor's own accuracy is therefore a thing that has to be measured
rather than assumed, which is what `data/gold/` and `eval.py` are for.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .schema import FactSet, FollowUp, Medication, RedFlag, Restriction
from .vocab import (ACTION_SEVERITY, ACTIONS, ACTIVITIES, BRAND_TO_GENERIC,
                    DRUGS, FOLLOWUP_ROLES, SYMPTOM_SUBSUMES, SYMPTOMS,
                    alias_pattern, matching_keys)

# ---------------------------------------------------------------------------
# numbers

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fourteen": 14, "twenty": 20, "thirty": 30, "sixty": 60,
    "ninety": 90, "half": 0.5, "a half": 0.5, "one half": 0.5,
}
_NUM_WORD_RE = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
NUM = rf"(?:\d+(?:\.\d+)?|{_NUM_WORD_RE})"


def parse_number(token: str) -> float | None:
    token = token.strip().lower()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return NUMBER_WORDS.get(token)


def _int_or_none(value: float | None) -> int | None:
    return None if value is None else int(round(value))


# ---------------------------------------------------------------------------
# units
#
# Normalisations that collapse two spellings of one instruction. Each one makes
# the measured error rate lower, so each is a plain equivalence a pharmacist
# would accept, not a loosening to make the numbers look good.

UNIT_ALIASES = [
    (r"milligrams?|\bmgs?\b", "mg", 1.0),
    (r"micrograms?|\bmcgs?\b|µg", "mcg", 1.0),
    (r"\bgrams?\b|\bgm\b|\bg\b", "g", 1.0),
    (r"milliliters?|millilitres?|\bmls?\b|\bcc\b", "mL", 1.0),
    (r"tablespoons?|\btbsp\b", "mL", 15.0),
    (r"teaspoons?|\btsp\b", "mL", 5.0),
    (r"\bunits?\b", "unit", 1.0),
    (r"tablets?|pills?|capsules?|caplets?|\btabs?\b|\bcaps?\b", "tablet", 1.0),
    (r"puffs?|inhalations?", "puff", 1.0),
    (r"drops?|\bgtts?\b", "drop", 1.0),
    (r"patch(?:es)?", "patch", 1.0),
    (r"sprays?", "spray", 1.0),
    (r"suppositor(?:y|ies)", "suppository", 1.0),
    (r"percent|%", "%", 1.0),
]
UNIT_RE = "|".join(pattern for pattern, _, _ in UNIT_ALIASES)


def normalise_unit(raw: str) -> tuple[str, float]:
    """Return (canonical unit, multiplier to apply to the dose)."""
    text = raw.strip().lower()
    for pattern, canonical, factor in UNIT_ALIASES:
        if re.fullmatch(pattern, text, re.I):
            return canonical, factor
    return text, 1.0


# ---------------------------------------------------------------------------
# frequency
#
# "every 6 hours" and "four times a day" are kept distinct on purpose. They are
# not the same instruction: one means setting an alarm for 2am and the other
# means meals and bedtime. A rewrite that swaps them is flagged, and that is
# the intended behaviour rather than an extractor limitation.

_FREQ_RULES: list[tuple[str, object]] = [
    (rf"\bevery\s+({NUM})\s*(?:to|-|–)\s*({NUM})\s*h(?:ours?|rs?)?\b",
     lambda m: f"every {_fmt(m.group(1))} to {_fmt(m.group(2))} hours"),
    (rf"\bq\.?\s*({NUM})\s*(?:to|-|–)\s*({NUM})\s*h\b",
     lambda m: f"every {_fmt(m.group(1))} to {_fmt(m.group(2))} hours"),
    (rf"\bevery\s+({NUM})\s*h(?:ours?|rs?)?\b",
     lambda m: f"every {_fmt(m.group(1))} hours"),
    (rf"\bq\.?\s*({NUM})\s*h\b", lambda m: f"every {_fmt(m.group(1))} hours"),
    (r"\bevery\s+other\s+day\b|\bq\.?o\.?d\.?\b", "every other day"),
    (r"\bfour\s+times?\s+(?:a|per|each)\s+day\b|\b4\s*(?:x|times?)\s*(?:a|per|each)?\s*day\b|\bq\.?i\.?d\.?\b",
     "four times daily"),
    (r"\bthree\s+times?\s+(?:a|per|each)\s+day\b|\b3\s*(?:x|times?)\s*(?:a|per|each)?\s*day\b|\bt\.?i\.?d\.?\b",
     "three times daily"),
    (r"\btwice\s+(?:a|per|each)\s+day\b|\btwice\s+daily\b|\btwo\s+times?\s+(?:a|per|each)\s+day\b"
     r"|\b2\s*(?:x|times?)\s*(?:a|per|each)?\s*day\b|\bb\.?i\.?d\.?\b"
     r"|\bmorning\s+and\s+(?:night|evening|bedtime)\b", "twice daily"),
    (r"\bonce\s+(?:a|per|each)\s+day\b|\bonce\s+daily\b|\bone\s+time\s+(?:a|per|each)\s+day\b"
     r"|\bevery\s+day\b|\bq\.?d\.?\b|\bdaily\b|\beach\s+morning\b|\bevery\s+morning\b", "once daily"),
    (r"\bat\s+bedtime\b|\bbefore\s+bed\b|\bq\.?h\.?s\.?\b|\bnightly\b|\bevery\s+night\b", "at bedtime"),
    (rf"\bonce\s+(?:a|per|each)\s+week\b|\bweekly\b|\bevery\s+week\b", "weekly"),
    (rf"\bevery\s+({NUM})\s+days?\b", lambda m: f"every {_fmt(m.group(1))} days"),
    (rf"\bevery\s+({NUM})\s+weeks?\b", lambda m: f"every {_fmt(m.group(1))} weeks"),
]

_PRN_RE = re.compile(
    r"\bas\s+needed\b|\bif\s+needed\b|\bwhen\s+needed\b|\bp\.?r\.?n\.?\b"
    r"|\bif\s+you\s+need\s+(?:it|them)\b|\bonly\s+if\b|\bas\s+required\b", re.I)


def _fmt(token: str) -> str:
    value = parse_number(token)
    if value is None:
        return token
    return str(int(value)) if float(value).is_integer() else str(value)


def normalise_frequency(text: str) -> str:
    """Canonical frequency string, with ' as needed' appended when PRN."""
    found = ""
    position = len(text) + 1
    for pattern, result in _FREQ_RULES:
        match = re.search(pattern, text, re.I)
        if match and match.start() < position:
            position = match.start()
            found = result(match) if callable(result) else result
    if _PRN_RE.search(text):
        found = f"{found} as needed".strip() if found else "as needed"
    return found


# ---------------------------------------------------------------------------
# duration

_PERIOD_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365, "hour": 0}

_UNTIL_FINISHED = re.compile(
    r"until\s+(?:it\s+is\s+|they\s+are\s+|the\s+\w+\s+is\s+)?(?:finished|gone|empty|done|used\s+up)"
    r"|finish\s+(?:the|all|every)\b|complete\s+the\s+(?:whole|entire|full)?\s*course"
    r"|take\s+all\s+of\s+(?:it|them)|until\s+you\s+finish", re.I)

_UNTIL_EVENT_RULES: list[tuple[str, str]] = [
    # Each rule maps many phrasings onto one canonical note, so that the
    # original's "until the staples are out" and the rewrite's "until your
    # staples come out" are the same instruction rather than a corruption.
    (r"until\s+(?:your\s+|the\s+)?(?:stitches|staples|sutures)\s+"
     r"(?:are\s+|have\s+|come\s+|been\s+)*(?:out|removed|taken\s+out)", "until staples out"),
    (r"until\s+(?:your\s+|the\s+)?(?:follow[\s-]?up|next\s+visit|next\s+appointment"
     r"|appointment|clinic\s+visit|recheck)", "until follow-up"),
    (r"until\s+(?:your\s+)?(?:surgeon|doctor|provider|clinician|nurse)\s+"
     r"(?:says|tells\s+you|clears\s+you|gives\s+the\s+ok|says\s+it\s+is\s+ok)"
     r"|until\s+(?:you\s+are\s+)?cleared", "until cleared"),
    (r"until\s+(?:the\s+|your\s+)?(?:incision|wound|cut)\s+(?:is\s+|has\s+)*"
     r"(?:healed|closed|fully\s+healed)", "until healed"),
]


_THEN_RE = re.compile(r",?\s*(?:and\s+)?then\b(?P<rest>[^.;]*)", re.I)
_THEN_STOP = re.compile(r"^\s*(?:stop|discontinue|nothing|no\s+more)\b", re.I)


def extract_second_phase(text: str) -> tuple[float | None, str, int | None, str]:
    """The "then ..." half of a phased regimen.

    "Apixaban 10 mg PO BID for 7 days, then 5 mg PO BID for at least 6 months"
    is two orders, and the second one is the plan. "Prednisone 40 mg QD for 5
    days, then stop" is also two, and the second is a hard stop that must not
    be silently dropped either.
    """
    match = _THEN_RE.search(text)
    if not match:
        return None, "", None, ""
    rest = match.group("rest")
    if _THEN_STOP.match(rest):
        return None, "", None, "stop"
    dose, _, _ = _dose(rest)
    frequency = normalise_frequency(rest)
    days, note = _numeric_duration(rest)
    if dose is None and not frequency and days is None and not note:
        return None, "", None, ""
    return dose, frequency, days, note


def extract_duration(text: str) -> tuple[int | None, str]:
    """Return (days, note). Exactly one of the two is set when a duration is found."""
    if _UNTIL_FINISHED.search(text):
        return None, "until finished"
    for pattern, note in _UNTIL_EVENT_RULES:
        if re.search(pattern, text, re.I):
            return None, note
    # Only the first phase: anything after "then" belongs to the second one, so
    # "for 7 days, then 5 mg for 6 months" must not read as a 180-day course.
    head = _THEN_RE.split(text)[0]
    return _numeric_duration(head)


def _numeric_duration(text: str) -> tuple[int | None, str]:
    if _UNTIL_FINISHED.search(text):
        return None, "until finished"
    for pattern, note in _UNTIL_EVENT_RULES:
        if re.search(pattern, text, re.I):
            return None, note
    pattern = (rf"(?:for|over|x|during)\s+(?:at\s+least\s+|about\s+)?"
               rf"(?:the\s+(?:next|first)\s+)?({NUM})"
               rf"\s*(?:to|-|–)?\s*(?:{NUM})?\s*(day|week|month|year|hour)s?")
    match = re.search(pattern, text, re.I)
    if not match:
        match = re.search(rf"\b({NUM})\s+more\s+(day|week|month)s?\b", text, re.I)
    if not match:
        return None, ""
    count = parse_number(match.group(1))
    if count is None:
        return None, ""
    period = match.group(2).lower()
    if period == "hour":
        # "scheduled for 72 hours, then PRN" is a three-day course.
        return _int_or_none(count / 24) or None, ""
    return _int_or_none(count * _PERIOD_DAYS[period]), ""


_WITHIN_RE = re.compile(
    rf"(?:in|within|after|at)\s+(?:about\s+|around\s+|roughly\s+)?({NUM})"
    rf"\s*(?:to|-|–)\s*(?:{NUM})\s*(day|week|month)s?"
    rf"|(?:in|within|after|at)\s+(?:about\s+|around\s+|roughly\s+)?({NUM})\s*(day|week|month)s?",
    re.I)


def extract_window(text: str) -> int | None:
    """Days until an appointment. A range takes its lower bound: the earlier
    date is the one that constrains the person."""
    match = _WITHIN_RE.search(text)
    if not match:
        if re.search(r"\b(?:tomorrow|next\s+day)\b", text, re.I):
            return 1
        if re.search(r"\bnext\s+week\b", text, re.I):
            return 7
        return None
    count = parse_number(match.group(1) or match.group(3) or "")
    period = (match.group(2) or match.group(4) or "day").lower()
    if count is None:
        return None
    return _int_or_none(count * _PERIOD_DAYS[period])


# ---------------------------------------------------------------------------
# segmentation and negation

_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")


def segments_with_lines(text: str) -> list[tuple[int, str]]:
    """Split into the units that carry one instruction each, keeping the line.

    Lines first, because discharge notes are bulleted, then sentences within a
    line. A line ending in a colon is kept as its own segment so the action it
    announces can be carried on to the bullets beneath it.

    The line number matters for one thing: a drug's daily ceiling is written in
    its own sentence but always on the drug's own line ("- Acetaminophen 1000 mg
    q8h PRN. Do not exceed 3000 mg per day."), so the carry can be restricted to
    the line rather than reaching back to whatever drug came last.
    """
    out: list[tuple[int, str]] = []
    for index, line in enumerate(text.splitlines()):
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        if line.endswith(":"):
            out.append((index, line))
            continue
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", line):
            piece = piece.strip()
            if piece:
                out.append((index, piece))
    return out


def segments(text: str) -> list[str]:
    return [segment for _, segment in segments_with_lines(text)]


_NEGATION = re.compile(
    r"\b(?:do\s+not|don'?t|never|avoid|refrain\s+from|abstain|stay\s+away"
    r"|stop\s+(?:taking|using)|hold\s+(?:your|the|all)|no\s+longer\s+take"
    r"|should\s+not|must\s+not|cannot|can'?t|hold\s+off"
    # "Wait 2 weeks before driving" is a restriction and the words between
    # "wait" and "before" are the duration, so the pattern has to span them.
    r"|wait\b[^.;]{0,30}?\bbefore\b|wait\s+(?:to|until)"
    r"|skip|not\s+at\s+all|nothing\s+by|nothing\s+to\s+eat)\b", re.I)
_BARE_NO = re.compile(r"(?:^|[:;,]\s*)no\s+(?!longer)", re.I)
_LIMIT = re.compile(r"\b(?:limit|restrict|keep\s+(?:it\s+)?(?:to|under|below)"
                    r"|not?\s+more\s+than|no\s+(?:heavier|greater)\s+than|up\s+to)\b", re.I)


def is_negated(segment: str) -> bool:
    return bool(_NEGATION.search(segment) or _BARE_NO.search(segment))


# ---------------------------------------------------------------------------
# medications

_DRUG_ALTERNATION = "|".join(
    re.escape(name) for name in
    sorted(set(DRUGS) | set(BRAND_TO_GENERIC), key=len, reverse=True))
_DRUG_RE = re.compile(rf"\b({_DRUG_ALTERNATION})\b", re.I)

_ROUTES = {
    "oral": (r"\bby\s+mouth\b", r"\borally\b", r"\bp\.?o\.?\b", r"\bswallow"),
    "sublingual": (r"\bunder\s+(?:the|your)\s+tongue\b", r"\bsublingual",
                   r"\bs\.?l\.?\b"),
    "topical": (r"\btopical", r"\bon\s+(?:the|your)\s+skin\b", r"\brub\s+(?:it|this)\s+on\b",
                r"\bapply\s+(?:a\s+)?(?:thin\s+)?(?:layer|amount)\b"),
    "subcutaneous": (r"\bsubcutaneous", r"\bsub[\s-]?q\b", r"\bunder\s+the\s+skin\b",
                     r"\binject"),
    "intravenous": (r"\bintravenous", r"\bi\.?v\.?\b", r"\bthrough\s+(?:the|your)\s+iv\b"),
    "ophthalmic": (r"\bin\s+(?:the|your|each)\s+eye", r"\bophthalmic", r"\beye\s+drops?\b"),
    "otic": (r"\bin\s+(?:the|your|each)\s+ear", r"\botic\b", r"\bear\s+drops?\b"),
    "inhaled": (r"\binhal", r"\bpuffs?\b", r"\bwith\s+(?:your|the)\s+inhaler\b"),
    "nasal": (r"\bin\s+(?:each|the|your)\s+nostril", r"\bnasal", r"\bnose\s+spray\b"),
    "rectal": (r"\brectal", r"\bp\.?r\.?\b", r"\bsuppositor"),
}


def _route(text: str) -> str:
    for canonical, patterns in _ROUTES.items():
        for pattern in patterns:
            if re.search(pattern, text, re.I):
                return canonical
    return ""


_DOSE_RE = re.compile(
    rf"({NUM})\s*(?:to|-|–|or)\s*({NUM})\s*({UNIT_RE})"
    rf"|({NUM})\s*({UNIT_RE})", re.I)


def _dose(text: str) -> tuple[float | None, float | None, str]:
    for match in _DOSE_RE.finditer(text):
        if match.group(3):
            low, high, raw = match.group(1), match.group(2), match.group(3)
        else:
            low, high, raw = match.group(4), None, match.group(5)
        unit, factor = normalise_unit(raw)
        value = parse_number(low)
        if value is None:
            continue
        top = parse_number(high) if high else None
        return (round(value * factor, 4),
                round(top * factor, 4) if top is not None else None,
                unit)
    return None, None, ""


def canonical_drug(name: str) -> str:
    key = name.strip().lower()
    return BRAND_TO_GENERIC.get(key, key)


def _drug_windows(segment: str) -> list[tuple[str, str]]:
    """Each drug in the segment with the slice of text that belongs to it.

    A drug owns the text from the end of the previous drug to the start of the
    next, so "take 500 mg of acetaminophen" and "acetaminophen 500 mg" both
    find their dose.
    """
    matches = list(_DRUG_RE.finditer(segment))
    if not matches:
        return []
    out = []
    for index, match in enumerate(matches):
        start = matches[index - 1].end() if index else 0
        end = matches[index + 1].start() if index + 1 < len(matches) else len(segment)
        out.append((canonical_drug(match.group(1)), segment[start:end]))
    return out


_MED_CUE = re.compile(
    r"\btake\b|\btaking\b|\buse\b|\busing\b|\bapply\b|\binject\b|\bcontinue\b"
    r"|\bstart\b|\bresume\b|\bgiven\b|\bprescri|\bswallow\b|\bplace\b|\bspray\b"
    r"|\bchew\b|\binhale\b|\bkeep\s+taking\b|\bmedication"
    # The plain-language forms too, so that rewriting "medications" as
    # "medicines" or "as prescribed" as "exactly as written here" does not stop
    # the sentence reading as a medication instruction. Checked mechanically by
    # scripts/check_vocab.py.
    r"|\bmedicine|\bprescription\b|\bwritten\s+here\b|\bput\s+ice\b", re.I)

# A ceiling on a PRN drug. "Do not exceed 6 tablets in 24 hours" is the
# instruction that stops someone taking a safe dose eight times, and it sits in
# its own sentence after the drug line, so it needs the carry below.
_MAX_PER_DAY = re.compile(
    rf"(?:do\s+not\s+exceed|not?\s+more\s+than|no\s+more\s+than|maximum\s+(?:of\s+)?"
    rf"|max\s+(?:of\s+)?|up\s+to\s+a\s+maximum\s+of|limit\s+(?:yourself\s+)?to)"
    rf"\s*({NUM})\s*({UNIT_RE})?"
    rf"(?:[^.;]{{0,30}}?(?:in\s+(?:a|one|24|twenty[\s-]four)\s*(?:hour|hours|hrs|day)"
    rf"|per\s+day|a\s+day|each\s+day|daily|in\s+one\s+day))", re.I)


def _max_per_day(segment: str) -> str:
    match = _MAX_PER_DAY.search(segment)
    if not match:
        return ""
    amount = parse_number(match.group(1))
    if amount is None:
        return ""
    unit = normalise_unit(match.group(2))[0] if match.group(2) else "doses"
    multiplier = normalise_unit(match.group(2))[1] if match.group(2) else 1.0
    total = amount * multiplier
    return f"{_fmt(str(total))} {unit}"


# Opposite instructions that otherwise look identical. The basal insulin in the
# DKA note must be taken even when the person is not eating; the bolus insulin
# must be skipped when the meal is.
_HOLD_RULES: list[tuple[str, str]] = [
    (r"even\s+if\s+you\s+(?:are\s+not|do\s+not|don'?t|aren'?t)\s+eat",
     "take even if not eating"),
    (r"even\s+if\s+you\s+(?:feel\s+better|are\s+better)", "take even if feeling better"),
    (r"(?:omit|skip|hold)\s+(?:the\s+|this\s+|your\s+)?dose\s+if\s+you\s+skip",
     "omit if the meal is skipped"),
    (r"(?:omit|skip|hold)\s+(?:the\s+|this\s+|your\s+)?dose\s+if", "omit if"),
    (r"take\s+with\s+(?:a\s+meal|food)|with\s+(?:food|meals?)\b", "take with food"),
    (r"on\s+an\s+empty\s+stomach|with\s+no\s+food", "take on an empty stomach"),
    (r"do\s+not\s+double", "do not double a missed dose"),
]


# A rule about whether to take the dose at all outranks advice about what to
# take it with. "Insulin lispro ... with meals. Omit the dose if you skip the
# meal." contains both, and keeping the first one found recorded the bolus
# insulin as "take with food", losing the instruction that distinguishes it
# from the basal one.
_HOLD_RANK = {"omit if the meal is skipped": 3, "take even if not eating": 3,
              "omit if": 2, "take even if feeling better": 2,
              "do not double a missed dose": 2}


def _hold_rule(segment: str) -> str:
    best, rank = "", -1
    for pattern, label in _HOLD_RULES:
        if re.search(pattern, segment, re.I) and _HOLD_RANK.get(label, 1) > rank:
            best, rank = label, _HOLD_RANK.get(label, 1)
    return best


def _better_hold(existing: str, candidate: str) -> bool:
    return bool(candidate) and _HOLD_RANK.get(candidate, 1) > _HOLD_RANK.get(existing, 0)


def extract_medications(text: str) -> list[Medication]:
    out: dict[str, Medication] = {}
    recent, recent_line = "", -1
    for line_no, segment in segments_with_lines(text):
        ceiling = _max_per_day(segment)
        if is_negated(segment) or not _drug_windows(segment):
            # A ceiling sentence is negated ("do not exceed") and names no drug,
            # so it attaches to the drug instructed on the same line. Restricted
            # to the line, not simply to the most recent drug: with the looser
            # rule, deleting a drug's sentence handed its ceiling to whichever
            # drug happened to come before it.
            if (ceiling and recent in out and line_no == recent_line
                    and not out[recent].max_per_day):
                out[recent] = replace(out[recent], max_per_day=ceiling)
            # A hold rule is written the same way: its own sentence, on the
            # drug's line, naming no drug. "Omit the dose if you skip the meal."
            rule = _hold_rule(segment)
            if (recent in out and line_no == recent_line
                    and _better_hold(out[recent].hold_rule, rule)):
                out[recent] = replace(out[recent], hold_rule=rule)
            if is_negated(segment):
                continue                  # otherwise handled as a restriction
        has_cue = bool(_MED_CUE.search(segment))
        for name, window in _drug_windows(segment):
            dose, dose_max, unit = _dose(window)
            if dose is None and any(other.startswith(name + " ") for other in out):
                # A bare "insulin" in a sentence about the insulin prescribed on
                # the line above is the same drug, not a third one. Creating a
                # fact for it invented a drug and stole the sentence's hold rule
                # from the drug it belonged to.
                rule = _hold_rule(window)
                if rule and recent.startswith(name + " ") and not out[recent].hold_rule:
                    out[recent] = replace(out[recent], hold_rule=rule)
                continue
            head = _THEN_RE.split(window)[0]
            frequency = normalise_frequency(head)
            days, note = extract_duration(window)
            then_dose, then_freq, then_days, then_note = extract_second_phase(window)
            if not (has_cue or dose is not None or frequency):
                continue                  # a bare drug mention is not an instruction
            candidate = Medication(
                name=name, dose=dose, dose_max=dose_max, unit=unit,
                route=_route(window), frequency=frequency,
                duration_days=days, duration_note=note,
                max_per_day=ceiling, then_dose=then_dose,
                then_frequency=then_freq, then_duration_days=then_days,
                then_duration_note=then_note, hold_rule=_hold_rule(window),
                span=segment.strip())
            existing = out.get(name)
            out[name] = candidate if existing is None else _merge_med(existing, candidate)
            recent, recent_line = name, line_no
    return list(out.values())


def _merge_med(first: Medication, second: Medication) -> Medication:
    """One drug mentioned twice is one instruction. Fill blanks from the second
    mention rather than letting a later partial restatement overwrite a full one."""
    def pick(a, b):
        return a if a not in (None, "") else b
    return Medication(
        name=first.name,
        dose=pick(first.dose, second.dose),
        dose_max=pick(first.dose_max, second.dose_max),
        unit=pick(first.unit, second.unit),
        route=pick(first.route, second.route),
        frequency=pick(first.frequency, second.frequency),
        duration_days=pick(first.duration_days, second.duration_days),
        duration_note=pick(first.duration_note, second.duration_note),
        max_per_day=pick(first.max_per_day, second.max_per_day),
        then_dose=pick(first.then_dose, second.then_dose),
        then_frequency=pick(first.then_frequency, second.then_frequency),
        then_duration_days=pick(first.then_duration_days, second.then_duration_days),
        then_duration_note=pick(first.then_duration_note, second.then_duration_note),
        hold_rule=pick(first.hold_rule, second.hold_rule),
        span=first.span)


# ---------------------------------------------------------------------------
# follow-ups

_FOLLOWUP_CUE = re.compile(
    # "\bsee\b" rather than "see your": a simplifier that rewrites "follow up
    # with cardiology" as "see cardiology" has kept the appointment, and the
    # narrower cue scored that as a deleted appointment.
    r"\bfollow[\s-]?up\b|\bappointment\b|\bsee\b|\bvisit\b"
    # "schedule" as a verb only. The bare word matched "resume the normal
    # schedule" in a missed-dose instruction and invented an appointment.
    r"|\bto\s+schedule\b|\bschedule\s+(?:an?\s+|your\s+)?(?:appointment|visit|follow)"
    r"|\bcome\s+back\b|\breturn\s+to\s+(?:clinic|your|work)\b"
    r"|\bcheck[\s-]?up\b|\brecheck\b|\bbe\s+seen\b|\bgo\s+over\s+your\b"
    # An appointment the clinic makes rather than the patient: "the
    # anticoagulation clinic will check your INR in 3 days", "physical therapy
    # begins in 2 days", "the programme will contact you within 2 weeks".
    r"|\bwill\s+(?:check|see|call|contact|phone)\b|\bbegins?\s+in\b"
    r"|\bhas\s+been\s+(?:placed|ordered|arranged)\b"
    # Both wordings: the glossary rewrites "indicated" as "needed", and with
    # only the clinical form here a scheduled chest x-ray disappeared.
    r"|\bis\s+(?:required|indicated|needed|due|scheduled)\s+in\b", re.I)

# "Contact your primary care physician within 1 week" is an appointment even
# though it names none of the cues above. A contact verb plus a named role plus
# a time window is enough, and all three are required so that "call the office
# if you have a fever" stays a red flag.
_CONTACT_CUE = re.compile(r"\b(?:contact|call|phone|telephone)\b", re.I)


def _roles(segment: str) -> list[str]:
    return matching_keys(FOLLOWUP_ROLES, segment)


def extract_follow_ups(text: str) -> list[FollowUp]:
    seen: dict[tuple[str, int | None], str] = {}
    for segment in segments(text):
        if is_negated(segment) or segment.rstrip().endswith(":"):
            # A bare section header ("Follow-up:") announces appointments; it is
            # not one itself.
            continue
        if _action(segment) == "emergency":
            # "Return to the emergency department immediately if ..." is a
            # warning sign, not an appointment. Without this the extractor
            # invented an undated appointment from every such sentence.
            continue
        window = extract_window(segment)
        roles = _roles(segment)
        if len(segment.split()) <= 3 and window is None and not roles:
            # A standalone section heading, "FOLLOW-UP" on its own line with no
            # colon. It was producing an undated appointment with an unnamed
            # clinic in every note that uses that heading style.
            continue
        if _FOLLOWUP_CUE.search(segment):
            pass
        elif _CONTACT_CUE.search(segment) and roles and window is not None:
            pass
        else:
            continue
        if not roles:
            # A named clinician with no role word lands in one generic bucket,
            # the same bucket the rewrite's "your doctor" lands in. Without this
            # a name-to-role simplification would score as a false error.
            roles = ["clinic"]
        for role in roles:
            seen.setdefault((role, window), segment.strip())
    return assign_ordinals([(role, window, span)
                            for (role, window), span in seen.items()])


def assign_ordinals(rows: list[tuple[str, int | None, str]]) -> list[FollowUp]:
    """Number repeated roles in window order, soonest first.

    Also drops a windowless duplicate of a role that has a dated appointment:
    "follow up with cardiology" and "follow up with cardiology in 7 days" are
    one appointment mentioned twice, not two.
    """
    dated = {role for role, window, _ in rows if window is not None}
    kept = [row for row in rows if row[1] is not None or row[0] not in dated]
    out: list[FollowUp] = []
    for role in sorted({row[0] for row in kept}):
        group = sorted((row for row in kept if row[0] == role),
                       key=lambda row: (row[1] is None, row[1] or 0))
        for index, (_, window, span) in enumerate(group, start=1):
            out.append(FollowUp(who=role, within_days=window, ordinal=index,
                                span=span))
    return out


# ---------------------------------------------------------------------------
# red flags

_CEILING_WORDS = r"over|above|higher\s+than|more\s+than|greater\s+than|at\s+or\s+above|exceeds?|>=?"
_FLOOR_WORDS = r"below|under|less\s+than|lower\s+than|fewer\s+than|at\s+or\s+below|<=?"
_THRESHOLD_UNITS = (r"°\s*f|°\s*c|\bf\b|fahrenheit|\bc\b|celsius|degrees?"
                    r"|pounds?|lbs?|kilograms?|kgs?|mg/dl|mmol/l|mg\b"
                    r"|minutes?|mins?|hours?|pads?|times?|beats?|litres?|liters?")
_THRESHOLD_RE = re.compile(
    rf"(?P<dir>{_CEILING_WORDS}|{_FLOOR_WORDS})\s*"
    rf"(?P<num>{NUM})\s*(?P<unit>{_THRESHOLD_UNITS})?"
    # The denominator, when there is one. "Over 3 pounds" is not a warning
    # sign; "over 3 pounds in one day" is. Two notes weigh gain that way and
    # one counts pads an hour, and without this the value looked complete
    # while the part that made it a threshold was gone.
    rf"(?:\s*(?:in|per|an|a|each)\s+(?:a\s+|one\s+|24\s+|twenty[\s-]four\s+)?"
    rf"(?P<per>day|hour|week|night|24\s*hours?))?", re.I)

_CEILING_RE = re.compile(_CEILING_WORDS, re.I)


def _threshold_text(match: re.Match) -> str:
    """Canonical threshold, keeping the comparison direction and the unit.

    "over 101 F" and "below 70 mg/dL" are a ceiling and a floor. Stored as the
    bare numbers 101 and 70 they were indistinguishable, so a rule about
    hypoglycaemia read as a rule about hyperglycaemia, and a rewrite could
    invert one without the diff noticing.
    """
    direction = "over" if _CEILING_RE.fullmatch(match.group("dir").strip()) else "below"
    unit = (match.group("unit") or "").strip().lower()
    unit = {"f": "F", "fahrenheit": "F", "°f": "F", "c": "C", "celsius": "C",
            "°c": "C", "lb": "pounds", "lbs": "pounds", "pound": "pounds",
            "kg": "kilograms", "kgs": "kilograms", "kilogram": "kilograms",
            "mg/dl": "mg/dL", "mmol/l": "mmol/L", "min": "minutes",
            "mins": "minutes", "minute": "minutes", "hour": "hours",
            "pad": "pads", "time": "times", "beat": "beats",
            "litre": "L", "litres": "L", "liter": "L", "liters": "L",
            "degree": "degrees"}.get(unit, unit)
    out = f"{direction} {_fmt(match.group('num'))}" + (f" {unit}" if unit else "")
    per = (match.groupdict().get("per") or "").strip().lower()
    if per:
        per = "day" if per.replace(" ", "").startswith("24hour") else per
        out += f" per {per}"
    return out


_URGENCY = re.compile(
    r"\bimmediately\b|\bright\s+away\b|\bat\s+once\b|\bstraight\s+away\b"
    r"|\bwithout\s+delay\b|\burgent", re.I)


def _action(segment: str) -> str:
    found = matching_keys(ACTIONS, segment)
    if not found:
        return ""
    best = max(found, key=lambda a: ACTION_SEVERITY[a])
    if best == "call" and _URGENCY.search(segment):
        # "Call the obstetric service immediately" is not the same instruction
        # as "call the office". The urgency word can sit anywhere in the
        # sentence, so it is matched separately rather than as a fixed phrase.
        return "call urgently"
    return best


def _drop_subsumed(found: list[str]) -> list[str]:
    covered = {general for name in found
               for general in SYMPTOM_SUBSUMES.get(name, ())}
    return [name for name in found if name not in covered]


def _symptoms(segment: str) -> list[str]:
    return _drop_subsumed(matching_keys(SYMPTOMS, segment))


def _symptom_positions(segment: str) -> list[tuple[int, str]]:
    """Where each symptom is mentioned, so a threshold can be tied to one.

    A warning list is often one sentence: "temperature above 101 F, intractable
    vomiting, worsening flank pain, hypotension, or decreased urine output".
    Attaching that sentence's single threshold to every symptom in it produced
    facts like "fainting, threshold 101", and then any edit anywhere in the
    sentence reported five corruptions instead of one.
    """
    out = []
    for name, aliases in SYMPTOMS.items():
        match = alias_pattern(tuple(aliases)).search(segment)
        if match:
            out.append((match.start(), name))
    keep = set(_drop_subsumed([name for _, name in out]))
    return sorted((position, name) for position, name in out if name in keep)


_CONDITIONAL = re.compile(
    r"\bif\b|\bwhen\b|\bany\s+of\s+(?:the\s+)?follow|\bfor\s+any\b"
    r"|\bshould\s+you\b|\bin\s+case\s+of\b|\bwatch\s+for\b"
    r"|\bsigns?\s+of\b|\bsymptoms?\s+of\b|\bdevelop\b|\bexperience\b"
    r"|\bnotice\b"
    # "Call the office for fever over 101" is the commonest form of all and has
    # no conditional word in it. An action verb followed by "for" is the
    # condition. The first version of this gate missed every note written that
    # way, which was five of fifteen.
    r"|\b(?:call|contact|phone|notify|return|present|come|go|seek)\b[^.;:]{0,40}?\bfor\b",
    re.I)


def extract_red_flags(text: str) -> list[RedFlag]:
    out: dict[str, RedFlag] = {}
    carried_action = ""
    for segment in segments(text):
        action = _action(segment)
        if segment.rstrip().endswith(":"):
            # In "Call the office right away for any of the following:" the
            # action belongs to the bullets underneath.
            carried_action = action
            if not _symptoms(segment):
                continue
        effective = action or carried_action
        if not effective:
            continue
        if not (_CONDITIONAL.search(segment) or carried_action):
            continue
        found = _symptom_positions(segment)
        thresholds = _attach_thresholds(segment, found)
        for _, symptom in found:
            existing = out.get(symptom)
            threshold = thresholds.get(symptom, "")
            if existing is None or ACTION_SEVERITY[effective] > ACTION_SEVERITY[existing.action]:
                out[symptom] = RedFlag(symptom=symptom, action=effective,
                                       threshold=threshold, span=segment.strip())
            elif threshold and not existing.threshold:
                out[symptom] = replace(existing, threshold=threshold)
    return list(out.values())


def _attach_thresholds(segment: str, found: list[tuple[int, str]]) -> dict[str, str]:
    """Give each threshold to the symptom it follows, not to the whole sentence."""
    out: dict[str, str] = {}
    for match in _THRESHOLD_RE.finditer(segment):
        if parse_number(match.group("num")) is None:
            continue
        owners = [name for position, name in found if position <= match.start()]
        if owners:
            out.setdefault(owners[-1], _threshold_text(match))
    return out


# ---------------------------------------------------------------------------
# restrictions

_WEIGHT_LIMIT = re.compile(
    rf"({NUM})\s*(pounds?|lbs?|kilograms?|kgs?)\b", re.I)


def _activities(segment: str) -> list[str]:
    return matching_keys(ACTIVITIES, segment)


_WAIT_RE = re.compile(
    rf"\bwait\s+(?:(?:at\s+least|about|around)\s+)?({NUM})\s*(day|week|month)s?", re.I)

# The clause that outlives the stated number. "Do not drive for 5 days, and not
# at all while taking oxycodone" has two limits and the second is the binding
# one; four notes in the corpus are written this way.
_CONDITION_RULES: list[tuple[str, str]] = [
    (rf"while\s+(?:you\s+are\s+)?(?:still\s+)?(?:tak|us)\w*\s+(?:the\s+|your\s+|any\s+)?"
     rf"({_DRUG_ALTERNATION}|opioids?|narcotics?|pain\s+medicine|sedating\s+medication)",
     "while taking {0}"),
    (r"\bunassisted\b", "unassisted"),
    (r"\bunsupervised\b", "unsupervised"),
    (r"until\s+you\s+are\s+symptom\s+free", "until symptom free"),
    (r"\bat\s+all\b(?=[^.;]*while)", ""),
]


def _condition(segment: str) -> str:
    for pattern, template in _CONDITION_RULES:
        match = re.search(pattern, segment, re.I)
        if not match or not template:
            continue
        if match.groups():
            return template.format(canonical_drug(match.group(1)).lower())
        return template
    return ""


def extract_restrictions(text: str) -> list[Restriction]:
    out: dict[str, Restriction] = {}
    for segment in segments(text):
        negated = is_negated(segment)
        limited = bool(_LIMIT.search(segment))
        if not (negated or limited):
            continue
        days, note = extract_duration(segment)
        if days is None and not note:
            wait = _WAIT_RE.search(segment)
            if wait:
                count = parse_number(wait.group(1))
                if count is not None:
                    days = _int_or_none(count * _PERIOD_DAYS[wait.group(2).lower()])
        weight = _WEIGHT_LIMIT.search(segment)
        limit = ""
        if weight:
            unit = "pounds" if weight.group(2).lower().startswith(("p", "lb")) else "kilograms"
            limit = f"{_fmt(weight.group(1))} {unit}"

        names = _activities(segment)
        covered = {alias.rstrip("*") for key in names for alias in ACTIVITIES[key]}
        for drug, _ in _drug_windows(segment):
            # "Do not take ibuprofen" is already the `nsaids` restriction, so it
            # must not also become a separate "taking ibuprofen" one.
            if drug not in covered:
                names.append(f"taking {drug}")
        condition = _condition(segment)
        for name in names:
            # A "while taking <drug>" clause does not constrain the drug it
            # names; "do not drive while taking oxycodone" restricts driving.
            own_condition = "" if name == f"taking {condition[13:]}" else condition
            candidate = Restriction(activity=name, duration_days=days,
                                    duration_note=note,
                                    limit=limit if name == "lifting" else "",
                                    condition=own_condition,
                                    span=segment.strip())
            existing = out.get(name)
            if existing is None:
                out[name] = candidate
            else:
                out[name] = Restriction(
                    activity=name,
                    duration_days=existing.duration_days if existing.duration_days is not None else days,
                    duration_note=existing.duration_note or note,
                    limit=existing.limit or candidate.limit,
                    condition=existing.condition or own_condition,
                    span=existing.span)
    return list(out.values())


# ---------------------------------------------------------------------------

def extract(text: str) -> FactSet:
    """The one entry point. Called on the original and on every rewrite."""
    return FactSet(
        medications=sorted(extract_medications(text), key=lambda m: m.name),
        follow_ups=sorted(extract_follow_ups(text), key=lambda f: (f.who, f.ordinal)),
        red_flags=sorted(extract_red_flags(text), key=lambda r: r.symptom),
        restrictions=sorted(extract_restrictions(text), key=lambda r: r.activity),
    )
