"""Two simplifiers, and a note on why there are two.

`GlossarySimplifier` runs offline, deterministically, with no API key. It is
the baseline, and it is deliberately a weak one: it substitutes words and
expands abbreviations and cannot restructure a sentence. Its value is that
anyone can reproduce every number in this project's README without credentials,
and that it exercises the full generate-verify-repair loop.

`ClaudeSimplifier` calls the Anthropic API and is what you would actually ship.
It restructures prose, which is why it is better at readability and why it can
lose a fact, which is the entire reason the verifier exists.

Neither simplifier is allowed to see the extracted fact set when it writes its
first draft. If it did, the measurement would be circular: a rewrite generated
from a list of facts cannot lose a fact on that list, and the fidelity score
would be measuring the renderer rather than the rewrite. The fact set reaches a
simplifier only on a repair round, and only as the specific discrepancies the
verifier found, which is what the spec's repair step asks for.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .glossary import (ABBREVIATIONS, ABBREVIATIONS_UPPER, GLOSSES, PHRASES,
                       WORDS)
from .vocab import DRUGS


class Simplifier(Protocol):
    name: str

    def simplify(self, text: str, target_grade: float,
                 constraints: list[str] | None = None) -> str:
        ...


# ---------------------------------------------------------------------------

def _preserve_case(replacement: str, original: str) -> str:
    """Match the original's capitalisation, with one exception.

    An ALL-CAPS source keeps its caps only when the replacement is a single
    word, so the heading "MEDICATIONS" becomes "MEDICINES" while the
    abbreviation "PO" becomes "by mouth" rather than "BY MOUTH". A multi-word
    expansion of an uppercase abbreviation is lowercased outright, because
    title-casing it produced "5 mg By mouth At bedtime" in the middle of a
    sentence.
    """
    if len(original) > 1 and original.isupper() and original.isalpha():
        return replacement.upper() if " " not in replacement else replacement
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# Hour-interval shorthand, which no fixed table can cover: q4h, q6h, q4-6h,
# Q8H. Expanded before the tables so the result is plain English by the time
# the word substitutions run.
_QNH = re.compile(r"(?<![A-Za-z0-9])q\.?\s*(\d+)\s*(?:-|to|–)\s*(\d+)\s*h\b"
                  r"|(?<![A-Za-z0-9])q\.?\s*(\d+)\s*h\b", re.I)


def _expand_hour_intervals(text: str) -> str:
    def swap(match: re.Match) -> str:
        if match.group(1):
            return f"every {match.group(1)} to {match.group(2)} hours"
        return f"every {match.group(3)} hours"
    return _QNH.sub(swap, text)


# "PRN severe pain" needs a preposition that "if you need it" does not supply.
# Inserted after expansion rather than encoded in the table, because the table
# entry is also correct on its own ("take one PRN").
_PRN_DANGLING = re.compile(
    r"(if you need (?:it|them))\s+(?=(?:severe|mild|moderate|bad|breakthrough|"
    r"nausea|pain|fever|anxiety|constipation|itching|sleep|cough)\b)", re.I)


def _fix_prn_grammar(text: str) -> str:
    return _PRN_DANGLING.sub(r"\1 for ", text)


_MULTIWORD_DRUGS = tuple(name for name in DRUGS if " " in name or "-" in name)


def _continues_a_drug_name(term: str, tail: str) -> bool:
    """True when `term` plus the next word is itself a known drug name."""
    nxt = re.match(r"[\s-]+([A-Za-z]+)", tail)
    if not nxt:
        return False
    longer = f"{term.lower()} {nxt.group(1).lower()}"
    hyphen = f"{term.lower()}-{nxt.group(1).lower()}"
    return any(name in (longer, hyphen) for name in _MULTIWORD_DRUGS)


_PLACEHOLDER = "\u0000META%d\u0000"


def _split_metadata(text: str) -> tuple[str, list[str]]:
    keep: list[str] = []
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            lines.append(_PLACEHOLDER % len(keep))
            keep.append(line)
        else:
            lines.append(line)
    return "\n".join(lines), keep


def _restore_metadata(text: str, keep: list[str]) -> str:
    for index, line in enumerate(keep):
        text = text.replace(_PLACEHOLDER % index, line)
    return text


def _substitute(text: str, table: dict[str, str], *, ignore_case: bool = True) -> str:
    if not table:
        return text
    keys = sorted(table, key=len, reverse=True)
    pattern = "|".join(re.escape(key) for key in keys)
    # \b does not fire next to a period or slash, so the boundary is written out.
    regex = re.compile(rf"(?<![A-Za-z0-9])({pattern})(?![A-Za-z0-9])",
                       re.I if ignore_case else 0)

    def swap(match: re.Match) -> str:
        found = match.group(1)
        value = table.get(found) or table.get(found.lower()) or table.get(found.upper())
        if value is None:
            return found
        return _preserve_case(value, found) if value else ""

    return regex.sub(swap, text)


class GlossarySimplifier:
    """Lexical substitution plus abbreviation expansion plus sentence splitting.

    Text in, text out. It never consults the extractor, so its fidelity score
    is a real measurement of a real transformation rather than a round trip.
    """

    name = "glossary"

    def __init__(self, max_sentence_words: int = 22) -> None:
        self.max_sentence_words = max_sentence_words

    def simplify(self, text: str, target_grade: float = 6.0,
                 constraints: list[str] | None = None) -> str:
        # Metadata lines are left alone. A note id like "afib-warfarin" contains
        # a drug name, and substituting inside it produced the comment
        # "# note: afib-warfarin (a blood thinner)".
        body, keep = _split_metadata(text)
        out = _expand_hour_intervals(body)
        out = _substitute(out, ABBREVIATIONS_UPPER, ignore_case=False)
        out = _substitute(out, ABBREVIATIONS)
        out = _substitute(out, PHRASES)
        out = _substitute(out, WORDS)
        out = _fix_prn_grammar(out)
        out = self._gloss(out)
        out = self._split_sentences(out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        out = _restore_metadata(out, keep)
        if constraints:
            out = self._repair(out, text, constraints)
        return out

    # -- gloss on first use -------------------------------------------------
    def _gloss(self, text: str) -> str:
        for term in sorted(GLOSSES, key=len, reverse=True):
            gloss = GLOSSES[term]
            for match in re.finditer(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])",
                                     text, re.I):
                tail = text[match.end():match.end() + 48]
                if tail.lstrip().startswith("("):
                    break
                # "Metoprolol" is a prefix of "metoprolol tartrate". Glossing
                # the prefix split the drug name in two, and the extractor then
                # reported one missing drug and one invented one.
                if _continues_a_drug_name(term, tail):
                    continue
                text = text[:match.end()] + f" ({gloss})" + text[match.end():]
                break
        return text

    # -- sentence splitting -------------------------------------------------
    def _split_sentences(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            lines.append(self._split_line(line))
        return "\n".join(lines)

    def _split_line(self, line: str) -> str:
        if len(line.split()) <= self.max_sentence_words:
            return line
        # Semicolons are already clause boundaries, so splitting there cannot
        # change meaning. Nothing else is split, because splitting at "and" or
        # a comma can detach a qualifier from what it qualifies, and a detached
        # "as needed" is exactly the kind of error this project exists to count.
        parts = [part.strip() for part in line.split(";") if part.strip()]
        if len(parts) == 1:
            return line
        out = []
        for part in parts:
            if not part.endswith((".", "!", "?", ":")):
                part += "."
            out.append(part[:1].upper() + part[1:])
        return " ".join(out)

    # -- repair -------------------------------------------------------------
    def _repair(self, rewrite: str, original: str, constraints: list[str]) -> str:
        """A substitution table cannot reason, so its repair is to restate.

        Lines of the original whose facts the verifier reported missing are
        appended verbatim under a heading. It is a blunt fix, and it is honest
        about being one: the loop converges, the facts come back, and the
        readability score pays for it. The README reports that cost rather than
        hiding it, because a repair strategy that trades reading level for
        fidelity is a real engineering choice and the numbers should show which
        way it went.
        """
        subjects = set()
        for message in constraints:
            for word in re.findall(r"[a-z][a-z\-]{3,}", message.lower()):
                subjects.add(word)
        wanted = []
        for line in original.splitlines():
            stripped = line.strip(" -*•\t")
            if not stripped or stripped.endswith(":"):
                continue
            tokens = set(re.findall(r"[a-z][a-z\-]{3,}", stripped.lower()))
            if tokens & subjects and stripped not in rewrite:
                wanted.append(stripped)
        if not wanted:
            return rewrite
        return (rewrite.rstrip()
                + "\n\nAlso, exactly as your care team wrote it:\n"
                + "\n".join(f"- {line}" for line in wanted) + "\n")


# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You rewrite hospital discharge instructions so that a person who has just \
woken up from anesthesia can follow them.

Rules you must not break:
- Every medication keeps its exact name, dose, unit, route, frequency and \
duration. Never replace a number with a word like "some" or "a few".
- Never drop "as needed" or "only if", and never add it where it was not.
- Every appointment keeps who it is with and the time window.
- Every warning sign keeps the symptom, the threshold number, and whether the \
person should call or go to the emergency room. Never downgrade an emergency \
instruction to a phone call.
- Every restriction keeps the activity, the duration, and any weight limit.
- Add nothing. No reassurance, no general advice, no instruction that was not \
in the original.

Style:
- Target a US grade {target_grade} reading level.
- Short sentences. Everyday words. Second person.
- Keep drug names, then say in parentheses what the drug is for.
- Keep the original's section structure and bullet lists.

Output the rewritten instructions only. No preamble, no commentary."""

REPAIR_PROMPT = """\
Your previous rewrite lost or changed information. A checker compared the \
facts in the original against the facts in your rewrite and found these \
problems:

{problems}

Rewrite the instructions again from the original below. Keep everything your \
previous attempt got right, fix every problem listed above, and introduce \
nothing new."""


DEFAULT_MODEL = "claude-sonnet-5-5"


class IncompleteRewrite(RuntimeError):
    """The model returned something that must not be scored as a rewrite."""


class ClaudeSimplifier:
    """Anthropic API simplifier.

    The key is read from ANTHROPIC_API_KEY and never written to disk, logged,
    or included in any output file. Without a key this class raises on
    construction, which is why every test and the whole offline evaluation path
    uses GlossarySimplifier or a recorded fixture instead.

    Two settings that look like tuning and are actually correctness:

    `max_tokens` is generous because current models think before answering and
    the thinking counts against it. At the old cap of 4,096 the thinking could
    spend the whole budget and return no text, and an empty rewrite scores as a
    rewrite that omitted every fact. The loop would have reported a model that
    loses everything, which is a number about the cap, not the model.

    No sampling parameters are sent. API version 2026-10-21 rejects
    `temperature`, `top_p` and `top_k` with a 400.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 32000,
                 effort: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "pip install anthropic to use the model-backed simplifier") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run the offline evaluation with "
                "--simplifier glossary, or export the key to reproduce the "
                "model-backed numbers.")
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.name = f"claude ({model}{', effort ' + effort if effort else ''})"

    def simplify(self, text: str, target_grade: float = 6.0,
                 constraints: list[str] | None = None) -> str:
        if constraints:
            problems = "\n".join(f"- {item}" for item in constraints)
            user = (REPAIR_PROMPT.format(problems=problems)
                    + "\n\nORIGINAL INSTRUCTIONS:\n" + text)
        else:
            user = "ORIGINAL INSTRUCTIONS:\n" + text
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT.format(target_grade=target_grade),
            messages=[{"role": "user", "content": user}],
        )
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        response = self._client.messages.create(**request)
        return rewrite_text(response)


def rewrite_text(response) -> str:
    """The text of a response, or an error if it is not a complete rewrite.

    A truncated or refused response is never returned as text, because the
    loop would score it: a half rewrite reads as a rewrite that dropped the
    second half of the note, and that would be reported as the model's
    fidelity.
    """
    stop = getattr(response, "stop_reason", None)
    text = "".join(block.text for block in response.content
                   if getattr(block, "type", "") == "text").strip()
    if stop == "max_tokens":
        raise IncompleteRewrite(
            "the response hit max_tokens before finishing; raise max_tokens "
            "rather than scoring a truncated rewrite")
    if stop == "refusal":
        raise IncompleteRewrite("the model declined this note")
    if not text:
        raise IncompleteRewrite(f"the response contained no text (stop_reason={stop})")
    return text


class RecordedSimplifier:
    """Replays saved model output so the model-backed path is testable offline.

    Fixtures live in tests/fixtures/ and were produced by ClaudeSimplifier. They
    are kept so that `pytest` needs no key and no network, and so the loop's
    behaviour on genuinely model-written text is covered by tests rather than
    only by the substitution baseline.
    """

    name = "recorded"

    def __init__(self, rounds: dict[str, list[str]]) -> None:
        self._rounds = rounds
        self._used: dict[str, int] = {}

    def simplify(self, text: str, target_grade: float = 6.0,
                 constraints: list[str] | None = None) -> str:
        key = _fixture_key(text)
        attempts = self._rounds.get(key)
        if attempts is None:
            raise KeyError(f"no recorded output for note {key}")
        index = min(self._used.get(key, 0), len(attempts) - 1)
        self._used[key] = index + 1
        return attempts[index]


def _fixture_key(text: str) -> str:
    match = re.search(r"^#\s*note:\s*(\S+)", text, re.M)
    if match:
        return match.group(1)
    return re.sub(r"\s+", " ", text.strip())[:60]


class LossySimplifier:
    """The glossary simplifier plus one deliberate error, with known identity.

    The offline baseline is conservative enough that it loses nothing, so on its
    own it never exercises the repair round: every note converges in one pass.
    That makes the loop's most interesting behaviour untested and undemonstrated.

    This wrapper applies one of the mutations from `core.mutate` to the rewrite,
    so there is a real discrepancy for the verifier to catch and the repair step
    to fix, and the expected discrepancy is known in advance rather than
    discovered from the diff. It is a test double and a demo, labelled as one,
    and it is never used to produce a reported fidelity number.
    """

    name = "lossy"

    def __init__(self, mutation: str = "halve_dose") -> None:
        self.mutation = mutation
        self._base = GlossarySimplifier()
        self._rounds = 0

    def simplify(self, text: str, target_grade: float = 6.0,
                 constraints: list[str] | None = None) -> str:
        from .mutate import mutations_for
        rewrite = self._base.simplify(text, target_grade, constraints)
        self._rounds += 1
        if constraints:
            # The repair round is where it behaves: told what it broke, it stops
            # breaking it.
            return rewrite
        candidates = [m for m in mutations_for("lossy", rewrite)
                      if m.name == self.mutation]
        return candidates[0].text if candidates else rewrite


SIMPLIFIERS = {"glossary": GlossarySimplifier, "claude": ClaudeSimplifier,
               "lossy": LossySimplifier}
