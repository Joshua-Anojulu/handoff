"""Reading level, measured two ways, because one way is not enough.

Flesch-Kincaid counts syllables and sentence length. By that measure
"anticoagulant" (5 syllables) and "responsibility" (6 syllables) are
comparable words, and the second is harder. General-English frequency says
otherwise: on the Zipf scale "responsibility" sits at 4.75 and
"anticoagulant" at 2.42, a difference of more than two orders of magnitude in
how often a reader has met the word.

So this module reports two numbers:

  grade              Flesch-Kincaid grade level, the standard everyone quotes
  unexplained_terms  words rare in general English that the text uses without
                     saying what they mean

A rewrite can move the first and not the second, which is the failure the
second number exists to catch: shorten the sentences, keep the jargon, and
Flesch-Kincaid will congratulate you.

Drug names are counted separately. "Warfarin" is a rare word and there is
nothing to be done about it, because the alternative to naming the drug is not
naming the drug. Lumping it in with avoidable jargon would make the metric
reward unsafe rewrites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import strip_metadata
from .vocab import BRAND_TO_GENERIC, DRUGS

try:
    from wordfreq import zipf_frequency
    HAVE_WORDFREQ = True
except ImportError:                                     # pragma: no cover
    HAVE_WORDFREQ = False

    def zipf_frequency(word: str, lang: str = "en") -> float:  # type: ignore[misc]
        return 5.0


# Below this Zipf value a word is treated as rare in general English.
#
# Chosen, not inherited: 3.0 is where the clinical and plain vocabularies
# separate in the pairs this project cares about. `python -m core.readability`
# prints the measurement that justifies it, so the threshold can be argued
# with rather than taken on faith.
RARE_ZIPF = 3.0

# Words that are rare by frequency but are not the kind of jargon a rewrite
# should be penalised for: units, times, and the structural words of a dosing
# instruction.
NOT_JARGON = frozenset({
    "mg", "mcg", "ml", "mls", "tsp", "tbsp", "lb", "lbs", "kg", "iu",
    "am", "pm", "ok", "okay", "er", "ed", "pcp", "pt", "md", "rn", "np",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "teaspoon", "teaspoons", "tablespoon", "tablespoons", "milligram", "milligrams",
})

GLOSS_MARKERS = (
    r"which\s+means", r"that\s+means", r"also\s+called", r"also\s+known\s+as",
    r"this\s+is", r"these\s+are", r"a\s+kind\s+of", r"a\s+type\s+of",
    r"in\s+other\s+words", r"that\s+is,", r"meaning",
)
_GLOSS_RE = re.compile("|".join(GLOSS_MARKERS), re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")

_DRUG_WORDS = frozenset(
    part for name in (set(DRUGS) | set(BRAND_TO_GENERIC))
    for part in re.split(r"[\s\-]+", name.lower()))


def syllables(word: str) -> int:
    """Vowel-group heuristic. Not exact, but it is the same heuristic applied to
    both texts, so the comparison between them holds."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    if word.endswith(("les", "le")) and len(word) > 2 and word[-3] not in "aeiouy":
        count = max(count, 1)
    return max(count, 1)


def count_sentences(text: str) -> int:
    """Lines count as sentences when they carry no terminal punctuation, because
    a bulleted instruction is a sentence whatever the typography says."""
    total = 0
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•·]|\d+[.)])\s+", "", line).strip()
        if not line or line.endswith(":"):
            continue
        found = len([p for p in _SENTENCE_RE.split(line) if p.strip()])
        total += max(found, 1)
    return max(total, 1)


def flesch_kincaid_grade(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    sentence_count = count_sentences(text)
    syllable_count = sum(syllables(word) for word in words)
    return round(0.39 * (len(words) / sentence_count)
                 + 11.8 * (syllable_count / len(words)) - 15.59, 2)


def _is_explained(word: str, text: str) -> bool:
    """True when the text says what the word means near its first use."""
    for match in re.finditer(rf"\b{re.escape(word)}\b", text, re.I):
        window = text[match.end():match.end() + 140]
        before = text[max(0, match.start() - 90):match.start()]
        if re.match(r"\s*\(", window):
            return True
        if _GLOSS_RE.search(window[:90]) or _GLOSS_RE.search(before):
            return True
        if re.match(r"\s*(?:,|\u2014|-{1,2}|:)\s*(?:a|an|the|your|this)\b", window):
            return True
        if re.search(r"\(\s*$", before):
            return True
    return False


@dataclass
class Readability:
    grade: float
    words: int
    sentences: int
    unexplained_terms: list[str] = field(default_factory=list)
    explained_terms: list[str] = field(default_factory=list)
    drug_terms: list[str] = field(default_factory=list)

    @property
    def unexplained_count(self) -> int:
        return len(self.unexplained_terms)

    def summary(self) -> str:
        return (f"grade {self.grade}, {self.unexplained_count} unexplained rare terms"
                + (f" ({', '.join(self.unexplained_terms[:6])})"
                   if self.unexplained_terms else ""))

    def as_dict(self) -> dict:
        return {"grade": self.grade, "words": self.words, "sentences": self.sentences,
                "unexplained_terms": self.unexplained_terms,
                "explained_terms": self.explained_terms,
                "drug_terms": self.drug_terms}


def assess(text: str, rare_zipf: float = RARE_ZIPF) -> Readability:
    text = strip_metadata(text)
    words = _WORD_RE.findall(text)
    seen, unexplained, explained, drugs = set(), [], [], []
    for raw in words:
        word = raw.lower().strip("'-")
        if word in seen or len(word) < 3 or word in NOT_JARGON:
            continue
        seen.add(word)
        if zipf_frequency(word, "en") >= rare_zipf:
            continue
        if word in _DRUG_WORDS:
            drugs.append(word)
        elif _is_explained(word, text):
            explained.append(word)
        else:
            unexplained.append(word)
    return Readability(grade=flesch_kincaid_grade(text), words=len(words),
                       sentences=count_sentences(text),
                       unexplained_terms=sorted(unexplained),
                       explained_terms=sorted(explained),
                       drug_terms=sorted(drugs))


# ---------------------------------------------------------------------------

CLINICAL_SAMPLE = [
    "anticoagulant", "dyspnea", "edema", "erythema", "purulent", "febrile",
    "incision", "ambulate", "analgesic", "antiemetic", "prophylaxis",
    "sublingual", "subcutaneous", "contraindicated", "exacerbation",
    "hypoglycemia", "dehiscence", "syncope", "palpitations", "diuretic",
]
PLAIN_SAMPLE = [
    "responsibility", "breathless", "swelling", "redness", "pus", "fever",
    "cut", "walk", "painkiller", "sickness", "prevention", "tongue", "skin",
    "unsafe", "flareup", "sugar", "opening", "faint", "pounding", "water",
]


def threshold_report(rare_zipf: float = RARE_ZIPF) -> str:
    """Why 3.0. Printed rather than asserted, so the number can be checked."""
    if not HAVE_WORDFREQ:                               # pragma: no cover
        return "wordfreq is not installed; install it to reproduce this table."
    rows = ["Zipf frequency in general English (higher = more familiar)",
            f"threshold = {rare_zipf}", "",
            f"{'clinical term':<18}{'zipf':>6}   {'plain term':<18}{'zipf':>6}"]
    clinical_scores, plain_scores = [], []
    for clinical, plain in zip(CLINICAL_SAMPLE, PLAIN_SAMPLE):
        a = zipf_frequency(clinical, "en")
        b = zipf_frequency(plain, "en")
        clinical_scores.append(a)
        plain_scores.append(b)
        rows.append(f"{clinical:<18}{a:>6.2f}   {plain:<18}{b:>6.2f}")
    below = sum(1 for s in clinical_scores if s < rare_zipf)
    above = sum(1 for s in plain_scores if s >= rare_zipf)
    rows += ["",
             f"clinical terms below the threshold: {below}/{len(clinical_scores)}",
             f"plain terms at or above it:         {above}/{len(plain_scores)}",
             "",
             "Flesch-Kincaid cannot make this distinction: it reads syllables,",
             "and 'anticoagulant' has fewer of them than 'responsibility'."]
    return "\n".join(rows)


if __name__ == "__main__":                              # pragma: no cover
    print(threshold_report())
