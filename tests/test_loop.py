"""Tests for the diff, the loop, and the readability measures.

The diff tests are mostly about one thing: that a changed value is reported as
a corruption rather than as an omission plus a fabrication. That distinction is
the whole reason the schema separates identity from value, and it is what makes
the most dangerous failure mode visible instead of hiding it inside the least
dangerous count.
"""
from __future__ import annotations

import pytest

from core.diff import CORRUPTION_WEIGHT, aggregate, diff
from core.extract import extract
from core.loop import run
from core.readability import (assess, flesch_kincaid_grade, syllables,
                              threshold_report)
from core.schema import FactSet, Medication
from core.simplify import GlossarySimplifier, LossySimplifier

ORIGINAL = """\
- Cephalexin 500 mg PO QID for 7 days.
- Oxycodone 5 mg PO q6h PRN pain.
- Follow up with your surgeon in 14 days.
- No lifting over 10 pounds for 4 weeks.
Call 911 for chest pain.
"""


def rewrite(text: str):
    return diff(extract(ORIGINAL), extract(text))


# --- the three classes -----------------------------------------------------

def test_dropped_instruction_is_an_omission():
    result = rewrite(ORIGINAL.replace("- Follow up with your surgeon in 14 days.\n", ""))
    assert [d.identity for d in result.omissions] == [("surgeon", 1)]
    assert not result.corruptions and not result.fabrications


def test_changed_dose_is_one_corruption_not_two_facts():
    """The point of separating identity from value. Keyed on the whole fact,
    a halved dose would read as one omission plus one fabrication and the
    dangerous case would vanish into the harmless counts."""
    result = rewrite(ORIGINAL.replace("500 mg", "250 mg"))
    assert len(result.corruptions) == 1
    assert result.corruptions[0].field == "dose"
    assert (result.corruptions[0].was, result.corruptions[0].now) == (500.0, 250.0)
    assert not result.omissions and not result.fabrications


def test_added_instruction_is_a_fabrication():
    result = rewrite(ORIGINAL + "- Amiodarone 200 mg PO QD.\n")
    assert [d.identity for d in result.fabrications] == [("amiodarone",)]


def test_dropped_prn_is_a_corruption_of_frequency():
    result = rewrite(ORIGINAL.replace("q6h PRN pain", "q6h"))
    assert [d.field for d in result.corruptions] == ["frequency"]


def test_dropped_value_counts_as_a_corruption():
    """A stated constraint that is no longer stated is a dropped constraint,
    not a rewording."""
    result = rewrite(ORIGINAL.replace(" for 7 days", ""))
    assert any(d.field == "duration (days)" for d in result.corruptions)


def test_route_is_wording_not_safety():
    result = rewrite(ORIGINAL.replace(" PO ", " "))
    assert not result.corruptions
    assert all(d.field == "route" for d in result.softenings)
    assert result.fidelity == 1.0


def test_downgraded_emergency_is_a_corruption_with_direction():
    result = rewrite(ORIGINAL.replace("Call 911", "Call the office"))
    corruption = next(d for d in result.corruptions if d.fact_kind == "red_flag")
    assert (corruption.was, corruption.now) == ("emergency", "call")
    assert "less than" in corruption.note


# --- the score -------------------------------------------------------------

def test_perfect_rewrite_scores_one():
    assert rewrite(ORIGINAL).fidelity == 1.0


def test_fidelity_is_errors_over_original_facts():
    result = rewrite(ORIGINAL.replace("500 mg", "250 mg"))
    assert result.fidelity == pytest.approx(1 - 1 / result.original_count)


def test_corruptions_weigh_more_than_omissions():
    corrupted = rewrite(ORIGINAL.replace("500 mg", "250 mg"))
    omitted = rewrite(ORIGINAL.replace(
        "- Follow up with your surgeon in 14 days.\n", ""))
    assert corrupted.fidelity == pytest.approx(omitted.fidelity)
    assert corrupted.weighted_fidelity < omitted.weighted_fidelity
    assert CORRUPTION_WEIGHT > 1


def test_fidelity_can_go_below_zero():
    """Fabrication has no upper bound while the other two are capped by the
    size of the original, so the score is left uncapped rather than clamped."""
    drugs = ("amiodarone", "clozapine", "colchicine", "allopurinol", "celecoxib",
             "cimetidine", "clozapine", "bosentan", "buspirone", "caffeine")
    invented = "\n".join(f"- {name.title()} 200 mg PO QD." for name in drugs)
    assert rewrite(ORIGINAL + invented).fidelity < 0


def test_empty_original():
    assert diff(FactSet(), FactSet()).fidelity == 1.0
    assert diff(FactSet(), FactSet(medications=[Medication("aspirin")])).fidelity == 0.0


def test_aggregate_pools_facts_rather_than_averaging_notes():
    big = rewrite(ORIGINAL.replace("500 mg", "250 mg"))
    small = diff(FactSet(medications=[Medication("aspirin", 81, unit="mg")]),
                 FactSet(medications=[Medication("aspirin", 81, unit="mg")]))
    summary = aggregate([big, small])
    assert summary["facts"] == big.original_count + 1
    assert summary["pooled_fidelity"] != summary["mean_note_fidelity"]


# --- the loop --------------------------------------------------------------

def test_conservative_rewrite_converges_in_one_round():
    outcome = run(ORIGINAL, GlossarySimplifier(), note_id="t")
    assert outcome.converged
    assert len(outcome.rounds) == 1


def test_loop_repairs_a_known_error():
    """The offline baseline loses nothing, so the repair round is exercised with
    a simplifier that breaks something on purpose and behaves once told."""
    outcome = run(ORIGINAL, LossySimplifier("halve_dose"), note_id="t")
    assert len(outcome.rounds) > 1
    assert outcome.rounds[0].result.corruptions
    assert outcome.converged, "the repair round should have fixed it"


def test_loop_feeds_the_diff_back_as_a_constraint():
    outcome = run(ORIGINAL, LossySimplifier("halve_dose"), note_id="t")
    assert outcome.rounds[1].constraints_given
    assert any("dose" in c for c in outcome.rounds[1].constraints_given)


def test_loop_returns_its_best_round_not_its_last():
    """Repair is not monotone: keeping the last attempt would throw away a
    better rewrite whenever round three is worse than round one."""
    outcome = run(ORIGINAL, LossySimplifier("halve_dose"), note_id="t")
    assert outcome.best.result.fidelity >= max(
        r.result.fidelity for r in outcome.rounds)


def test_loop_is_capped_and_surfaces_rather_than_ships():
    class Stubborn:
        name = "stubborn"

        def simplify(self, text, target_grade=6.0, constraints=None):
            return text.replace("500 mg", "250 mg")

    outcome = run(ORIGINAL, Stubborn(), note_id="t", max_rounds=3)
    assert outcome.needs_review
    assert len(outcome.rounds) <= 3


def test_loop_stops_when_repair_stops_moving():
    class Stubborn:
        name = "stubborn"

        def simplify(self, text, target_grade=6.0, constraints=None):
            return text.replace("500 mg", "250 mg")

    outcome = run(ORIGINAL, Stubborn(), note_id="t", max_rounds=3)
    assert len(outcome.rounds) == 2, "same complaint twice buys nothing"


def test_simplifier_never_sees_the_fact_set_on_the_first_draft():
    """A rewrite generated from a list of facts cannot lose a fact on that list,
    so the first draft is text-to-text or the measurement is circular."""
    seen = []

    class Spy:
        name = "spy"

        def simplify(self, text, target_grade=6.0, constraints=None):
            seen.append(constraints)
            return text

    run(ORIGINAL, Spy(), note_id="t")
    assert seen == [None]


# --- readability -----------------------------------------------------------

@pytest.mark.parametrize("word,count", [
    ("cat", 1), ("water", 2), ("medicine", 3), ("anticoagulant", 5),
    ("responsibility", 6), ("the", 1),
])
def test_syllables(word, count):
    assert syllables(word) == count


def test_grade_rises_with_harder_prose():
    easy = "Take one pill a day. Call us if you get a fever."
    hard = ("Administer one tablet per diem and contact the clinical service "
            "should pyrexia or any sign of systemic inflammatory response "
            "manifest itself.")
    assert flesch_kincaid_grade(hard) > flesch_kincaid_grade(easy)


def test_jargon_signal_catches_what_syllables_miss():
    """The point of the second metric: Flesch-Kincaid reads syllables, and
    'anticoagulant' has fewer of them than 'responsibility'."""
    assert syllables("anticoagulant") < syllables("responsibility")
    clinical = assess("You are taking an anticoagulant.")
    plain = assess("That is a big responsibility.")
    assert clinical.unexplained_count == 1
    assert plain.unexplained_count == 0


def test_a_defined_term_is_not_counted_as_unexplained():
    defined = assess("You are taking an anticoagulant (a blood thinner).")
    assert defined.unexplained_terms == []
    assert "anticoagulant" in defined.explained_terms


def test_drug_names_are_counted_separately():
    """'Warfarin' is a rare word and there is no safe alternative to it.
    Counting it as avoidable jargon would reward unsafe rewrites."""
    result = assess("Take warfarin 5 mg at bedtime.")
    assert "warfarin" in result.drug_terms
    assert "warfarin" not in result.unexplained_terms


def test_metadata_is_not_scored():
    """A note id like 'post-op-chole' is a rare word and not prose."""
    assert assess("# note: post-op-chole\nTake one pill a day.").unexplained_terms == []


def test_threshold_report_separates_the_two_vocabularies():
    report = threshold_report()
    assert "clinical terms below the threshold: 19/20" in report


# --- the model-backed path, without a key or a network --------------------

class _Block:
    def __init__(self, kind, text=""):
        self.type, self.text = kind, text


class _Response:
    def __init__(self, stop, *blocks):
        self.stop_reason, self.content = stop, list(blocks)


def test_rewrite_text_skips_thinking_blocks():
    from core.simplify import rewrite_text
    response = _Response("end_turn", _Block("thinking"), _Block("text", "Take one pill."))
    assert rewrite_text(response) == "Take one pill."


@pytest.mark.parametrize("response", [
    _Response("max_tokens", _Block("thinking")),
    _Response("max_tokens", _Block("text", "Take one pill a day. Call us if")),
    _Response("refusal"),
    _Response("end_turn", _Block("thinking")),
])
def test_incomplete_response_is_never_scored(response):
    """Thinking counts against max_tokens. A response that ran out of budget
    mid-thought has no text, and an empty rewrite would score as one that
    omitted every fact in the note: a number about the cap, not the model."""
    from core.simplify import IncompleteRewrite, rewrite_text
    with pytest.raises(IncompleteRewrite):
        rewrite_text(response)


def test_claude_simplifier_sends_no_sampling_parameters(monkeypatch):
    """API version 2026-10-21 rejects temperature, top_p and top_k."""
    from core import simplify
    sent = {}

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return _Response("end_turn", _Block("text", "ok"))

    instance = object.__new__(simplify.ClaudeSimplifier)
    instance._client = type("Client", (), {"messages": Messages()})()
    instance.model, instance.max_tokens, instance.effort = "m", 100, "medium"
    instance.simplify("Take one pill.")
    assert not {"temperature", "top_p", "top_k"} & set(sent)
    assert sent["output_config"] == {"effort": "medium"}
