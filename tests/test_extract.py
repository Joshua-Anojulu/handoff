"""Extractor tests.

The ones that matter most are the register-invariance tests: the same
instruction written in clinical register and in plain English must extract to
the same fact. If that fails, every fidelity number this project reports is
measuring the extractor's vocabulary rather than the rewrite's faithfulness.
"""
from __future__ import annotations

import pytest

from core.extract import (canonical_drug, extract, extract_duration,
                          extract_second_phase, extract_window, is_negated,
                          normalise_frequency, segments)
from core.vocab import ACTIONS, ACTIVITIES, FOLLOWUP_ROLES, SYMPTOMS, alias_pattern


# --- frequency -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("take one PO BID", "twice daily"),
    ("take one twice a day", "twice daily"),
    ("take one two times a day", "twice daily"),
    ("take one morning and night", "twice daily"),
    ("take one TID", "three times daily"),
    ("take one QID", "four times daily"),
    ("take one 4 times a day", "four times daily"),
    ("take one QD", "once daily"),
    ("take one every day", "once daily"),
    ("take one QHS", "at bedtime"),
    ("take one at bedtime", "at bedtime"),
    ("take one q6h", "every 6 hours"),
    ("take one every 6 hours", "every 6 hours"),
    ("take one q4-6h", "every 4 to 6 hours"),
    ("take one every 4 to 6 hours", "every 4 to 6 hours"),
    ("take one q6h PRN", "every 6 hours as needed"),
    ("take one every 6 hours if you need it", "every 6 hours as needed"),
    ("take one weekly", "weekly"),
    ("take one every other day", "every other day"),
])
def test_frequency_normalises_across_registers(text, expected):
    assert normalise_frequency(text) == expected


def test_every_six_hours_is_not_four_times_daily():
    """Deliberate strictness. 'Every 6 hours' means an alarm at 2am and 'four
    times a day' means meals and bedtime. A rewrite that swaps them has changed
    the instruction, and the extractor must let the diff say so."""
    assert normalise_frequency("q6h") != normalise_frequency("QID")


def test_prn_is_part_of_the_frequency():
    """Dropping 'as needed' turns a PRN opioid into a scheduled one."""
    assert normalise_frequency("every 6 hours") != normalise_frequency("every 6 hours PRN")


# --- doses, durations, windows ---------------------------------------------

@pytest.mark.parametrize("text,days,note", [
    ("for 7 days", 7, ""),
    ("x 7 days", 7, ""),
    ("for ten days", 10, ""),
    ("for 2 weeks", 14, ""),
    ("for 6 weeks", 42, ""),
    ("for 3 months", 90, ""),
    ("for 72 hours", 3, ""),
    ("complete the full course", None, "until finished"),
    ("take all of it", None, "until finished"),
    ("until the staples are out", None, "until staples out"),
    ("until your staples come out", None, "until staples out"),
    ("until cleared by your surgeon", None, "until cleared"),
])
def test_duration(text, days, note):
    assert extract_duration(text) == (days, note)


@pytest.mark.parametrize("text,days", [
    ("in 14 days", 14), ("in 2 weeks", 14), ("within 1 week", 7),
    ("in 3 months", 90), ("in 7 to 10 days", 7), ("tomorrow", 1),
])
def test_window(text, days):
    assert extract_window(text) == days


def test_window_range_takes_the_lower_bound():
    """The earlier date is the one that constrains the person."""
    assert extract_window("in 7 to 10 days") == 7


def test_second_phase():
    dose, freq, days, note = extract_second_phase(
        "Apixaban 10 mg PO BID for 7 days, then 5 mg PO BID for at least 6 months.")
    assert (dose, freq, days) == (5.0, "twice daily", 180)
    assert note == ""


def test_second_phase_stop():
    assert extract_second_phase("Prednisone 40 mg PO QD for 5 days, then stop.") == (
        None, "", None, "stop")


def test_first_phase_duration_excludes_the_second():
    """'For 7 days, then 5 mg for 6 months' is a 7-day first phase, not 180."""
    med = next(m for m in extract(
        "- Apixaban 10 mg PO BID for 7 days, then 5 mg PO BID for at least 6 months."
    ).medications if m.name == "apixaban")
    assert med.duration_days == 7
    assert med.then_duration_days == 180


def test_phased_regimen_survives_a_plain_rewrite():
    clinical = "- Apixaban 10 mg PO BID for 7 days, then 5 mg PO BID for 6 months."
    plain = ("- Apixaban 10 mg by mouth twice a day for 7 days, "
             "then 5 mg by mouth twice a day for 6 months.")
    assert extract(clinical).by_key() .keys() == extract(plain).by_key().keys()
    assert (extract(clinical).medications[0].to_fact().value
            == extract(plain).medications[0].to_fact().value)


# --- drug identity ---------------------------------------------------------

@pytest.mark.parametrize("brand,generic", [
    ("Keflex", "cephalexin"), ("Tylenol", "acetaminophen"),
    ("Advil", "ibuprofen"), ("Coumadin", "warfarin"), ("Colace", "docusate"),
])
def test_brand_maps_to_generic(brand, generic):
    """Swapping a brand name for its generic is a readability improvement and
    no information loss. Without the mapping it scored as one omission plus one
    fabrication."""
    assert canonical_drug(brand) == generic


def test_two_insulins_are_two_drugs():
    """Basal and bolus insulin are not interchangeable. Collapsing both to
    'insulin' would let a rewrite delete one for free."""
    facts = extract("- Insulin glargine 18 units subcutaneous QHS.\n"
                    "- Insulin lispro 6 units subcutaneous TID with meals.")
    names = {m.name for m in facts.medications}
    assert names == {"insulin glargine", "insulin lispro"}


def test_dose_ceiling_attaches_to_the_drug_on_its_own_line():
    facts = extract("- Oxycodone 5 mg PO q4h PRN. Do not exceed 6 tablets in 24 hours.\n"
                    "- Docusate 100 mg PO BID.")
    by_name = {m.name: m for m in facts.medications}
    assert by_name["oxycodone"].max_per_day == "6 tablet"
    assert by_name["docusate"].max_per_day == ""


# --- negation --------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Do not drive", "Don't drive", "Avoid alcohol", "Abstain from alcohol",
    "Stay away from alcohol", "Refrain from lifting", "No lifting over 10 pounds",
    "Stop taking warfarin", "Wait 2 weeks before driving",
])
def test_negation(text):
    assert is_negated(text)


def test_a_negated_drug_is_a_restriction_not_a_prescription():
    facts = extract("Do not take ibuprofen, naproxen, or aspirin for 2 weeks.")
    assert facts.medications == []
    assert [r.activity for r in facts.restrictions] == ["nsaids"]


def test_restriction_condition_outlives_the_number():
    """'Do not drive for 5 days, and not at all while taking oxycodone' clears
    the patient on day 6 unless the second clause is kept."""
    restriction = next(r for r in extract(
        "- Do not drive for 5 days, and not at all while taking oxycodone."
    ).restrictions if r.activity == "driving")
    assert restriction.duration_days == 5
    assert restriction.condition == "while taking oxycodone"


# --- red flags -------------------------------------------------------------

def test_threshold_keeps_its_direction():
    """A floor and a ceiling stored as bare numbers are indistinguishable, and
    a hypoglycaemia rule then reads as a hyperglycaemia rule."""
    facts = extract("Call us for blood glucose above 300 mg/dL or "
                    "blood glucose below 70 mg/dL.")
    thresholds = {f.symptom: f.threshold for f in facts.red_flags}
    assert thresholds["high blood sugar"] == "over 300 mg/dL"
    assert thresholds["low blood sugar"] == "below 70 mg/dL"


def test_threshold_attaches_to_its_own_symptom():
    """One sentence, one threshold, several symptoms. The threshold belongs to
    the symptom it follows, not to all of them."""
    facts = extract("Return to the ED for temperature above 101 F, "
                    "intractable vomiting, or worsening flank pain.")
    thresholds = {f.symptom: f.threshold for f in facts.red_flags}
    assert thresholds["fever"] == "over 101 F"
    assert thresholds.get("vomiting", "") == ""


def test_action_severity_is_recorded():
    facts = extract("Call 911 for chest pain.\nCall the office for a rash.")
    actions = {f.symptom: f.action for f in facts.red_flags}
    assert actions["chest pain"] == "emergency"
    assert actions["rash"] == "call"


def test_seek_immediate_attention_is_an_emergency_not_a_phone_call():
    facts = extract("Seek immediate medical attention for chest pain.")
    assert facts.red_flags[0].action == "emergency"


def test_call_the_office_for_x_is_a_red_flag():
    """The commonest construction in a real note, and it contains no
    conditional word at all."""
    facts = extract("Call the surgical office right away for fever over 100.4 F.")
    assert [f.symptom for f in facts.red_flags] == ["fever"]


def test_bullets_inherit_the_action_from_their_heading():
    facts = extract("Call the clinic immediately for any of the following:\n"
                    "- Fever over 101 F\n- Purulent drainage\n- Chest pain")
    assert {f.symptom for f in facts.red_flags} == {"fever", "wound drainage", "chest pain"}
    assert all(f.action for f in facts.red_flags)


def test_specific_symptom_subsumes_the_general_one():
    """'Vomiting blood' is not merely vomiting."""
    symptoms = {f.symptom for f in extract("Call 911 for haematemesis.").red_flags}
    assert symptoms == {"vomiting blood"}


# --- follow-ups ------------------------------------------------------------

def test_section_heading_is_not_an_appointment():
    for heading in ("FOLLOW-UP", "Follow-up:", "Follow up"):
        assert extract(heading).follow_ups == []


def test_resume_the_normal_schedule_is_not_an_appointment():
    assert extract("If you miss one, take it as soon as you remember, "
                   "then resume the normal schedule.").follow_ups == []


def test_emergency_instruction_is_not_an_appointment():
    assert extract("Return to the emergency department immediately if the "
                   "erythema advances.").follow_ups == []


def test_two_appointments_with_one_specialty_stay_distinct():
    """A two-week incision check and a six-week postpartum visit are both
    obstetrics. Keyed on the role alone one of them disappeared."""
    facts = extract("- Follow up with obstetrics in 2 weeks.\n"
                    "- Postpartum visit with obstetrics in 6 weeks.")
    assert [(f.who, f.ordinal, f.within_days) for f in facts.follow_ups] == [
        ("obstetrics", 1, 14), ("obstetrics", 2, 42)]


def test_contact_plus_role_plus_window_is_an_appointment():
    facts = extract("Contact your primary care physician within 1 week "
                    "for medication review.")
    assert [(f.who, f.within_days) for f in facts.follow_ups] == [("primary care", 7)]


# --- vocabulary hygiene ----------------------------------------------------

@pytest.mark.parametrize("table", [SYMPTOMS, ACTIVITIES, ACTIONS, FOLLOWUP_ROLES])
def test_vocab_aliases_do_not_match_inside_words(table):
    """'er' must not match 'fever' and 'ed' must not match 'exceed'.

    The first version of the vocabulary matched substrings, and the extractor
    invented an emergency-department appointment out of 'do not exceed 6
    tablets'.
    """
    decoys = ("fever exceeded the expected number, considered and delivered, "
              "prescribed, interpreted, lifted, covered, ordered")
    for key, aliases in table.items():
        for alias in aliases:
            bare = alias.rstrip("*")
            if len(bare) > 3 or " " in bare:
                continue
            assert not alias_pattern((alias,)).search(decoys), (
                f"{key!r} alias {alias!r} matches inside an unrelated word")


# --- segmentation ----------------------------------------------------------

def test_segments_split_bullets_and_sentences():
    found = segments("- Take A. Take B.\n- Take C.\nHeading:")
    assert found == ["Take A.", "Take B.", "Take C.", "Heading:"]
