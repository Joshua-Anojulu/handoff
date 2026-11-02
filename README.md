# Handoff

Simplify the discharge note without losing the dose.

Plenty of tools will rewrite discharge instructions at a lower reading level.
Almost none check whether the rewrite still says the same thing. A
simplification that turns *"take 5 mg twice daily for 10 days"* into *"take your
pill twice a day"* is more readable and more dangerous, and nobody measures it,
because measuring it is harder than doing it.

That gap is the project. Handoff extracts the clinical facts from a note,
rewrites the prose, re-extracts from the rewrite with the **same extractor**,
diffs the two fact sets, and classifies every discrepancy as an omission, a
corruption or a fabrication. If the diff is not empty it feeds the discrepancies
back as constraints and tries again, three rounds at most. A note it cannot
simplify without losing something is surfaced for human review rather than
shipped.

```
python eval.py            every measurement below, offline, no API key
open web/index.html       original, verifier, rewrite, side by side
```

---

## The design decision everything else rests on

**The extractor is deterministic.** Regex and a clinical vocabulary, not a
language model.

This is the whole reason the numbers mean anything. If the extractor were a
model, a model would be grading a model: the measurement would inherit the
grader's blind spots, two runs would disagree, and nobody could reproduce the
score. Regex is worse at language and far better at being an instrument. The
model's job here is the rewrite, which is the part that needs language.

The second half of that decision: **one code path**. There is no flag anywhere
in `core/extract.py` that says whether the text being read is an original or a
simplification. The moment such a flag exists the comparison stops meaning
anything.

The cost is that the extractor's own accuracy becomes a thing that has to be
measured rather than assumed. That is what the gold labels are for, and it is
the first number below.

---

## Results

### 1. Is the instrument any good?

The extractor against hand-labelled gold facts for all fifteen notes. The
labelling was done by a separate annotator working from the notes and a
published value space, explicitly barred from reading the extractor, so this is
agreement with an independent reading rather than with itself.

| | identity F1 | exact value agreement |
|---|---|---|
| **first measurement, before any fix** | **0.884** | **0.694** |
| current extractor vs round-1 labels | 0.879 | 0.780 |
| current extractor vs round-2 labels | 0.907 | 0.910 |

Only the first row is independent of what the annotator said. The other two
were measured after their findings changed the schema, so they are reported
underneath rather than in place of it. Per-family numbers, every disagreement,
and the fields that disagree most are printed by `python eval.py --study
extraction`.

Medications score highest (F1 0.992, the family where an error matters most).
Restrictions score lowest (F1 0.76 to 0.80), and the reason is in section 4.

**What the annotator found.** Three of their findings were not scoring
disagreements but design flaws, and all three are now fixed:

- **A phased regimen had nowhere to go.** *"Apixaban 10 mg twice daily for 7
  days, then 5 mg twice daily for at least 6 months"* produced a fact that said
  to **stop a blood thinner on day 7 after a pulmonary embolism**. Three notes
  prescribe in phases and in all three the dropped half was the plan. The
  schema now carries a second phase.
- **A threshold was a bare number, so its direction was lost.** *"Below 70
  mg/dL"* and *"above 300 mg/dL"* both stored as a number were indistinguishable,
  turning a hypoglycaemia rule into a hyperglycaemia one. Thresholds now carry
  direction, unit and, where there is one, the denominator: *"over 3 pounds per
  day"* rather than *"3"*.
- **A restriction's conditional clause was dropped.** *"Do not drive for 5 days,
  and not at all while taking oxycodone"* appears in four notes and every one
  produced a fact clearing the patient to drive on day 6 while still taking an
  opioid.

They also caught a regression I introduced while fixing their own complaint: I
replaced `one-sided weakness` with `weakness or numbness` when I should have
added it, and the warfarin note's *"weakness or numbness on one side"* lost the
laterality that makes it a stroke flag. Both values exist now.

### 2. Does the detector detect?

Measuring the verifier against a simplifier's output is circular, because nobody
knows the right answer for a given rewrite. Measuring it against a deliberate
edit is not, because the right answer is whatever was edited.

Twelve families of injected error, each one declaring in advance what it broke:

| mutation | cases | found | rate |
|---|---|---|---|
| drop a medication | 29 | 29 | 1.000 |
| halve a dose | 30 | 30 | 1.000 |
| drop "as needed" | 18 | 18 | 1.000 |
| drop a course length | 12 | 12 | 1.000 |
| drop a daily ceiling | 9 | 9 | 1.000 |
| downgrade an emergency instruction to a phone call | 8 | 8 | 1.000 |
| replace a threshold with a plain phrase | 11 | 11 | 1.000 |
| drop a follow-up | 28 | 28 | 1.000 |
| triple an appointment window | 28 | 28 | 1.000 |
| drop a restriction's duration | 13 | 13 | 1.000 |
| drop a weight limit | 3 | 3 | 1.000 |
| invent a medication | 15 | 15 | 1.000 |
| **all** | **204** | **204** | **1.000** |

**Zero false positives**: across 204 mutated notes the detector reported nothing
that the mutation had not declared. That column matters as much as the first
one, because a detector that flags everything scores 1.000 on detection alone.

**Read this with its limit in mind.** The detector and the harness were built
together, and four early false positives were resolved by fixing the extractor
and two by fixing the harness. So 1.000 means the detector handles these twelve
known error families, not that it would handle a thirteenth.

### 3. Does the loop do anything?

The headline experiment is one-shot against the same rewrite with the loop
around it. It is an honest comparison because the one-shot column is literally
round 1 of the same run: same simplifier, same note, no second sampling.

With a simplifier that loses facts (`--simplifier lossy`, a deliberate
error with known identity injected into each first draft):

| | one-shot | looped |
|---|---|---|
| pooled fidelity | 0.944 | **1.000** |
| weighted fidelity | 0.889 | **1.000** |
| notes with no errors | 0 / 15 | **15 / 15** |
| corruptions | 15 | **0** |

All fifteen repaired on the second round. The reading level pays for it in
places (the knee-replacement note goes from grade 7.3 to 8.4) because the
offline simplifier's repair strategy is to restate the original line verbatim.
That is a blunt fix and the table shows which way the trade went rather than
hiding it.

With the conservative offline simplifier there is **no difference**, because it
loses nothing on the first pass. That is a fact about the simplifier, not about
the loop, and the evaluation says so on screen rather than quietly reporting a
flattering number.

### 4. Reading level, measured two ways

Flesch-Kincaid counts syllables. By that measure "anticoagulant" (5 syllables)
is easier than "responsibility" (6). General-English frequency says otherwise.

```
clinical term       zipf   plain term          zipf
anticoagulant       2.42   responsibility      4.75
dyspnea             2.16   breathless          3.19
ambulate            1.40   walk                5.08
dehiscence          1.79   opening             4.94
prophylaxis         2.59   prevention          4.28
```

At a Zipf threshold of 3.0, 19 of 20 clinical terms fall below it and 17 of 20
plain terms sit at or above it. The three failures are real and worth stating:
"redness" (2.95) is flagged though it is plain, and "flare-up" written as one
word (1.28) is rarer than "exacerbation". `python eval.py --study threshold`
prints the table that justifies the number, so the threshold can be argued with
rather than taken on faith.

So Handoff reports **two** numbers, because a rewrite can move the first and not
the second: shorten the sentences, keep the jargon, and Flesch-Kincaid will
congratulate you.

Across the corpus, offline simplifier:

| | before | after |
|---|---|---|
| mean reading grade | 8.57 | 7.25 |
| unexplained rare terms | 183 | 88 |
| fidelity | | 1.000 |

Drug names are counted separately from avoidable jargon. "Warfarin" is a rare
word and the alternative to naming the drug is not naming the drug; counting it
as jargon would make the metric reward unsafe rewrites.

### 5. What the schema cannot hold

**Fifty of the instructions in these fifteen notes do not fit any of the four
families** (58 before the schema revision moved eight of them in). Weigh
yourself each morning. Use the incentive spirometer ten times an hour. Restrict
sodium to 2 grams daily. Elevate the leg above your heart. Check urine ketones
if glucose exceeds 250.

They are excluded from every fidelity score here, which means **a rewrite could
lose all fifty and still score 1.000.** That is the most important limitation of
the measurement and the reason this section exists rather than being left for a
reader to discover. Against 274 facts the schema does hold, that is about
fifteen per cent of the instructions in a note going unmeasured. `python
eval.py --study extraction` prints every one of them.

They are not a vocabulary gap that more enum values would close. They need a
fifth family, something like `self_care` with an action, a quantity and a
frequency. That is the next structural change.

Other known gaps, in the annotator's order of how much they matter:

- No action value for *"stop the drug"*, so *"report any tendon pain
  immediately and discontinue the drug"* is half-represented.
- Restrictions have no second phase, so concussion's *"no screens for 24 hours,
  then limit screen exposure"* keeps only the ban. Same failure as the apixaban
  one, one family over.
- The second-phase fields hardcode exactly two phases. Enough for these notes;
  an ordinary three-step steroid taper would break it.
- `condition` is free text with no canonicalisation rule, which is the largest
  remaining source of avoidable annotator disagreement.
- Home oxygen (*"2 litres per minute by nasal cannula"*) is unrepresentable:
  no flow-rate unit, no continuous frequency, no cannula route.
- `reaching or pushing` merges two distinct sternal precautions, and the cabg
  sentence *"no lifting, pushing, or pulling anything over 10 pounds"* straddles
  it and `lifting` with no rule for which wins.

### 6. The bug class that cost the most, and the check that now catches it

The subtlest failure in this project is not in the model and not in the diff.

If the simplifier rewrites a phrase the extractor recognises into a phrase it
does not, the fidelity score reports an information loss **that did not happen**.
Three real instances, each found by hand and each costing a debugging session:

| rewrite | what broke |
|---|---|
| "seek immediate medical attention" → "get medical help right away" | the plain form was in no action table, so seven warning signs vanished at once |
| "follow up with cardiology" → "see cardiology" | the cue required "see your", so an appointment vanished |
| "emesis" → "vomiting" | "emesis" was in no symptom table, so the rewrite looked like it had invented a warning sign |

`scripts/check_vocab.py` now asserts the invariant mechanically: for every
substitution the glossary can make, if the source phrase resolves to a canonical
value or trips one of the extractor's gating patterns, the replacement must
resolve to the same value and trip the same gate. On its first run it found six
more latent instances the fifteen notes happened not to exercise, and on the
run after the glossary was expanded it found six more. It is part of the test
suite.

---

## Running it

```bash
pip install -r requirements.txt

python eval.py                        all studies, offline, no key needed
python eval.py --study sensitivity    just the injected-error study
python eval.py --study headline --simplifier lossy    the loop repairing
python eval.py --json results.json    machine-readable

python -m pytest tests/ -q            117 tests
python scripts/check_vocab.py         the vocabulary invariant
python scripts/build_web_data.py      regenerate web/data.json
```

**The model-backed path.** Everything above runs offline with no credentials.
The offline simplifier is a substitution table: it cannot restructure a
sentence, so it loses almost nothing and reaches only about grade 7.25. **Its
high fidelity is the floor, not a result.** It is the control.

For the number that matters:

```bash
export ANTHROPIC_API_KEY=...
python eval.py --study headline --simplifier claude --json claude-results.json
```

That runs Claude Sonnet 5.5 (`claude-sonnet-5-5`) at its default effort.
`--model` and `--effort` change either; the exact model is printed in the
study header and recorded in the JSON, so the number in this README can say
which model produced it. Fifteen notes at up to three rounds each is at most
45 requests.

Two safeguards matter more than they look. Current models think before they
answer and the thinking counts against `max_tokens`, so a tight cap can spend
the whole budget and return no text. An empty or truncated response is raised
as an error rather than scored, because the loop would otherwise grade a half
rewrite as one that dropped the second half of the note, and report that as
the model's fidelity. And no sampling parameters are sent: the current API
version rejects `temperature`, `top_p` and `top_k`.

The key is read from the environment, never written to disk, never logged and
never included in any output file. `pytest` needs neither key nor network.

---

## Layout

```
core/
  schema.py        four fact families; identity separate from value
  vocab.py         clinical vocabulary, alias-matching rules
  extract.py       note to facts. deterministic. one code path
  glossary.py      plain-language substitutions
  simplify.py      offline baseline, model-backed, and the test doubles
  diff.py          omission, corruption, fabrication, and the fidelity score
  loop.py          extract, simplify, re-extract, repair, cap at 3
  readability.py   Flesch-Kincaid plus the rare-term signal
  mutate.py        twelve families of injected error with known ground truth
  text.py          shared segmentation helpers
data/
  notes/           15 synthetic discharge notes
  gold/            hand-labelled facts, both rounds, and the labelling guide
  drug_names.txt   205 drug names from the FDA interaction tables
web/index.html     original left, verifier middle, rewrite right
eval.py            every number in this README
scripts/           the vocabulary invariant, and the web data builder
tests/             117 tests
```

### Why identity is separate from value

Every fact has an identity (the drug, the symptom, the activity) and a value
(the dose, the action, the duration). Two facts are the same fact when their
identities match, whatever their values say.

Without that split, a changed dose looks like one omission plus one fabrication,
and the most dangerous failure mode disappears into the least dangerous counting.
With it, `halve_dose` reports exactly one corruption on the `dose` field, which
is what the test asserts.

### Deliberate strictness

`"every 6 hours"` and `"four times daily"` are kept as different values. They are
not the same instruction: one means setting an alarm for 2am and the other means
meals and bedtime. A rewrite that swaps them is flagged, and that is intended
rather than a limitation.

Dropping `"as needed"` is a corruption of the frequency, not a wording change: it
turns a PRN opioid into a scheduled one.

Route is the one medication field treated as wording. "By mouth" dropped from a
tablet instruction is a style loss; every other field is not.

---

## Data

Fifteen synthetic discharge notes, written for this project in clinical
register: cholecystectomy, heart failure, pneumonia, new atrial fibrillation on
warfarin, appendectomy, DKA, knee replacement, pyelonephritis, COPD
exacerbation, caesarean delivery, cellulitis, upper GI bleed, pulmonary
embolism, CABG, concussion.

**No real patient data, at any point.** Synthetic removes every privacy question
and costs nothing here, since what is being measured is whether a rewrite
preserves facts, not whether the facts are anyone's.

The drug vocabulary is 205 names extracted from the FDA's drug interaction
tables, shipped as `data/drug_names.txt` so the list is inspectable rather than
buried in code, plus a curated list of drugs that appear on discharge paperwork
and are not CYP substrates.

---

## AI disclosure

This project was built in collaboration with Claude (Anthropic), which wrote
the code, the synthetic notes and this README. The independent gold labelling
was done by a separate Claude instance given the notes and the labelling guide
and barred from reading the extractor; its findings changed the schema three
times, and section 1 reports which numbers were measured before that feedback
and which after.

The fidelity scores, sensitivity rates and readability numbers in this README
are all produced by `python eval.py` from the code in this repository. Nothing
here is estimated or reported from memory. Where a number is weaker than it
looks (the offline simplifier's 1.000 fidelity, the detector's 1.000 on
mutations it was debugged against, the fifty instructions outside the schema),
the section that reports it says so.
