"""Clinical vocabulary for the extractor.

Two sources, both auditable:

1. `FDA_DRUGS` is lifted from the FDA's drug interaction tables (the same
   source the Cascade project uses) and shipped as data/drug_names.txt so the
   list is inspectable rather than buried in code.
2. `DISCHARGE_DRUGS` is a hand-curated list of drugs that actually appear on
   discharge paperwork and are not CYP substrates, so are absent from (1).

`BRAND_TO_GENERIC` is the part that keeps the fidelity numbers honest. Swapping
"acetaminophen" for "Tylenol" makes a note easier to read and changes nothing
about what the person should swallow. An extractor that did not know they are
the same drug would score that rewrite as one omission plus one fabrication,
and the project's headline number would look better than the truth. Every
entry here makes the measured error rate go down, which is the direction that
requires justification, so each is a generic/brand pair, nothing looser.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=None)
def alias_pattern(aliases: tuple[str, ...]) -> re.Pattern:
    """Compile a table's aliases under the convention documented below.

    `\\b` at the start of every alias. A trailing `*` becomes "any word
    continuation"; without it the alias must end on a word boundary too.
    """
    parts = []
    for alias in aliases:
        if alias.endswith("*"):
            parts.append(r"\b" + re.escape(alias[:-1]) + r"\w*")
        else:
            parts.append(r"\b" + re.escape(alias) + r"\b")
    return re.compile("|".join(parts), re.I)


def matching_keys(table: dict[str, tuple[str, ...]], text: str) -> list[str]:
    """Every key in `table` whose aliases appear in `text`, table order kept."""
    return [key for key, aliases in table.items()
            if alias_pattern(tuple(aliases)).search(text)]


def _load_fda_drugs() -> frozenset[str]:
    path = _DATA / "drug_names.txt"
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


FDA_DRUGS = _load_fda_drugs()

# Drugs common on discharge paperwork. Not a formulary, just coverage for the
# synthetic notes plus the obvious over-the-counter and post-op items.
DISCHARGE_DRUGS = frozenset({
    "acetaminophen", "ibuprofen", "naproxen", "aspirin", "oxycodone",
    "hydrocodone", "tramadol", "morphine", "gabapentin", "amoxicillin",
    "amoxicillin-clavulanate", "cephalexin", "cefdinir", "doxycycline",
    "azithromycin", "nitrofurantoin", "trimethoprim-sulfamethoxazole",
    "metronidazole", "penicillin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "carvedilol", "atenolol", "hydrochlorothiazide",
    "furosemide", "spironolactone", "metformin", "insulin", "glipizide",
    # Multi-word names, which matter more than they look. "Insulin glargine"
    # and "insulin lispro" are a basal and a bolus insulin; collapsing both to
    # "insulin" would let a rewrite delete one of them and score as a dose
    # change rather than a missing drug. The alternation is sorted longest-first
    # at use, so the specific name always wins over the bare one.
    "insulin glargine", "insulin lispro", "insulin aspart", "insulin detemir",
    "insulin degludec", "insulin glulisine", "nph insulin", "regular insulin",
    "metoprolol succinate", "metoprolol tartrate",
    "fluticasone-salmeterol", "fluticasone salmeterol",
    "amoxicillin-clavulanate", "amoxicillin clavulanate",
    "trimethoprim-sulfamethoxazole", "polyethylene glycol", "ferrous sulfate",
    "potassium chloride", "calcium carbonate", "folic acid", "vitamin d",
    "prenatal vitamin", "magnesium oxide",
    "levothyroxine", "prednisone", "methylprednisolone", "albuterol",
    "fluticasone", "montelukast", "pantoprazole", "omeprazole",
    "famotidine", "ondansetron", "promethazine", "docusate", "senna",
    "polyethylene glycol", "bisacodyl", "enoxaparin", "apixaban",
    "rivaroxaban", "warfarin", "clopidogrel", "atorvastatin", "simvastatin",
    "sertraline", "escitalopram", "trazodone", "melatonin", "ferrous sulfate",
    "potassium chloride", "vitamin d", "calcium carbonate", "folic acid",
})

DRUGS = FDA_DRUGS | DISCHARGE_DRUGS

BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "percocet": "oxycodone",
    "norco": "hydrocodone",
    "vicodin": "hydrocodone",
    "ultram": "tramadol",
    "neurontin": "gabapentin",
    "augmentin": "amoxicillin-clavulanate",
    "keflex": "cephalexin",
    "zithromax": "azithromycin",
    "bactrim": "trimethoprim-sulfamethoxazole",
    "flagyl": "metronidazole",
    "macrobid": "nitrofurantoin",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "cozaar": "losartan",
    "norvasc": "amlodipine",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "coreg": "carvedilol",
    "lasix": "furosemide",
    "aldactone": "spironolactone",
    "glucophage": "metformin",
    "synthroid": "levothyroxine",
    "deltasone": "prednisone",
    "ventolin": "albuterol",
    "proair": "albuterol",
    "flovent": "fluticasone",
    "singulair": "montelukast",
    "protonix": "pantoprazole",
    "prilosec": "omeprazole",
    "pepcid": "famotidine",
    "zofran": "ondansetron",
    "phenergan": "promethazine",
    "colace": "docusate",
    "senokot": "senna",
    "miralax": "polyethylene glycol",
    "dulcolax": "bisacodyl",
    "lovenox": "enoxaparin",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "coumadin": "warfarin",
    "plavix": "clopidogrel",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "zoloft": "sertraline",
    "lexapro": "escitalopram",
    "desyrel": "trazodone",
    "motrin ib": "ibuprofen",
}

# Alias convention, used by every table below.
#
# An alias is matched on a word boundary at its start, never as a bare
# substring. A trailing "*" means "and anything that continues the word", so
# "cardiolog*" matches cardiology and cardiologist while "er" matches only the
# standalone word. The first version of this file matched substrings; "er" then
# matched "fever" and "ed" matched "exceed", and the extractor quietly invented
# an emergency-department appointment out of "do not exceed 6 tablets". Hence
# the convention, and hence test_vocab_aliases_do_not_match_inside_words.

# Who a follow-up is with. Identity is the role, not the person's name, because
# "Dr. Alvarez" in the original and "your surgeon" in the rewrite are the same
# appointment.
FOLLOWUP_ROLES = {
    # Not a bare "surgical": that matched "surgical site", so every mention of
    # the wound read as an appointment with the surgeon.
    "surgeon": ("surgeon", "surgical team", "surgical office", "surgical clinic",
                "surgical service", "operating surgeon", "surgery office",
                "surgery clinic"),
    "primary care": ("primary care", "pcp", "family doctor", "family physician",
                     "regular doctor", "gp", "general practitioner",
                     "own doctor", "usual doctor"),
    "cardiology": ("cardiolog*", "heart doctor", "heart specialist", "heart clinic"),
    "orthopedics": ("orthoped*", "bone doctor", "bone specialist"),
    "wound care": ("wound care", "wound clinic", "wound nurse", "dressing clinic"),
    "physical therapy": ("physical therap*", "pt", "physiotherap*", "rehab*"),
    "obstetrics": ("obstetric*", "ob/gyn", "obgyn", "midwife", "prenatal"),
    "oncology": ("oncolog*", "cancer doctor", "cancer specialist", "infusion"),
    "nephrology": ("nephrolog*", "kidney doctor", "kidney specialist", "dialysis"),
    "pulmonology": ("pulmonolog*", "lung doctor", "lung specialist"),
    "neurology": ("neurolog*", "nerve doctor", "brain doctor"),
    "gastroenterology": ("gastroenterolog*", "gi doctor", "stomach doctor"),
    "anticoagulation clinic": ("anticoagulation", "coumadin clinic",
                               "blood thinner clinic", "inr clinic", "inr check"),
    "lab": ("lab", "labs", "laboratory", "blood draw", "blood test", "blood work",
            "complete blood count", "basic metabolic panel", "hba1c",
            "urine culture"),
    "cardiac surgery": ("cardiac surgery", "cardiothoracic", "heart surgeon"),
    "cardiac rehabilitation": ("cardiac rehabilitation", "cardiac rehab"),
    "pulmonary rehabilitation": ("pulmonary rehabilitation", "pulmonary rehab"),
    "endocrinology": ("endocrinolog*", "diabetes service", "diabetes clinic",
                      "diabetes doctor"),
    "diabetes education": ("diabetes education", "diabetes educator",
                           "diabetes nurse"),
    "lactation": ("lactation", "breastfeeding support", "feeding support"),
    "hematology": ("haematolog*", "hematolog*", "blood doctor"),
    "imaging": ("imaging", "radiolog*", "ultrasound", "chest radiograph",
                "x-ray", "ct scan", "mri", "endoscopy"),
    "dental": ("dentist", "dental"),
}
# The "clinic" value is a deliberate catch-all for an appointment whose
# specialty the note never names. It is the one role identity that can hold
# several distinct appointments, so a loss inside it is reported as a changed
# window rather than a missing appointment. Narrowing it is a matter of adding
# roles above; every role added moves appointments out of the bucket.

# Red-flag symptoms. Clinical term first, then the plain-language forms a
# simplifier is likely to reach for. Same reasoning as the brand map: without
# these, correct simplification scores as corruption.
SYMPTOMS = {
    "fever": ("fever", "fevers", "feverish", "temperature over", "temperature above",
              "temperature higher", "temp over", "temp above", "febrile",
              "pyrexia", "running a temperature"),
    "chest pain": ("chest pain", "pain in your chest", "chest pressure",
                   "chest tightness", "tightness in your chest",
                   "squeezing in your chest"),
    "dyspnea": ("dyspnea", "shortness of breath", "short of breath",
                "trouble breathing", "difficulty breathing", "breathless*",
                "hard to breathe", "can't breathe", "cannot breathe",
                "out of breath"),
    "bleeding": ("bleeding", "blood soaking", "soaks through", "soaking through",
                 "hemorrhage", "blood that will not stop", "blood that won't stop",
                 "will not stop bleeding", "won't stop bleeding"),
    "wound drainage": ("drainage", "draining", "pus", "oozing", "fluid coming from",
                       "discharge from the", "bad smell", "foul smell", "smells bad"),
    "wound redness": ("redness", "red streaks", "erythema", "spreading red",
                      "getting redder", "red and warm"),
    "swelling": ("swelling", "edema", "swollen", "puffiness"),
    "calf pain": ("calf pain", "pain in your calf", "pain in one leg",
                  "leg pain and swelling", "one leg is swollen"),
    "vomiting": ("vomit*", "emesis", "retching", "throwing up", "cannot keep",
                 "can't keep", "unable to keep", "keep anything down",
                 "being sick"),
    "confusion": ("confusion", "confused", "altered mental", "disoriented",
                  "not acting like", "hard to wake"),
    "severe pain": ("severe pain", "pain that is not controlled",
                    "uncontrolled pain", "pain the medicine does not",
                    "pain the medicine doesn't", "worsening pain",
                    "pain gets worse", "pain is getting worse"),
    "no urine": ("not urinating", "unable to urinate", "cannot urinate",
                 "can't pee", "cannot pee", "no urine", "not passing urine",
                 "have not urinated", "haven't urinated"),
    "rash": ("rash", "hives", "itchy bumps"),
    "fainting": ("fainting", "passed out", "pass out", "syncope", "blacked out",
                 "feel like you will faint"),
    "palpitations": ("palpitations", "heart racing", "racing heartbeat",
                     "racing or pounding", "pounding heartbeat",
                     "fluttering in your chest", "irregular heartbeat",
                     "heart is pounding", "skipping beats"),
    "weight gain": ("weight gain", "gained weight", "gaining weight",
                    "pounds in a day", "pounds overnight", "pounds in one day"),
    "incision opening": ("incision opens", "wound opens", "stitches come apart",
                         "dehiscence", "edges separate", "incision separates",
                         "cut opens", "opens up", "cut opening up",
                         "wound opening up"),
    "high blood sugar": ("hyperglycemia", "hyperglycaemia", "high blood sugar",
                         "blood glucose above", "blood glucose exceeds",
                         "blood sugar above", "blood sugar over",
                         "glucose above", "sugar is high"),
    "low blood sugar": ("hypoglycemia", "hypoglycaemia", "low blood sugar",
                        "blood glucose below", "blood sugar below",
                        "glucose below", "sugar is low"),
    "ketones": ("ketone*", "ketoacidosis"),
    "black stool": ("melaena", "melena", "black tarry", "black stool",
                    "tarry stool"),
    "vomiting blood": ("haematemesis", "hematemesis", "vomiting blood",
                       "coffee grounds", "coffee-ground", "blood in your vomit"),
    "blood in urine": ("haematuria", "hematuria", "blood in the urine",
                       "blood in your urine"),
    "rectal bleeding": ("blood per rectum", "rectal bleeding", "blood in the stool",
                        "blood in your stool", "blood when you poop"),
    "coughing blood": ("haemoptysis", "hemoptysis", "coughing up blood",
                       "blood in your spit", "blood when you cough"),
    "headache": ("headache*", "head ache"),
    "head injury": ("head injury", "hit your head", "bang your head",
                    "blow to the head"),
    "seizure": ("seizure*", "convulsion*", "fit*"),
    "slurred speech": ("slurred speech", "trouble speaking", "garbled speech",
                       "dysarthria"),
    "vision change": ("diplopia", "double vision", "visual change",
                      "blurred vision", "vision change*", "unequal pupil*",
                      "see spots"),
    # Two values, not one. The concussion note says "new weakness or numbness"
    # with no laterality, so a canonical value that adds it asserts something
    # the note does not say. The warfarin note says "on one side", and in an
    # anticoagulated patient that unilaterality is exactly what makes it a
    # stroke flag rather than benign paraesthesia. An earlier revision replaced
    # the specific value with the general one and traded a false assertion in
    # one note for a lost distinction in the other; both are kept instead, with
    # the specific one subsuming the general.
    "one-sided weakness": ("weakness on one side", "numbness on one side",
                           "one side of your body", "face droop*",
                           "hemiparesis", "unilateral weakness"),
    "weakness or numbness": ("new weakness", "new numbness",
                             "weakness or numbness"),
    "induration": ("induration", "hardness", "a hard area", "firm area",
                   "fluctuant", "a lump", "a soft swollen area"),
    "hypotension": ("hypotension", "low blood pressure"),
    "sputum change": ("sputum", "phlegm", "mucus", "change in colour of what you cough",
                      "colour of your phlegm", "color of your phlegm"),
    "low urine output": ("decreased urine output", "urinating less",
                         "passing less urine", "urine output"),
    "sternal instability": ("sternal instability", "clicking", "shifting sensation",
                            "breastbone mov*", "breastbone click*"),
    "fluid from ears or nose": ("fluid from the nose", "fluid from your nose",
                                "fluid from the ears", "fluid from your ears",
                                "blood from the nose or ears", "otorrhoea", "rhinorrhoea"),
    "hopelessness": ("hopeless*", "feeling very low", "despair*"),
    # Its own value, not folded into hopelessness. On a postpartum discharge
    # note this is the most urgent line on the page, and compressing it into a
    # mood symptom was the single worst loss the schema produced.
    "thoughts of harm": ("thoughts of harming", "harming yourself",
                         "thoughts of hurting", "hurting yourself",
                         "harm yourself", "harming the baby",
                         "hurting the baby", "thoughts of suicide"),
    "flank pain": ("flank pain", "pain in your side", "pain in your back and side",
                   "loin pain"),
    "epigastric pain": ("epigastric", "upper belly", "pain in your upper belly",
                        "pain below your ribs", "upper abdominal pain"),
    "tendon pain": ("tendon pain", "tendon", "achilles"),
    "crepitus": ("crepitus", "crackling under the skin"),
    "skin necrosis": ("necrosis", "skin turning black", "dying tissue",
                      "dusky skin"),
    "abdominal rigidity": ("rigidity", "rigid abdomen", "hard belly",
                           "board-like"),
    "advancing redness": ("beyond the marked border", "past the mark",
                          "advances beyond", "outside the line"),
    "worsening abdominal pain": ("abdominal pain that is worsening",
                                 "belly pain that is getting worse",
                                 "worsening abdominal pain",
                                 "abdominal pain is worsening"),
    "orthopnea": ("orthopnea", "orthopnoea", "breathless when lying",
                  "breathlessness when lying", "need more pillows",
                  "cannot lie flat"),
    "bruising": ("unexplained bruising", "bruising", "bruise easily"),
    "bleeding gums": ("bleeding gums", "gums bleed"),
    "somnolence": ("somnolence", "sleepiness", "very sleepy", "hard to rouse",
                   "difficulty rousing", "drowsiness"),
}

# When a specific symptom fires, the general one it contains does not.
#
# "Vomiting blood" is not merely vomiting, "orthopnea" is not merely
# breathlessness, and induration is not merely swelling. Without this the
# specific phrase matched both keys, so rewriting "haematemesis" as "vomiting
# blood" added a `vomiting` fact that the diff reported as a fabrication the
# rewrite had not committed.
SYMPTOM_SUBSUMES = {
    "one-sided weakness": ("weakness or numbness",),
    "thoughts of harm": ("hopelessness",),
    "epigastric pain": ("worsening abdominal pain",),
    "vomiting blood": ("vomiting",),
    "orthopnea": ("dyspnea",),
    "induration": ("swelling",),
    "advancing redness": ("wound redness",),
    "coughing blood": ("sputum change",),
    "black stool": ("rectal bleeding",),
}


# What a red flag tells you to do. Severity order matters: these are compared
# in the diff, and downgrading "go to the emergency room" to "call the office"
# is a corruption even though both are a contact instruction.
ACTIONS = {
    "emergency": ("911", "emergency room", "emergency department", "nearest er",
                  "go to the er", "go to the ed", "call an ambulance",
                  "return to the er", "return to the ed",
                  "back to the emergency", "nearest emergency",
                  "ambulance", "immediately go", "go straight to the hospital",
                  # The plain-language forms matter as much as the clinical
                  # ones. A rewrite that turns "seek immediate medical
                  # attention" into "get medical help right away" has changed
                  # nothing, and when these were missing the extractor read
                  # that rewrite as having deleted seven warning signs.
                  "get medical help right away", "get help right away",
                  "get emergency help", "get medical care right away",
                  "go to the hospital",
                  # "Seek immediate medical attention" is not a phone call. It
                  # was classified as one here, so rewriting it in plain words
                  # read as an escalation in severity on every warning sign in
                  # the sentence.
                  "seek immediate", "seek medical attention", "seek care now",
                  "seek urgent"),
    "call urgently": ("call right away", "call immediately", "call us at once",
                      "call straight away", "call without delay",
                      "call us right away", "call the office right away",
                      "call your doctor right away", "tell your doctor right away",
                      "contact your doctor right away"),
    "call": ("call*", "contact*", "telephone", "reach out", "let us know",
             "notify", "tell your", "get in touch", "ring"),
}
ACTION_SEVERITY = {"emergency": 3, "call urgently": 2, "call": 1, "": 0}

# Restricted activities.
ACTIVITIES = {
    "driving": ("driv*", "behind the wheel", "operate a car", "operating a car",
                "get behind the wheel"),
    "lifting": ("lift*", "carry anything heavy", "heavy object*",
                "pick up anything heavy", "carrying anything heavy"),
    "bathing": ("bath*", "soak*", "tub", "tubs", "swim*", "hot tub", "submerge*",
                "pool"),
    "alcohol": ("alcohol*", "beer", "wine", "liquor", "alcoholic drink*"),
    "smoking": ("smok*", "cigarette*", "nicotine", "vap*", "tobacco"),
    "exercise": ("exercis*", "strenuous activity", "strenuous exercise", "sports",
                 "working out", "work out", "run*", "jog*", "gym", "weights"),
    "work": ("work", "your job", "return to your job", "back to your job"),
    "sex": ("sexual activity", "intercourse", "sex"),
    "stairs": ("stair*",),
    "nsaids": ("ibuprofen", "advil", "motrin", "naproxen", "aleve", "aspirin",
               "anti-inflammator*", "nsaid*"),
    "screens": ("screen*", "phone", "tablet screen", "television", "computer"),
    "air travel": ("air travel", "fly*", "flight*", "aeroplane", "airplane"),
    "sun exposure": ("sun exposure", "sunlight", "prolonged sun", "sunbathing",
                     "tanning"),
    "reaching or pushing": ("reach behind", "reach above", "reaching behind",
                            "reaching above", "push up from", "pushing",
                            "pulling", "over your head"),
    # Named for what the three notes using it actually forbid: one says lotion
    # or powder, one says ointment or cream, one says topical antibiotic
    # ointment. Only one of the three is an ointment.
    "topical products on the wound": ("ointment*", "cream*", "lotion*",
                                      "powder", "salve*"),
    "prolonged immobility": ("prolonged immobility", "sitting still",
                             "staying still", "immobility",
                             "sitting or lying still"),
    "wound wetting": ("get the incision wet", "get your incision wet",
                      "get the wound wet", "get your wound wet",
                      "wet the incision", "wet the wound", "get the cut wet",
                      "get your cut wet"),
}
