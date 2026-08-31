LABELS = [
    "loaded_language",
    "name_calling",
    "exaggeration_minimization",
    "appeal_to_fear",
    "unsupported_claim",
    "factual_neutral",
]

LABEL_DESCRIPTIONS = {
    "loaded_language": "Emotionally charged words/phrases chosen to influence rather than inform",
    "name_calling": "Attacking a person/group with a label instead of addressing their argument",
    "exaggeration_minimization": "Overstating or understating something's importance or scale",
    "appeal_to_fear": "Framing built around provoking fear or alarm rather than evidence",
    "unsupported_claim": "An assertion presented as fact with no evidence or citation",
    "factual_neutral": "Sentence presents verifiable information without rhetorical framing",
}

TECHNIQUE_TO_LABEL = {
    "Loaded_Language": "loaded_language",
    "Name_Calling,Labeling": "name_calling",
    "Exaggeration,Minimisation": "exaggeration_minimization",
    "Appeal_to_fear-prejudice": "appeal_to_fear",
    "Doubt": "unsupported_claim",
    "Flag-Waving": "unsupported_claim",
    "Causal_Oversimplification": "unsupported_claim",
    "Slogans": "unsupported_claim",
    "Appeal_to_Authority": "unsupported_claim",
    "Black-and-White_Fallacy": "unsupported_claim",
    "Thought-terminating_Cliches": "unsupported_claim",
    "Whataboutism,Straw_Men,Red_Herring": "unsupported_claim",
    "Reductio_ad_hitlerum": "unsupported_claim",
    "Bandwagon": "unsupported_claim",
    "Bandwagon,Reductio_ad_hitlerum": "unsupported_claim",
    "Obfuscation,Intentional_Vagueness,Confusion": "unsupported_claim",
    "Repetition": "unsupported_claim",
}

NUM_LABELS = len(LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
