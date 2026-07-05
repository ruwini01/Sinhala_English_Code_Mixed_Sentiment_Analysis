"""
Sinhala informal spelling normalizer.
Maps social-media variants → canonical forms to reduce vocabulary.
"""

SINHALA_NORM_DICT = {
    # intensifiers
    "හරිිිම": "හරිම",
    "හරිීීම": "හරිම",
    "හරිිම": "හරිම",
    "ගොඩාක්ක": "ගොඩාක්",
    "ගොඩාක": "ගොඩාක්",
    "ගොඩක්ක": "ගොඩාක්",
    "ගොඩක": "ගොඩාක්",
    "තිකක්": "ටිකක්",
    "තිකක": "ටිකක්",
    "හොඳාක": "හොඳ",
    # negations
    "නෑෑ": "නෑ",
    "නෑෑෑ": "නෑ",
    "නෙවෙයි": "නෙවේ",
    "නෙවෙ": "නෙවේ",
    "නෙමෙ": "නෙමේ",
    # common emotion words
    "ආස්සා": "ආසා",
    "ආස්": "ආසා",
    "ආස": "ආසා",
    "ලස්සනයි": "ලස්සනයි",
    "ලස්සනෙ": "ලස්සනයි",
    # common informal
    "දෙයියනේ": "දෙවියනේ",
    "ඔව්ව": "ඔව්",
    "ඔව්වා": "ඔව්",
    "නෑනෑ": "නෑ",
    "ඇත්තතෙන්ම": "ඇත්තෙන්ම",
    "ඇත්තතටම": "ඇත්තටම",
    "සුපිරිය": "සුපිරි",
    "සුපිරිි": "සුපිරි",
    "කොහොමද": "කොහොමද",
    "කොහමද": "කොහොමද",
    "මොකද": "මොකද",
    "මෝකද": "මොකද",
}

# Transliterated / Romanized Sinhala variants → canonical Sinhala
TRANSLIT_NORM_DICT = {
    "harima": "harima",
    "hariiima": "harima",
    "hariima": "harima",
    "godak": "godak",
    "godaak": "godak",
    "tikak": "tikak",
    "honda": "honda",
    "nehe": "nehe",
    "ne": "ne",
    "epa": "epa",
    "asa": "asa",
    "lassanai": "lassanai",
    "lasanai": "lassanai",
}


def normalize_sinhala(text: str) -> str:
    """Replace known informal Sinhala spellings with canonical forms."""
    for variant, canonical in SINHALA_NORM_DICT.items():
        text = text.replace(variant, canonical)
    words = text.split()
    normalized = [TRANSLIT_NORM_DICT.get(w.lower(), w) for w in words]
    return " ".join(normalized)
