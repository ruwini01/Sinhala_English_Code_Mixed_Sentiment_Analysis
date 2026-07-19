"""
Language Identification (LID) tagger.

Tags each token in a sentence as:
  [SI]  — Sinhala (Unicode range U+0D80–U+0DFF)
  [EN]  — English / Latin script
  [EM]  — Emoji
  [NUM] — Number / digit sequence
  [PUN] — Punctuation / other symbol

Also computes the Code-Mixing Index (CMI) per sentence.
"""

import re
import unicodedata

SINHALA_RANGE = (0x0D80, 0x0DFF)
LATIN_RANGE_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"^\d+([.,]\d+)?$")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _is_sinhala_token(token: str) -> bool:
    return any(SINHALA_RANGE[0] <= ord(ch) <= SINHALA_RANGE[1] for ch in token)


def _is_latin_token(token: str) -> bool:
    return bool(LATIN_RANGE_RE.search(token)) and not _is_sinhala_token(token)


def _is_emoji_token(token: str) -> bool:
    return bool(EMOJI_RE.fullmatch(token))


def _is_number_token(token: str) -> bool:
    return bool(DIGIT_RE.match(token))


def tag_token(token: str) -> str:
    """Return the language tag for a single token."""
    if _is_emoji_token(token):
        return "EM"
    if _is_number_token(token):
        return "NUM"
    if _is_sinhala_token(token):
        return "SI"
    if _is_latin_token(token):
        return "EN"
    return "PUN"


def tag_sentence(text: str) -> list[tuple[str, str]]:
    """
    Tokenize by whitespace and return list of (token, tag) pairs.
    """
    tokens = text.split()
    return [(tok, tag_token(tok)) for tok in tokens]


def compute_cmi(tags: list[str]) -> float:
    """
    Code-Mixing Index: fraction of tokens that are NOT the majority language.
    Ignores PUN, NUM, EM in the calculation.
    Returns value in [0, 1].  0 = monolingual, 1 = maximally mixed.
    """
    lang_tags = [t for t in tags if t in ("SI", "EN")]
    if len(lang_tags) < 2:
        return 0.0
    si_count = lang_tags.count("SI")
    en_count = lang_tags.count("EN")
    majority = max(si_count, en_count)
    return round(1 - majority / len(lang_tags), 4)


def tag_and_enrich(text: str) -> dict:
    """
    Full LID enrichment for one sentence.
    Returns dict with:
      tagged_text   — original text with tags appended per token
      token_tags    — list of (token, tag) pairs
      lid_sequence  — space-joined tag sequence (for model input)
      n_sinhala     — count of Sinhala tokens
      n_english     — count of English tokens
      n_emoji       — count of emoji tokens
      cmi           — code-mixing index
      is_code_mixed — bool: True if both SI and EN tokens present
    """
    pairs = tag_sentence(text)
    tags = [tag for _, tag in pairs]

    tagged_tokens = [f"{tok}[{tag}]" for tok, tag in pairs]

    return {
        "tagged_text": " ".join(tagged_tokens),
        "token_tags": pairs,
        "lid_sequence": " ".join(tags),
        "n_sinhala": tags.count("SI"),
        "n_english": tags.count("EN"),
        "n_emoji": tags.count("EM"),
        "cmi": compute_cmi(tags),
        "is_code_mixed": "SI" in tags and "EN" in tags,
    }
