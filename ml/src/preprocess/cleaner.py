"""
Text cleaning pipeline for Sinhala-English code-mixed social media text.

Steps (in order):
  1. Unicode NFC normalization
  2. URL / mention / hashtag / HTML removal
  3. Repeated character normalization  (booooring → booring, max 2 repeats)
  4. Emoji handling  (demojize for BiLSTM; keep raw for XLM-R)
  5. Sinhala informal spelling normalization
  6. Whitespace normalization
"""

import re
import unicodedata

from src.preprocess.sinhala_normalizer import normalize_sinhala

try:
    import emoji as emoji_lib
    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False

# ── regex patterns ──────────────────────────────────────────────────────────
_URL_RE = re.compile(
    r"http[s]?://\S+|www\.\S+",
    re.IGNORECASE,
)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_HTML_RE = re.compile(r"<[^>]+>|&[a-z]+;|&#\d+;")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")   # 3+ of same char → keep 2
_WHITESPACE_RE = re.compile(r"\s+")


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_noise(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    return text


def normalize_repeated_chars(text: str) -> str:
    return _REPEATED_CHAR_RE.sub(r"\1\1", text)


def handle_emojis(text: str, demojize: bool = False) -> str:
    """
    demojize=False (default, for XLM-R): keep emojis as-is.
    demojize=True (for BiLSTM / TF-IDF):  convert to :text_description:.
    """
    if demojize and _EMOJI_AVAILABLE:
        return emoji_lib.demojize(text, delimiters=(" :", ": "))
    return text


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, demojize: bool = False) -> str:
    """Full cleaning pipeline for one text string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicode_normalize(text)
    text = remove_noise(text)
    text = normalize_repeated_chars(text)
    text = handle_emojis(text, demojize=demojize)
    text = normalize_sinhala(text)
    text = normalize_whitespace(text)
    return text
