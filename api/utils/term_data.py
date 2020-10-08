from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

PUNCTUATION_CUSTOM = [
    ";",
    ":",
    ",",
    "\.",
    '"',
    "'",
    "\?",
    "\(",
    "\)",
    "!",
    "？",
    "！",
    "；",
    "：",
    "。",
    "、",
    "《",
    "》",
    "，",
    "¿",
    "¡",
    "؟",
    "،",
]

PUNCTUATION = PUNCTUATION_CUSTOM

STOP_WORDS_SKLEARN = list(stop_words.ENGLISH_STOP_WORDS)

STOP_WORDS_NLTK = stopwords.words('english')

STOP_WORDS_CUSTOM = [
    "an",
    "a",
    "in",
    "on",
    "be",
    "or",
    "of",
    "and",
    "can",
    "is",
    "to",
    "the",
    "i"
]

STOP_WORDS = STOP_WORDS_SKLEARN + STOP_WORDS_NLTK + STOP_WORDS_CUSTOM