"""
Text Preprocessing Module for Sentiment Analysis.

Includes:
- HTML / URL / Mentions cleaning
- Contraction expansion
- Emoticon / Emoji translation
- Tokenization (NLTK or robust regex fallback)
- Stopword filtering with negation preservation
"""

import re
import html
from typing import List, Optional

# Common English contractions mapping
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}

# Emoticons mapped to descriptive sentiment tokens
EMOTICONS = {
    ":-)": " emoji_happy ",
    ":)": " emoji_happy ",
    ":D": " emoji_happy ",
    ";)": " emoji_happy ",
    ":-]": " emoji_happy ",
    ":]": " emoji_happy ",
    "(:": " emoji_happy ",
    "(-:": " emoji_happy ",
    ":-(": " emoji_sad ",
    ":(": " emoji_sad ",
    ":-[": " emoji_sad ",
    ":[": " emoji_sad ",
    ":/": " emoji_sad ",
    ":\\": " emoji_sad ",
    ":'(": " emoji_sad ",
    ":'-(": " emoji_sad ",
    ":-O": " emoji_surprise ",
    ":O": " emoji_surprise ",
    ":-o": " emoji_surprise ",
    ":o": " emoji_surprise ",
    ":-p": " emoji_playful ",
    ":p": " emoji_playful ",
    ":-P": " emoji_playful ",
    ":P": " emoji_playful ",
}

# Words indicating negation that MUST NOT be removed during stopword filtering
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "hardly", "scarcely",
    "barely", "without", "against", "cannot", "n't", "nothing", "nowhere"
}

# Standard English stopwords (excluding negations)
BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves"
}

# Final stopword set: base stopwords minus critical negation words
FILTER_STOPWORDS = BASE_STOPWORDS - NEGATION_WORDS


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    lower_text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        if contraction in lower_text:
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(expansion, text)
    return text


def replace_emoticons(text: str) -> str:
    """Replace ASCII emoticons with sentiment tokens."""
    for emo, replacement in EMOTICONS.items():
        text = text.replace(emo, replacement)
    return text


def clean_text(text: str) -> str:
    """
    Remove URLs, HTML tags, user mentions, and extraneous characters.
    Preserves alphanumeric characters, sentiment tokens, and basic punctuation spacing.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)

    # Convert emoticons
    text = replace_emoticons(text)

    # Expand contractions
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove Twitter-style @mentions and hashtags
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    # Lowercase
    text = text.lower()

    # Normalize repeated punctuation (!+, ?+, ..+) to single marker
    text = re.sub(r"[!]{2,}", " exclamation_mark ", text)
    text = re.sub(r"[\?]{2,}", " question_mark ", text)

    # Replace remaining punctuation with whitespace, preserving words and underscores
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text using NLTK word_tokenize if installed, else regex boundary matching.
    """
    try:
        # pyrefly: ignore[missing-import]
        import nltk
        # pyrefly: ignore[missing-import]
        from nltk.tokenize import word_tokenize
        try:
            return word_tokenize(text)
        except LookupError:
            # If punkt resource isn't downloaded, use regex fallback
            return re.findall(r"\b[a-zA-Z_]{2,}\b", text)
    except ImportError:
        return re.findall(r"\b[a-zA-Z_]{2,}\b", text)


def mark_negations(tokens: List[str]) -> List[str]:
    """
    Transform negation + word sequences into compound tokens (e.g. 'not good' -> 'not_good').
    This preserves sentiment polarity through bag-of-words / TF-IDF.
    """
    output = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token in NEGATION_WORDS and i + 1 < n:
            next_token = tokens[i + 1]
            # Link negation with following descriptive token
            output.append(token)
            output.append(f"{token}_{next_token}")
            i += 2
        else:
            output.append(token)
            i += 1
    return output


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filter stopwords while strictly retaining negations and sentiment tokens.
    """
    return [
        token for token in tokens
        if (token not in FILTER_STOPWORDS and (len(token) > 1 or token in NEGATION_WORDS))
        or "_" in token
    ]


def preprocess_text(text: str, remove_stop: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline for sentiment text.
    Returns space-separated processed tokens.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = mark_negations(tokens)
    if remove_stop:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


class TextPreprocessor:
    """Reusable class interface for scikit-learn pipeline or standalone usage."""

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def transform(self, texts: List[str]) -> List[str]:
        return [preprocess_text(t, remove_stop=self.remove_stopwords) for t in texts]

    def __call__(self, text: str) -> str:
        return preprocess_text(text, remove_stop=self.remove_stopwords)
