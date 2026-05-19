"""Text preprocessing helpers for the information retrieval system.

The functions in this module normalize raw text into tokens that work well with
TF-IDF ranking. They also handle the required NLTK resource setup once and then
reuse the same tokenizer across the app.
"""

import re
from functools import lru_cache

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


@lru_cache(maxsize=1)
def ensure_nltk_resources():
    """Download the NLTK resources needed by the tokenizer if they are missing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def get_wordnet_pos(tag):
    """Map a Penn Treebank POS tag to the WordNet tag used by the lemmatizer."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess(text):
    """Convert raw text into a cleaned list of lemmatized tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]

    tagged_tokens = pos_tag(tokens)

    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]
