"""Pickle-compatible estimators: TF-IDF selection plus linguistic features."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean_text

LINGUISTIC_NAMES = ["word_count", "sentence_count", "mean_word_length", "mean_sentence_length",
                    "uppercase_ratio", "punctuation_ratio", "exclamations_per_word",
                    "questions_per_word"]


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t) for t in X]


class SafeChi2Selector(BaseEstimator, TransformerMixin):
    """Allow requested k to exceed a small dataset's vocabulary."""
    def __init__(self, k=6000):
        self.k = k

    def fit(self, X, y):
        self.selector_ = SelectKBest(chi2, k=min(self.k, X.shape[1]) if self.k != "all" else "all")
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            words = re.findall(r"\b\w+\b", text)
            count = max(len(words), 1)
            sentences = max(len(re.findall(r"[.!?]+", text)), 1)
            letters = max(sum(c.isalpha() for c in text), 1)
            rows.append([len(words), sentences, sum(map(len, words)) / count,
                         len(words) / sentences, sum(c.isupper() for c in text) / letters,
                         sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1),
                         text.count("!") / count, text.count("?") / count])
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(LINGUISTIC_NAMES))


def make_features(k=6000, linguistic=True):
    text = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000,
                                  min_df=3, max_df=.98, sublinear_tf=True, dtype=np.float32)),
        ("select", SafeChi2Selector(k=k)),
    ])
    branches = [("text", text)]
    if linguistic:
        branches.append(("linguistic", Pipeline([
            ("extract", LinguisticFeatures()), ("scale", StandardScaler())
        ])))
    return Pipeline([("clean", TextCleaner()), ("union", FeatureUnion(branches))])
