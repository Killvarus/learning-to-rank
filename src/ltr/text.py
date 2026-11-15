"""
Текстовые меры близости запроса и товара: токенизация, BM25 с нуля, TF-IDF.

BM25 (Robertson, Zaragoza):
    score(q, d) = Σ_{t ∈ q} idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d| / avgdl))
    idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
k1 ограничивает вклад повторов слова, b штрафует длинные документы.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_token(token: str) -> str:
    """Грубая нормализация множественного числа: chairs → chair, benches → bench, shelves → shelf."""
    if len(token) > 4 and token.endswith("ves"):
        return token[:-3] + "f"
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [normalize_token(t) for t in tokens if not (drop_stop and t in ENGLISH_STOP_WORDS)]


class BM25:
    """BM25 по одному полю каталога. Вес каждого слова в каждом документе считается заранее,
    тогда оценка пары — скалярное произведение строки матрицы весов и мешка слов запроса."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs: Sequence[str]) -> BM25:
        self.vectorizer_ = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer_.fit_transform(docs).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf_ = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        weights = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * self.idf_[tf.indices]
        self.weights_ = sparse.csr_matrix((weights, tf.indices, tf.indptr), shape=tf.shape)
        return self

    def score_pairs(self, queries: Sequence[str], doc_idx: np.ndarray) -> np.ndarray:
        """Оценка для пар (queries[i], документ doc_idx[i])."""
        q = self.vectorizer_.transform(queries)
        q.data[:] = 1.0  # повтор слова в запросе не удваивает его вес
        return np.asarray(self.weights_[doc_idx].multiply(q).sum(1)).ravel()


def rowwise_cosine(a, b) -> np.ndarray:
    """Косинус между соответствующими строками двух матриц (разреженных или плотных)."""
    if sparse.issparse(a):
        dot = np.asarray(a.multiply(b).sum(1)).ravel()
        na = np.sqrt(np.asarray(a.multiply(a).sum(1)).ravel())
        nb = np.sqrt(np.asarray(b.multiply(b).sum(1)).ravel())
    else:
        dot = (a * b).sum(1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return dot / np.maximum(na * nb, 1e-12)
