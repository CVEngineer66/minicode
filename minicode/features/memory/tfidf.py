from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    total = len(tokens)
    return {term: count / total for term, count in Counter(tokens).items()}


def _idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents)
    if n == 0:
        return {}
    df: dict[str, int] = {}
    for doc in documents:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (d + 1)) + 1 for term, d in df.items()}


def tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = _tf(doc_tokens)
    return sum(tf.get(t, 0.0) * idf.get(t, 0.0) for t in query_tokens)


def compute_corpus_idf(documents: list[list[str]]) -> dict[str, float]:
    return _idf(documents)
