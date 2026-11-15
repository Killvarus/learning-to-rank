"""
Метрики ранжирования. Считаются внутри каждого запроса и усредняются по запросам.

NDCG@k = DCG@k / IDCG@k,  DCG@k = Σ_{i ≤ k} (2^rel_i − 1) / log2(i + 1).
MRR    — 1 / позиция первого товара с оценкой Exact.
P@k    — доля Exact среди первых k.
Равные прогнозы упорядочиваются случайно (фиксированный seed): иначе константа «ранжирует»
товары по порядку строк в файле.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    return float(((2.0 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))).sum())


def per_query(frame: pd.DataFrame, score_col: str = "score", k: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for qid, group in frame.groupby("query_id", sort=True):
        rel = group["rel"].to_numpy()
        tiebreak = rng.random(len(group))
        order = np.lexsort((tiebreak, -group[score_col].to_numpy()))
        ranked = rel[order]
        ideal = _dcg(np.sort(rel)[::-1], k)
        exact = np.flatnonzero(ranked == 2)
        rows.append({
            "query_id": qid,
            f"ndcg@{k}": _dcg(ranked, k) / ideal if ideal > 0 else np.nan,
            "ndcg@5": _dcg(ranked, 5) / _dcg(np.sort(rel)[::-1], 5) if ideal > 0 else np.nan,
            "mrr": 1.0 / (exact[0] + 1) if len(exact) else np.nan,
            f"p@{k}": float((ranked[:k] == 2).mean()),
        })
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, score_col: str = "score", k: int = 10) -> dict[str, float]:
    return per_query(frame, score_col, k).drop(columns="query_id").mean(skipna=True).to_dict()
