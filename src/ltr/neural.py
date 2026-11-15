"""
Нейросетевые ранжировщики на PyTorch.

RankNet (Burges, 2005) — попарный подход: для пары товаров одного запроса с rel_i > rel_j
    P(i выше j) = σ(s_i − s_j),  loss = −log σ(s_i − s_j).
ListNet (Cao et al., 2007) — списочный подход: распределения «кто окажется первым»
    P_model = softmax(s),  P_true = softmax(rel)  →  loss = кросс-энтропия между ними.
Обе модели — MLP над теми же признаками, что у бустинга.

DSSM (Huang et al., 2013) — две башни над сырым текстом: запрос и товар хешируются в
буквенные триграммы («#ch», «cha», «hai», ...), каждая башня — мешок триграмм → MLP →
вектор; оценка — косинус. Обучение — ListNet-лосс по товарам запроса.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import torch
from torch import nn


def _groups(qids: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(qids, kind="stable")
    bounds = np.flatnonzero(np.diff(qids[order])) + 1
    return np.split(order, bounds)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def ranknet_loss(scores: torch.Tensor, rel: torch.Tensor, max_pairs: int = 2000) -> torch.Tensor:
    diff_rel = rel[:, None] - rel[None, :]
    i, j = torch.nonzero(diff_rel > 0, as_tuple=True)
    if len(i) == 0:
        return scores.sum() * 0
    if len(i) > max_pairs:  # в запросе бывают сотни тысяч пар — берём случайную подвыборку
        keep = torch.randperm(len(i))[:max_pairs]
        i, j = i[keep], j[keep]
    return nn.functional.softplus(-(scores[i] - scores[j])).mean()  # = −log σ(s_i − s_j)


def listnet_loss(scores: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    target = torch.softmax(rel.float(), 0)
    return -(target * torch.log_softmax(scores, 0)).sum()


class NeuralRanker:
    """Обучение MLP по запросам: один шаг оптимизатора — один запрос (все его товары)."""

    def __init__(self, loss: str = "listnet", epochs: int = 30, lr: float = 1e-3, hidden: int = 64, seed: int = 0):
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.seed = seed

    def fit(self, X: np.ndarray, rel: np.ndarray, qids: np.ndarray) -> NeuralRanker:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.nan_to_num(np.asarray(X, np.float32))
        self.mean_, self.std_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy((X - self.mean_) / self.std_)
        rt = torch.from_numpy(np.asarray(rel, np.float32))
        self.model_ = MLP(X.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = ranknet_loss if self.loss == "ranknet" else listnet_loss
        groups = _groups(np.asarray(qids))
        self.model_.train()
        for _ in range(self.epochs):
            for g in rng.permutation(len(groups)):
                idx = torch.from_numpy(groups[g])
                loss = loss_fn(self.model_(Xt[idx]), rt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, np.float32))
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy((X - self.mean_) / self.std_)).numpy()


# ---------------------------------------------------------------- DSSM

def trigram_ids(text: str, n_buckets: int) -> list[int]:
    """Слова → буквенные триграммы с границами слова → номера корзин хеша."""
    ids = []
    for word in text.lower().split():
        padded = f"#{word}#"
        ids.extend(zlib.crc32(padded[i:i + 3].encode()) % n_buckets for i in range(max(len(padded) - 2, 1)))
    return ids or [0]


class Tower(nn.Module):
    def __init__(self, n_buckets: int, dim: int = 128, out: int = 64):
        super().__init__()
        # sparse=True: градиент только по триграммам из батча, иначе Adam обновляет всю таблицу на каждом шаге
        self.bag = nn.EmbeddingBag(n_buckets, dim, mode="mean", sparse=True)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(dim, out))

    def forward(self, ids, offsets):
        return nn.functional.normalize(self.mlp(self.bag(ids, offsets)), dim=-1)


def _pack(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor(np.concatenate([[0], np.cumsum([len(s) for s in seqs])[:-1]]), dtype=torch.long)
    return torch.tensor(np.concatenate(seqs), dtype=torch.long), offsets


class DSSM:
    """shared=True — одна башня для запроса и товара (сиамская сеть). С раздельными башнями случайная
    инициализация проецирует запрос и товар в несогласованные пространства, а 384 обучающих запросов
    мало, чтобы их согласовать: башня запросов просто запоминает обучающие запросы."""

    def __init__(self, n_buckets: int = 2 ** 17, epochs: int = 8, lr: float = 3e-3, temperature: float = 0.05,
                 shared: bool = True, seed: int = 0):
        self.shared = shared
        self.n_buckets = n_buckets
        self.epochs = epochs
        self.lr = lr
        self.temperature = temperature
        self.seed = seed

    def _ids(self, texts: dict, keys) -> list[list[int]]:
        cache = self.cache_
        out = []
        for k in keys:
            if k not in cache:
                cache[k] = trigram_ids(texts[k], self.n_buckets)
            out.append(cache[k])
        return out

    def fit(self, pairs: pd.DataFrame, q_text: dict, d_text: dict) -> DSSM:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.q_cache_, self.d_cache_ = {}, {}
        self.q_tower_ = Tower(self.n_buckets)
        self.d_tower_ = self.q_tower_ if self.shared else Tower(self.n_buckets)
        towers = (self.q_tower_,) if self.shared else (self.q_tower_, self.d_tower_)
        sparse_opt = torch.optim.SparseAdam([p for t in towers for p in t.bag.parameters()], lr=self.lr)
        dense_opt = torch.optim.Adam([p for t in towers for p in t.mlp.parameters()], lr=self.lr)
        groups = list(pairs.groupby("query_id"))
        for _ in range(self.epochs):
            for g in rng.permutation(len(groups)):
                qid, group = groups[g]
                if group["rel"].nunique() < 2:
                    continue
                if len(group) > 256:
                    group = group.sample(256, random_state=int(rng.integers(1 << 30)))
                self.cache_ = self.q_cache_
                q = self.q_tower_(*_pack(self._ids(q_text, [qid])))
                self.cache_ = self.d_cache_
                d = self.d_tower_(*_pack(self._ids(d_text, group["product_id"])))
                scores = (d @ q.T).squeeze(-1) / self.temperature
                loss = listnet_loss(scores, torch.tensor(group["rel"].to_numpy(), dtype=torch.float32))
                sparse_opt.zero_grad()
                dense_opt.zero_grad()
                loss.backward()
                sparse_opt.step()
                dense_opt.step()
        return self

    def predict(self, pairs: pd.DataFrame, q_text: dict, d_text: dict) -> np.ndarray:
        with torch.no_grad():
            qids = pairs["query_id"].unique()
            self.cache_ = self.q_cache_
            q_emb = dict(zip(qids, self.q_tower_(*_pack(self._ids(q_text, qids))).numpy()))
            out = np.empty(len(pairs), np.float32)
            pids = pairs["product_id"].to_numpy()
            for start in range(0, len(pairs), 20_000):
                sl = slice(start, start + 20_000)
                self.cache_ = self.d_cache_
                d = self.d_tower_(*_pack(self._ids(d_text, pids[sl]))).numpy()
                q = np.stack([q_emb[x] for x in pairs["query_id"].to_numpy()[sl]])
                out[sl] = (d * q).sum(1)
        return out
