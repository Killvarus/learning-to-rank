"""
Признаки пары «запрос — товар». Все обучаются только на текстах каталога и запросов, без меток
релевантности, поэтому их можно посчитать один раз для всех пар без утечки между фолдами.

Группы (для абляции):
  lexical   — BM25 по полям, косинусы TF-IDF по словам и символьным n-граммам;
  overlap   — доля слов запроса в названии / классе / категории / описании, вхождение фразы;
  qclass    — совпадение класса запроса (из query.csv) с классом товара;
  meta      — длины полей, рейтинг и число отзывов;
  semantic  — LSA (SVD над TF-IDF) и BERT-эмбеддинги (all-MiniLM-L6-v2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .text import BM25, rowwise_cosine, tokenize

GROUPS = {
    "lexical": ["bm25_name", "bm25_class", "bm25_category", "bm25_desc", "bm25_features", "bm25_all",
                "tfidf_word_name", "tfidf_char_name", "tfidf_char_all"],
    "overlap": ["q_len", "q_in_name", "q_all_in_name", "q_phrase_in_name", "q_in_class", "q_in_category", "q_in_desc"],
    "qclass": ["qclass_eq", "qclass_in_category", "qclass_char_cos"],
    "meta": ["name_len", "desc_len", "n_features", "avg_rating", "log_rating_count", "log_review_count"],
    "semantic": ["lsa_all", "bert_name"],
}
ALL_FEATURES = [f for group in GROUPS.values() for f in group]
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK = 40_000


def _chunked(fn, n: int) -> np.ndarray:
    return np.concatenate([fn(slice(i, min(i + CHUNK, n))) for i in range(0, n, CHUNK)])


def _pair_cosine(vec, q_text, d_text, q_idx, d_idx) -> np.ndarray:
    Q, D = vec.transform(q_text), vec.transform(d_text)
    return _chunked(lambda s: rowwise_cosine(Q[q_idx[s]], D[d_idx[s]]), len(q_idx))


def _overlap_share(field_text, q_text, q_idx, d_idx) -> tuple[np.ndarray, np.ndarray]:
    """Сколько разных слов запроса встречается в поле товара: доля и число."""
    vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, binary=True)
    D = vec.fit_transform(field_text).tocsr()
    Q = vec.transform(q_text).tocsr()
    q_len = np.array([len(set(tokenize(t))) for t in q_text], float)
    hits = _chunked(lambda s: np.asarray(D[d_idx[s]].multiply(Q[q_idx[s]]).sum(1)).ravel(), len(q_idx))
    return hits / np.maximum(q_len[q_idx], 1), hits


def bert_embeddings(texts: list[str], cache: Path, batch_size: int = 256) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BERT_MODEL, device="cpu")
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    np.save(cache, emb)
    return emb


def build(queries: pd.DataFrame, products: pd.DataFrame, labels: pd.DataFrame, cache_dir: Path,
          use_bert: bool = True) -> pd.DataFrame:
    queries = queries.set_index("query_id")
    products = products.reset_index(drop=True)
    pos = pd.Series(np.arange(len(products)), index=products["product_id"])
    qpos = pd.Series(np.arange(len(queries)), index=queries.index)
    d_idx = pos.loc[labels["product_id"]].to_numpy()
    q_idx = qpos.loc[labels["query_id"]].to_numpy()
    q_text = queries["query"].tolist()
    q_class = queries["query_class"].tolist()
    pair_q = [q_text[i] for i in q_idx]

    name, pclass, cat = products["product_name"], products["product_class"], products["category"]
    desc, feats = products["product_description"], products["features_text"]
    all_text = (name + " " + pclass + " " + cat + " " + desc + " " + feats).tolist()
    short_text = (name + " " + pclass + " " + cat).tolist()

    out = pd.DataFrame({"query_id": labels["query_id"].to_numpy(), "product_id": labels["product_id"].to_numpy(),
                        "rel": labels["rel"].to_numpy()})
    # ---- lexical
    for col, field in (("bm25_name", name), ("bm25_class", pclass), ("bm25_category", cat), ("bm25_desc", desc),
                       ("bm25_features", feats), ("bm25_all", all_text)):
        bm = BM25().fit(list(field))
        out[col] = _chunked(lambda s: bm.score_pairs(pair_q[s], d_idx[s]), len(d_idx))
    word = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True).fit(name)
    out["tfidf_word_name"] = _pair_cosine(word, q_text, name, q_idx, d_idx)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(name)
    out["tfidf_char_name"] = _pair_cosine(char, q_text, name, q_idx, d_idx)
    char_all = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(short_text)
    out["tfidf_char_all"] = _pair_cosine(char_all, q_text, short_text, q_idx, d_idx)

    # ---- overlap
    q_len = np.array([len(set(tokenize(t))) for t in q_text], float)
    out["q_len"] = q_len[q_idx]
    out["q_in_name"], hits = _overlap_share(name, q_text, q_idx, d_idx)
    out["q_all_in_name"] = (hits >= q_len[q_idx]).astype(float)
    norm_name = [" ".join(tokenize(t, drop_stop=False)) for t in name]
    norm_q = [" ".join(tokenize(t, drop_stop=False)) for t in q_text]
    out["q_phrase_in_name"] = np.array([norm_q[q] in norm_name[d] for q, d in zip(q_idx, d_idx)], float)
    out["q_in_class"], _ = _overlap_share(pclass, q_text, q_idx, d_idx)
    out["q_in_category"], _ = _overlap_share(cat, q_text, q_idx, d_idx)
    out["q_in_desc"], _ = _overlap_share(desc, q_text, q_idx, d_idx)

    # ---- qclass: класс запроса дан в датасете (результат классификатора запросов Wayfair)
    pclass_l, cat_l = pclass.str.lower().tolist(), cat.str.lower().tolist()
    qclass_l = [c.lower() for c in q_class]
    out["qclass_eq"] = np.array([qclass_l[q] != "" and qclass_l[q] == pclass_l[d] for q, d in zip(q_idx, d_idx)], float)
    out["qclass_in_category"] = np.array([qclass_l[q] != "" and qclass_l[q] in cat_l[d] for q, d in zip(q_idx, d_idx)], float)
    cls_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(list(pclass) + q_class)
    out["qclass_char_cos"] = _pair_cosine(cls_vec, q_class, pclass, q_idx, d_idx)

    # ---- meta
    out["name_len"] = name.str.split().str.len().to_numpy()[d_idx]
    out["desc_len"] = np.log1p(desc.str.split().str.len().to_numpy()[d_idx])
    out["n_features"] = (products["product_features"].str.count(r"\|") + 1).to_numpy()[d_idx]
    out["avg_rating"] = products["average_rating"].to_numpy()[d_idx]
    out["log_rating_count"] = np.log1p(products["rating_count"].to_numpy()[d_idx])
    out["log_review_count"] = np.log1p(products["review_count"].to_numpy()[d_idx])

    # ---- semantic
    tfidf_all = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=2, sublinear_tf=True)
    D = tfidf_all.fit_transform(all_text)
    svd = TruncatedSVD(n_components=min(150, D.shape[1] - 1), random_state=0).fit(D)
    D_lsa, Q_lsa = svd.transform(D), svd.transform(tfidf_all.transform(q_text))
    out["lsa_all"] = rowwise_cosine(Q_lsa[q_idx], D_lsa[d_idx])
    if use_bert:
        cache_dir.mkdir(parents=True, exist_ok=True)
        q_emb = bert_embeddings(q_text, cache_dir / "bert_queries.npy")
        d_emb = bert_embeddings((name + " | " + pclass).tolist(), cache_dir / "bert_products.npy")
        out["bert_name"] = (q_emb[q_idx] * d_emb[d_idx]).sum(1)
    else:
        out["bert_name"] = 0.0
    return out
