import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from ltr.features import ALL_FEATURES, build
from ltr.metrics import evaluate, per_query
from ltr.neural import DSSM, NeuralRanker, listnet_loss, ranknet_loss, trigram_ids
from ltr.text import BM25, normalize_token, rowwise_cosine, tokenize


def test_tokenize_and_plurals():
    assert tokenize("Wooden Chairs and Benches") == ["wooden", "chair", "bench"]
    assert normalize_token("shelves") == "shelf" and normalize_token("glass") == "glass"


def test_bm25_matches_formula():
    docs = ["red chair", "red red sofa table", "blue lamp"]
    bm = BM25(k1=1.2, b=0.75).fit(docs)
    score = bm.score_pairs(["red sofa"], np.array([1]))[0]
    n, avgdl, dl = 3, (2 + 4 + 2) / 3, 4
    idf = lambda df: np.log1p((n - df + 0.5) / (df + 0.5))  # noqa: E731
    tf_part = lambda tf: tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))  # noqa: E731
    assert score == pytest.approx(idf(2) * tf_part(2) + idf(1) * tf_part(1))
    assert bm.score_pairs(["red sofa"], np.array([2]))[0] == 0


def test_rowwise_cosine_sparse_and_dense():
    a = np.array([[1.0, 0, 0], [1, 1, 0]])
    b = np.array([[1.0, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(rowwise_cosine(a, b), [1, 1 / np.sqrt(2)])
    np.testing.assert_allclose(rowwise_cosine(sparse.csr_matrix(a), sparse.csr_matrix(b)), [1, 1 / np.sqrt(2)])


def test_ndcg_perfect_and_reversed():
    frame = pd.DataFrame({"query_id": [0] * 4, "rel": [2, 1, 0, 0], "score": [4, 3, 2, 1]})
    m = evaluate(frame)
    assert m["ndcg@10"] == pytest.approx(1) and m["mrr"] == 1
    worst = evaluate(frame.assign(score=[1, 2, 3, 4]))
    ideal = 3 + 1 / np.log2(3)
    assert worst["ndcg@10"] == pytest.approx((1 / np.log2(4) + 3 / np.log2(5)) / ideal)
    assert worst["mrr"] == pytest.approx(1 / 4)


def test_constant_score_ties_are_random_not_file_order():
    frame = pd.DataFrame({"query_id": np.repeat(np.arange(200), 5), "rel": np.tile([2, 0, 0, 0, 0], 200), "score": 0.0})
    # при порядке строк релевантный документ всегда был бы первым и NDCG = 1
    assert per_query(frame)["ndcg@10"].mean() < 0.8


def test_losses():
    scores = torch.tensor([3.0, 1.0, 0.0], requires_grad=True)
    rel = torch.tensor([2.0, 1.0, 0.0])
    assert ranknet_loss(scores, rel) < ranknet_loss(-scores, rel)
    assert listnet_loss(scores, rel) < listnet_loss(-scores, rel)
    ranknet_loss(scores, rel).backward()
    assert scores.grad is not None


def test_neural_ranker_learns_order():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    rel = (X[:, 0] > 0.5).astype(float) + (X[:, 0] > 1.2)
    qids = np.repeat(np.arange(30), 20)
    model = NeuralRanker(loss="listnet", epochs=20).fit(X, rel, qids)
    frame = pd.DataFrame({"query_id": qids, "rel": rel, "score": model.predict(X)})
    assert evaluate(frame)["ndcg@10"] > 0.9


def test_trigram_hashing_is_deterministic():
    assert trigram_ids("chair", 1000) == trigram_ids("chair", 1000)
    assert len(trigram_ids("chair", 1000)) == len("#chair#") - 2


@pytest.fixture
def catalog():
    queries = pd.DataFrame({"query_id": [0, 1], "query": ["wooden chair", "blue lamp"], "query_class": ["Chairs", "Lamps"]})
    products = pd.DataFrame({
        "product_id": [10, 11, 12],
        "product_name": ["oak wooden chair", "modern blue desk lamp", "glass table"],
        "product_class": ["Chairs", "Lamps", ""],
        "category": ["Furniture / Chairs", "Lighting / Lamps", "Furniture / Tables"],
        "product_description": ["solid oak", "", "tempered glass top"],
        "product_features": ["material:oak|color:brown", "color:blue", "material:glass"],
        "features_text": ["material oak color brown", "color blue", "material glass"],
        "rating_count": [5.0, np.nan, 2.0], "average_rating": [4.5, np.nan, 3.0], "review_count": [3.0, np.nan, 1.0],
    })
    labels = pd.DataFrame({"query_id": [0, 0, 0, 1, 1, 1], "product_id": [10, 11, 12, 10, 11, 12], "rel": [2, 0, 0, 0, 2, 0]})
    return queries, products, labels


def test_feature_builder(catalog, tmp_path):
    queries, products, labels = catalog
    frame = build(queries, products, labels, cache_dir=tmp_path, use_bert=False)
    assert set(ALL_FEATURES) <= set(frame.columns) and len(frame) == len(labels)
    exact = frame[frame.rel == 2]
    other = frame[frame.rel == 0]
    assert exact["bm25_name"].min() > other["bm25_name"].max()
    assert exact["qclass_eq"].tolist() == [1.0, 1.0]
    assert frame.loc[(frame.query_id == 0) & (frame.product_id == 10), "q_all_in_name"].item() == 1.0


def test_dssm_runs(catalog):
    queries, products, labels = catalog
    q_text = dict(zip(queries.query_id, queries["query"]))
    d_text = dict(zip(products.product_id, products.product_name))
    scores = DSSM(n_buckets=2 ** 10, epochs=3).fit(labels, q_text, d_text).predict(labels, q_text, d_text)
    assert scores.shape == (len(labels),) and np.isfinite(scores).all()
