"""
Сравнение подходов к ранжированию на WANDS. 5-fold кросс-валидация по запросам (GroupKFold):
запросы test-фолда модель не видела. Пишет reports/results.md, reports/metrics.json и графики.

    python scripts/run_experiments.py             # после scripts/build_features.py
    python scripts/run_experiments.py --fast      # без CatBoost, RF и DSSM

CatBoost с PairLogit и YetiRank в сравнение не входят: на CPU при группах до 4 329 товаров
одно обучение на 45 запросах занимало 2,5–3 минуты, полная кросс-валидация — часы.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from lightgbm import LGBMRanker, LGBMRegressor, early_stopping
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltr.data import DATA_DIR, load  # noqa: E402
from ltr.features import ALL_FEATURES, GROUPS  # noqa: E402
from ltr.metrics import per_query  # noqa: E402
from ltr.neural import DSSM, NeuralRanker  # noqa: E402

REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def sort_by_query(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("query_id", kind="stable")


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("query_id", sort=True).size().to_numpy()


def inner_split(train: pd.DataFrame, share: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Часть запросов train — для ранней остановки бустинга."""
    qids = np.sort(train["query_id"].unique())
    rng = np.random.default_rng(SEED)
    valid_q = set(rng.choice(qids, int(len(qids) * share), replace=False))
    mask = train["query_id"].isin(valid_q)
    return sort_by_query(train[~mask]), sort_by_query(train[mask])


# ------------------------------------------------------------------ модели: (train, test, features) → оценки test

def lgbm_ranker(train, test, feats):
    fit, valid = inner_split(train)
    model = LGBMRanker(objective="lambdarank", n_estimators=2000, learning_rate=0.03, num_leaves=31,
                       min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       lambdarank_truncation_level=20, random_state=SEED, verbose=-1, n_jobs=8)
    model.fit(fit[feats], fit["rel"], group=group_sizes(fit), eval_set=[(valid[feats], valid["rel"])],
              eval_group=[group_sizes(valid)], eval_at=[10], callbacks=[early_stopping(100, verbose=False)])
    lgbm_ranker.last_model = model
    return model.predict(test[feats])


def lgbm_regressor(train, test, feats):
    fit, valid = inner_split(train)
    model = LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=8)
    model.fit(fit[feats], fit["rel"], eval_set=[(valid[feats], valid["rel"])], callbacks=[early_stopping(100, verbose=False)])
    return model.predict(test[feats])


def ridge(train, test, feats):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return model.fit(train[feats].fillna(0), train["rel"]).predict(test[feats].fillna(0))


def random_forest(train, test, feats):
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=0.5, max_samples=0.3,
                                  n_jobs=-1, random_state=SEED)
    return model.fit(train[feats].fillna(-1), train["rel"]).predict(test[feats].fillna(-1))


def xgb_regressor(train, test, feats):
    model = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, random_state=SEED, n_jobs=8)
    return model.fit(train[feats], train["rel"]).predict(test[feats])


def xgb_ranker(objective):
    def run(train, test, feats):
        train = sort_by_query(train)
        model = XGBRanker(objective=objective, n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, random_state=SEED, n_jobs=8,
                          lambdarank_pair_method="topk" if objective == "rank:ndcg" else "mean")
        model.fit(train[feats], train["rel"], qid=train["query_id"])
        return model.predict(test[feats])
    return run


def catboost(loss):
    def run(train, test, feats):
        train = sort_by_query(train)
        pool = Pool(train[feats], label=train["rel"], group_id=train["query_id"])
        model = CatBoost({"loss_function": loss, "iterations": 500, "learning_rate": 0.08, "depth": 6,
                          "random_seed": SEED, "verbose": 0, "thread_count": 8, "allow_writing_files": False})
        model.fit(pool)
        return model.predict(Pool(test[feats]))
    return run


def neural(loss):
    def run(train, test, feats):
        model = NeuralRanker(loss=loss, epochs=25, seed=SEED).fit(train[feats].to_numpy(), train["rel"].to_numpy(),
                                                                  train["query_id"].to_numpy())
        return model.predict(test[feats].to_numpy())
    return run


def column(name):
    return lambda train, test, feats: test[name].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    frame = pd.read_parquet(DATA_DIR / "features.parquet")
    queries, products, _ = load()
    q_text = dict(zip(queries["query_id"], queries["query"]))
    d_text = dict(zip(products["product_id"], products["product_name"] + " " + products["product_class"].fillna("")))

    def dssm(train, test, feats):
        return DSSM(n_buckets=2 ** 16, epochs=2, shared=True, seed=SEED).fit(train, q_text, d_text).predict(test, q_text, d_text)

    no_bert = [f for f in ALL_FEATURES if f != "bert_name"]
    models = {
        # name: (подход, функция, признаки)
        "случайный порядок": ("бейзлайн", lambda tr, te, f: np.zeros(len(te)), ALL_FEATURES),
        "BM25 по названию": ("бейзлайн", column("bm25_name"), ALL_FEATURES),
        "BM25 по всем полям": ("бейзлайн", column("bm25_all"), ALL_FEATURES),
        "TF-IDF символьные n-граммы": ("бейзлайн", column("tfidf_char_all"), ALL_FEATURES),
        "BERT-косинус без обучения": ("бейзлайн", column("bert_name"), ALL_FEATURES),
        "линейная регрессия (Ridge)": ("поточечный", ridge, ALL_FEATURES),
        "random forest": ("поточечный", random_forest, ALL_FEATURES),
        "XGBoost регрессия": ("поточечный", xgb_regressor, ALL_FEATURES),
        "LightGBM регрессия": ("поточечный", lgbm_regressor, ALL_FEATURES),
        "CatBoost RMSE": ("поточечный", catboost("RMSE"), ALL_FEATURES),
        "RankNet (MLP)": ("попарный", neural("ranknet"), ALL_FEATURES),
        "XGBoost rank:pairwise": ("попарный", xgb_ranker("rank:pairwise"), ALL_FEATURES),
        "ListNet (MLP)": ("списочный", neural("listnet"), ALL_FEATURES),
        "LightGBM LambdaMART": ("списочный", lgbm_ranker, ALL_FEATURES),
        "XGBoost rank:ndcg": ("списочный", xgb_ranker("rank:ndcg"), ALL_FEATURES),
        "DSSM (общая башня, буквенные триграммы)": ("нейросеть по тексту", dssm, ALL_FEATURES),
        # абляция признаков на LambdaMART
        "LambdaMART без BERT": ("абляция", lgbm_ranker, no_bert),
        "LambdaMART без BERT и LSA": ("абляция", lgbm_ranker, [f for f in no_bert if f != "lsa_all"]),
        "LambdaMART без класса запроса": ("абляция", lgbm_ranker, [f for f in ALL_FEATURES if f not in GROUPS["qclass"]]),
        "LambdaMART только lexical": ("абляция", lgbm_ranker, GROUPS["lexical"]),
    }
    if args.fast:
        models = {k: v for k, v in models.items() if not any(s in k for s in ("CatBoost", "random forest", "DSSM"))}

    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(frame, groups=frame["query_id"]))
    per_q, fold_rows, timing, importance = [], [], {}, []
    for name, (kind, fn, feats) in models.items():
        start = time.time()
        for f, (tr, te) in enumerate(folds):
            train, test = frame.iloc[tr], frame.iloc[te].copy()
            test["score"] = fn(train, test, feats)
            pq = per_query(test).assign(model=name, fold=f)
            per_q.append(pq)
            fold_rows.append({"model": name, "kind": kind, "fold": f, **pq.drop(columns=["query_id", "model", "fold"]).mean().to_dict()})
            if name == "LightGBM LambdaMART":
                booster = lgbm_ranker.last_model.booster_
                importance.append(pd.Series(booster.feature_importance("gain"), index=booster.feature_name()))
        timing[name] = time.time() - start
        last = pd.DataFrame(fold_rows).query("model == @name")
        print(f"{name:45s} NDCG@10 {last['ndcg@10'].mean():.4f} ± {last['ndcg@10'].std():.4f}  {timing[name]:.0f} c", flush=True)

    per_q = pd.concat(per_q)
    folds_df = pd.DataFrame(fold_rows)
    summary = folds_df.groupby(["model", "kind"], sort=False).agg(
        ndcg10=("ndcg@10", "mean"), ndcg10_std=("ndcg@10", "std"), ndcg5=("ndcg@5", "mean"),
        mrr=("mrr", "mean"), p10=("p@10", "mean")).reset_index()

    def paired(a, b):
        x = per_q[per_q.model == a].set_index("query_id")["ndcg@10"]
        y = per_q[per_q.model == b].set_index("query_id")["ndcg@10"]
        both = pd.concat([x, y], axis=1).dropna()
        return float((both.iloc[:, 1] - both.iloc[:, 0]).mean()), float(wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue)

    best = summary[summary.kind != "абляция"].sort_values("ndcg10", ascending=False).iloc[0]["model"]
    best_point = summary[summary.kind == "поточечный"].sort_values("ndcg10", ascending=False).iloc[0]["model"]
    tests = {
        f"{best} vs BM25 по всем полям": paired("BM25 по всем полям", best),
        f"{best} vs {best_point}": paired(best_point, best) if best != best_point else (0.0, 1.0),
    }
    if "LambdaMART без BERT" in models:
        tests["LambdaMART: с BERT vs без BERT"] = paired("LambdaMART без BERT", "LightGBM LambdaMART")

    md = ["# Результаты", "",
          f"WANDS: {frame.query_id.nunique()} запросов, {len(frame):,} размеченных пар, {len(ALL_FEATURES)} признаков. "
          "5-fold кросс-валидация по запросам; метрики усреднены по запросам внутри фолда, затем по фолдам.", "",
          "| подход | модель | NDCG@10 | NDCG@5 | MRR | P@10 (Exact) | время, с |", "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        md.append(f"| {r.kind} | {r.model} | {r.ndcg10:.4f} ± {r.ndcg10_std:.4f} | {r.ndcg5:.4f} | {r.mrr:.4f} | "
                  f"{r.p10:.4f} | {timing[r.model]:.0f} |")
    md += ["", "## Парные сравнения по запросам (ΔNDCG@10, критерий Уилкоксона)", "", "| сравнение | Δ | p-value |", "|---|---|---|"]
    md += [f"| {k} | {d:+.4f} | {p:.2g} |" for k, (d, p) in tests.items()]
    if importance:
        imp = pd.concat(importance, axis=1).mean(1).sort_values(ascending=False)
        imp = imp / imp.sum()
        md += ["", "## Важность признаков в LambdaMART (доля gain, среднее по фолдам)", "", "| признак | доля |", "|---|---|"]
        md += [f"| {k} | {v:.3f} |" for k, v in imp.head(12).items()]
        fig, ax = plt.subplots(figsize=(6, 6))
        imp.head(15)[::-1].plot.barh(ax=ax, color="#4a78b5")
        ax.set_title("LambdaMART: доля gain"); fig.tight_layout(); fig.savefig(FIG / "importance.png", dpi=110)

    main_models = summary[summary.kind != "абляция"].sort_values("ndcg10")
    colors = {"бейзлайн": "#a0a0a0", "поточечный": "#4a78b5", "попарный": "#e08a3c", "списочный": "#3c9e5f",
              "нейросеть по тексту": "#8e5cb5"}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(main_models.model, main_models.ndcg10, xerr=main_models.ndcg10_std, color=[colors[k] for k in main_models.kind])
    ax.set_xlim(main_models.ndcg10.min() - 0.05, main_models.ndcg10.max() + 0.03)
    ax.set_title("NDCG@10, 5-fold CV по запросам")
    for kind, c in colors.items():
        ax.barh([], [], color=c, label=kind)
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "ndcg.png", dpi=110)

    md += ["", f"Время полного прогона: {time.time() - t0:.0f} с."]
    (REPORTS / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps({"summary": summary.to_dict("records"), "tests": tests,
                                                      "timing": timing}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
