"""
WANDS — Wayfair ANnotation DataSet (Chen et al., ECIR 2022).

480 поисковых запросов, 42 994 товара мебельного магазина, 233 448 пар «запрос — товар»,
размеченных асессорами: Exact (товар полностью подходит), Partial (частично), Irrelevant.
Метка переводится в оценку релевантности 2 / 1 / 0.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/{}.csv"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GRADE = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def download(data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("query", "product", "label"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL.format(name), path)


def load(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (queries, products, labels). В labels колонка rel ∈ {0, 1, 2}."""
    download(data_dir)
    queries = pd.read_csv(data_dir / "query.csv", sep="\t")
    products = pd.read_csv(data_dir / "product.csv", sep="\t").rename(columns={"category hierarchy": "category"})
    labels = pd.read_csv(data_dir / "label.csv", sep="\t")
    labels["rel"] = labels["label"].map(GRADE).astype(int)
    for col in ("product_name", "product_class", "category", "product_description", "product_features"):
        products[col] = products[col].fillna("").astype(str)
    # product_features — строка «ключ:значение|ключ:значение», превращаем в обычный текст
    products["features_text"] = products["product_features"].str.replace("|", " ", regex=False).str.replace(":", " ", regex=False)
    queries["query_class"] = queries["query_class"].fillna("").astype(str)
    return queries, products, labels.drop(columns=["id", "label"])
