"""
Скачивает WANDS и считает признаки всех 233 тыс. пар → data/features.parquet.

    python scripts/build_features.py            # с BERT (скачает all-MiniLM-L6-v2, ~90 МБ)
    python scripts/build_features.py --no-bert
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ltr import features  # noqa: E402
from ltr.data import DATA_DIR, load  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert", action="store_true")
    args = parser.parse_args()
    start = time.time()
    queries, products, labels = load()
    frame = features.build(queries, products, labels, cache_dir=DATA_DIR / "cache", use_bert=not args.no_bert)
    frame.to_parquet(DATA_DIR / "features.parquet")
    print(f"{len(frame)} пар, {len(features.ALL_FEATURES)} признаков, {time.time() - start:.0f} с")
    print(frame[features.ALL_FEATURES].describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
