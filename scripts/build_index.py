#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restmex_recommender.build import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye los índices locales de RestMex 2025")
    parser.add_argument("--data", type=Path, default=ROOT / "data/Rest-Mex_2025_train_clean.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--semantic", action="store_true", help="Agrega embeddings multilingual-e5")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual-e5-base")
    args = parser.parse_args()
    manifest = build_index(
        args.data,
        args.output,
        with_semantic=args.semantic,
        model_name=args.model,
        model_dir=args.model_dir if args.semantic else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

