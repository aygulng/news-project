#!/usr/bin/env python3
"""Prepare spaCy DocBin datasets for transformer+ner training from local CSV labels.

Input schema (CSV):
sample_id,news_id,published_at,source,title,context,entity,entity_type,start_char,end_char

Outputs:
- data/spacy/train.spacy
- data/spacy/dev.spacy
- optional data/spacy/holdout.spacy
- data/spacy/spacy_data_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)

ALLOWED_TYPES = {
    "Company",
    "Index",
    "Event",
    "Commodity",
    "Currency",
    "Person",
    "Location",
    "Sector",
    "Time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare spaCy DocBin train/dev sets from NER CSV.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "ner_entity_dataset.csv",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_ner_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "spacy",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_type(label: object) -> str | None:
    value = str(label or "").strip()
    if value in {"Sector / Industry", "SectorIndustry"}:
        value = "Sector"
    return value if value in ALLOWED_TYPES else None


def load_and_group(csv_path: Path, doc_id_col: str) -> tuple[list[dict], dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required = {"context", "entity_type", "start_char", "end_char", doc_id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df["context"] = df["context"].astype(str)
    df["entity_type"] = df["entity_type"].apply(normalize_type)
    df = df[df["entity_type"].notna()].copy()
    df["start_char"] = pd.to_numeric(df["start_char"], errors="coerce").fillna(-1).astype(int)
    df["end_char"] = pd.to_numeric(df["end_char"], errors="coerce").fillna(-1).astype(int)
    df = df[(df["start_char"] >= 0) & (df["end_char"] > df["start_char"])].copy()

    grouped_docs: list[dict] = []
    dropped_overlap = 0
    dropped_oob = 0

    for _, g in df.groupby([doc_id_col, "context"], sort=False):
        doc_id = int(g.iloc[0][doc_id_col])
        text = str(g.iloc[0]["context"])
        length = len(text)
        spans = []
        seen = set()
        for row in g.itertuples(index=False):
            start = int(getattr(row, "start_char"))
            end = int(getattr(row, "end_char"))
            label = str(getattr(row, "entity_type"))
            if start < 0 or end > length or end <= start:
                dropped_oob += 1
                continue
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            spans.append(key)

        spans.sort(key=lambda x: (x[0], x[1]))
        filtered = []
        occupied: list[tuple[int, int]] = []
        for s, e, label in spans:
            intersects = any(max(s, os) < min(e, oe) for os, oe in occupied)
            if intersects:
                dropped_overlap += 1
                continue
            occupied.append((s, e))
            filtered.append((s, e, label))

        if filtered:
            grouped_docs.append({"doc_id": doc_id, "text": text, "entities": filtered})

    info = {
        "rows_input": int(len(df)),
        "documents_total": int(len(grouped_docs)),
        "dropped_overlap": int(dropped_overlap),
        "dropped_out_of_bounds": int(dropped_oob),
        "label_distribution": dict(Counter(df["entity_type"].tolist())),
    }
    return grouped_docs, info


def docs_to_docbin(docs: list[dict], nlp) -> tuple[DocBin, dict]:
    db = DocBin(store_user_data=False)
    stats = {"docs": 0, "ents": 0, "dropped_unaligned": 0}

    for item in docs:
        text = item["text"]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in item["entities"]:
            span = doc.char_span(int(start), int(end), label=str(label), alignment_mode="contract")
            if span is None:
                stats["dropped_unaligned"] += 1
                continue
            ents.append(span)
        if not ents:
            continue
        doc.ents = ents
        db.add(doc)
        stats["docs"] += 1
        stats["ents"] += len(ents)

    return db, stats


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_docs, train_info = load_and_group(args.train_csv, doc_id_col="news_id")
    random.Random(args.seed).shuffle(train_docs)
    cut = int(round(len(train_docs) * (1.0 - float(args.dev_ratio))))
    cut = max(1, min(cut, len(train_docs) - 1))
    train_split = train_docs[:cut]
    dev_split = train_docs[cut:]

    nlp = spacy.blank("ru")

    train_db, train_stats = docs_to_docbin(train_split, nlp)
    dev_db, dev_stats = docs_to_docbin(dev_split, nlp)

    train_path = args.output_dir / "train.spacy"
    dev_path = args.output_dir / "dev.spacy"
    train_db.to_disk(train_path)
    dev_db.to_disk(dev_path)

    summary = {
        "seed": int(args.seed),
        "dev_ratio": float(args.dev_ratio),
        "input": {
            "train_csv": str(args.train_csv),
        },
        "grouping": train_info,
        "splits": {
            "train_docs": int(len(train_split)),
            "dev_docs": int(len(dev_split)),
        },
        "docbin": {
            "train": train_stats,
            "dev": dev_stats,
            "train_path": str(train_path),
            "dev_path": str(dev_path),
        },
    }

    if args.holdout_csv.exists():
        holdout_docs, holdout_info = load_and_group(args.holdout_csv, doc_id_col="news_id")
        holdout_db, holdout_stats = docs_to_docbin(holdout_docs, nlp)
        holdout_path = args.output_dir / "holdout.spacy"
        holdout_db.to_disk(holdout_path)
        summary["input"]["holdout_csv"] = str(args.holdout_csv)
        summary["holdout_grouping"] = holdout_info
        summary["docbin"]["holdout"] = holdout_stats
        summary["docbin"]["holdout_path"] = str(holdout_path)

    summary_path = args.output_dir / "spacy_data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] train.spacy: {train_path}")
    print(f"[ok] dev.spacy:   {dev_path}")
    if "holdout_path" in summary["docbin"]:
        print(f"[ok] holdout.spacy: {summary['docbin']['holdout_path']}")
    print(f"[ok] summary:     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

