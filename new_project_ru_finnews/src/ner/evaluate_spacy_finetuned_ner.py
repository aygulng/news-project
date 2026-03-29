#!/usr/bin/env python3
"""Evaluate fine-tuned spaCy NER model on holdout CSV with project-compatible matching."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, classification_report

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
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned spaCy NER on holdout.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "spacy_transformer_ner" / "model-best",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_ner_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "spacy_transformer_ner",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = text.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text.strip()


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_label(label: str) -> str | None:
    x = str(label or "").strip()
    if x in {"SectorIndustry", "Sector / Industry"}:
        x = "Sector"
    if x in ALLOWED_TYPES:
        return x
    return None


def match_strength(gold_norm: str, pred_norm: str) -> int:
    if not gold_norm or not pred_norm:
        return 0
    if gold_norm == pred_norm:
        return 3
    if len(gold_norm) >= 4 and len(pred_norm) >= 4 and (gold_norm in pred_norm or pred_norm in gold_norm):
        return 2
    gt = set(gold_norm.split())
    pt = set(pred_norm.split())
    if gt and pt and len(gt & pt) / max(1, len(gt | pt)) >= 0.6:
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")
    if not args.eval_csv.exists():
        raise FileNotFoundError(f"Eval CSV not found: {args.eval_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(str(args.model_dir))
    df = pd.read_csv(args.eval_csv).copy()
    required = {"sample_id", "context", "entity", "entity_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.eval_csv}: {sorted(missing)}")

    rows = []
    for row in df.itertuples(index=False):
        context = clean_text(getattr(row, "context", ""))
        gold_entity = str(getattr(row, "entity", "")).strip()
        gold_type = canonical_label(getattr(row, "entity_type", "")) or "None"
        doc = nlp(context)

        scored = []
        gold_norm = normalize_text(gold_entity)
        for ent in doc.ents:
            pred_label = canonical_label(ent.label_)
            if not pred_label:
                continue
            s = match_strength(gold_norm, normalize_text(ent.text))
            if s > 0:
                scored.append((s, ent.text, pred_label))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            _, pred_entity, pred_type = scored[0]
        else:
            pred_entity, pred_type = "", "None"

        rows.append(
            {
                "sample_id": int(getattr(row, "sample_id", 0)),
                "context": context,
                "gold_entity": gold_entity,
                "gold_type": gold_type,
                "pred_entity": pred_entity,
                "pred_type": pred_type,
                "is_correct": int(gold_type == pred_type),
            }
        )

    pred_df = pd.DataFrame(rows)
    metrics = classification_report(
        pred_df["gold_type"].astype(str),
        pred_df["pred_type"].astype(str),
        output_dict=True,
        zero_division=0,
    )
    metrics["accuracy"] = float(accuracy_score(pred_df["gold_type"], pred_df["pred_type"]))

    metrics_path = args.output_dir / "spacy_ft_holdout_metrics.json"
    pred_path = args.output_dir / "spacy_ft_holdout_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print("[ok] spaCy fine-tuned evaluation completed.")
    print(f"[ok] metrics: {metrics_path}")
    print(f"[ok] preds:   {pred_path}")
    print(f"[result] Holdout macro F1: {metrics.get('macro avg', {}).get('f1-score', 0.0):.4f}")
    print(f"[result] Holdout accuracy: {metrics.get('accuracy', 0.0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

