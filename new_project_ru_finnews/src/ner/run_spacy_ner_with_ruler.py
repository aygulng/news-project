#!/usr/bin/env python3
"""Run spaCy pretrained NER + EntityRuler dictionary for finance entities.

Outputs:
1) Extracted entities CSV for input corpus.
2) Optional evaluation vs labeled set (if provided).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)


@dataclass(frozen=True)
class Entity:
    text: str
    entity_type: str
    start_char: int
    end_char: int
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spaCy pretrained NER + EntityRuler for rare companies/indexes.")
    parser.add_argument(
        "--input-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "russian_financial_news_clean_1500.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--spacy-model", default="ru_core_news_sm")
    parser.add_argument(
        "--ruler-patterns",
        type=Path,
        default=PROJECT_ROOT / "config" / "rare_company_index_patterns.json",
    )
    parser.add_argument(
        "--eval-labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_ner_dataset.csv",
        help="Optional labeled NER file with columns context/entity/entity_type.",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="Limit number of documents for quick debug.")
    return parser.parse_args()


def clean_text(text: object) -> str:
    value = str(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = value.replace("\u00A0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_text(value: str) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = text.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text.strip()


def load_nlp(model_name: str, patterns_path: Path):
    import spacy

    nlp = spacy.load(model_name)
    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

    patterns = json.loads(patterns_path.read_text(encoding="utf-8"))
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"},
    )
    ruler.add_patterns(patterns)
    return nlp


def map_spacy_label(label: str) -> str | None:
    custom = {"Company", "Index", "Event", "Commodity", "Currency", "Person", "Location", "Sector", "Time"}
    if label in custom:
        return label
    mapping = {
        "ORG": "Company",
        "PERSON": "Person",
        "GPE": "Location",
        "LOC": "Location",
        "EVENT": "Event",
        "MONEY": "Currency",
        "DATE": "Time",
        "TIME": "Time",
        "Sector / Industry": "Sector",
        "SectorIndustry": "Sector",
    }
    return mapping.get(label)


def dedupe_entities(rows: list[Entity]) -> list[Entity]:
    seen: set[tuple[str, str, int, int]] = set()
    out: list[Entity] = []
    for ent in sorted(rows, key=lambda x: (x.start_char, x.end_char, x.entity_type, x.text.lower())):
        key = (normalize_text(ent.text), ent.entity_type, ent.start_char, ent.end_char)
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def extract_entities(nlp, text: str) -> list[Entity]:
    doc = nlp(text)
    rows: list[Entity] = []
    for ent in doc.ents:
        mapped = map_spacy_label(ent.label_)
        if not mapped:
            continue
        method = "ruler" if ent.ent_id_.startswith("ruler:") else "spacy"
        rows.append(
            Entity(
                text=ent.text.strip(),
                entity_type=mapped,
                start_char=int(ent.start_char),
                end_char=int(ent.end_char),
                method=method,
            )
        )
    return dedupe_entities(rows)


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


def evaluate(eval_df: pd.DataFrame, nlp) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for row in eval_df.itertuples(index=False):
        context = clean_text(getattr(row, "context", ""))
        gold_entity = str(getattr(row, "entity", "")).strip()
        gold_type = str(getattr(row, "entity_type", "")).strip()
        preds = extract_entities(nlp, context)

        scored = []
        gold_norm = normalize_text(gold_entity)
        for ent in preds:
            strength = match_strength(gold_norm, normalize_text(ent.text))
            if strength > 0:
                scored.append((strength, ent))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            pred_type = best.entity_type
            pred_entity = best.text
            pred_method = best.method
        else:
            pred_type = "None"
            pred_entity = ""
            pred_method = ""

        rows.append(
            {
                "sample_id": int(getattr(row, "sample_id", 0)),
                "context": context,
                "gold_entity": gold_entity,
                "gold_type": gold_type,
                "pred_entity": pred_entity,
                "pred_type": pred_type,
                "pred_method": pred_method,
                "is_correct": int(pred_type == gold_type),
            }
        )

    pred_df = pd.DataFrame(rows)
    report = classification_report(
        pred_df["gold_type"].astype(str),
        pred_df["pred_type"].astype(str),
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = float(accuracy_score(pred_df["gold_type"], pred_df["pred_type"]))
    return report, pred_df


def metric(metrics: dict, section: str, key: str) -> float:
    block = metrics.get(section, {})
    try:
        return float(block.get(key, 0.0))
    except Exception:
        return 0.0


def main() -> int:
    args = parse_args()
    if not args.input_corpus.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input_corpus}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nlp = load_nlp(args.spacy_model, args.ruler_patterns)
    corpus = pd.read_csv(args.input_corpus).copy()
    if args.max_docs > 0:
        corpus = corpus.head(args.max_docs).copy()

    id_col = "news_id" if "news_id" in corpus.columns else "test_news_id"
    if id_col not in corpus.columns:
        corpus[id_col] = range(1, len(corpus) + 1)
    if "text" not in corpus.columns:
        corpus["text"] = (corpus.get("title", "").fillna("") + ". " + corpus.get("body", "").fillna("")).astype(str)

    extracted_rows: list[dict] = []
    for row in corpus.itertuples(index=False):
        text = clean_text(getattr(row, "text", ""))
        ents = extract_entities(nlp, text)
        for ent in ents:
            extracted_rows.append(
                {
                    "doc_id": int(getattr(row, id_col)),
                    "source": str(getattr(row, "source", "")),
                    "title": str(getattr(row, "title", "")),
                    "entity": ent.text,
                    "entity_type": ent.entity_type,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "method": ent.method,
                }
            )

    extracted_df = pd.DataFrame(extracted_rows)
    extracted_path = args.output_dir / "spacy_ruler_ner_predictions.csv"
    extracted_df.to_csv(extracted_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_docs": int(len(corpus)),
        "predicted_rows": int(len(extracted_df)),
        "method_dist": extracted_df["method"].value_counts().to_dict() if not extracted_df.empty else {},
        "entity_type_dist": extracted_df["entity_type"].value_counts().to_dict() if not extracted_df.empty else {},
        "spacy_model": args.spacy_model,
        "ruler_patterns": str(args.ruler_patterns),
        "evaluation": {},
    }

    eval_pred_path = args.output_dir / "spacy_ruler_eval_predictions.csv"
    eval_metrics_path = args.output_dir / "spacy_ruler_eval_metrics.json"
    summary_path = args.output_dir / "spacy_ruler_summary.json"

    if args.eval_labels.exists():
        eval_df = pd.read_csv(args.eval_labels).copy()
        required = {"context", "entity", "entity_type"}
        missing = [c for c in required if c not in eval_df.columns]
        if missing:
            raise KeyError(f"Eval file missing columns: {missing}")
        eval_metrics, eval_pred_df = evaluate(eval_df, nlp)
        eval_pred_df.to_csv(eval_pred_path, index=False, encoding="utf-8-sig")
        eval_metrics_path.write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["evaluation"] = {
            "rows": int(len(eval_df)),
            "accuracy": float(eval_metrics.get("accuracy", 0.0)),
            "precision_macro": metric(eval_metrics, "macro avg", "precision"),
            "recall_macro": metric(eval_metrics, "macro avg", "recall"),
            "f1_macro": metric(eval_metrics, "macro avg", "f1-score"),
            "f1_weighted": metric(eval_metrics, "weighted avg", "f1-score"),
            "metrics_path": str(eval_metrics_path),
            "predictions_path": str(eval_pred_path),
        }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] spaCy + EntityRuler run completed.")
    print(f"[ok] Predictions: {extracted_path}")
    if summary["evaluation"]:
        print(f"[ok] Eval metrics: {eval_metrics_path}")
        print(f"[result] f1_macro: {summary['evaluation']['f1_macro']:.4f}")
        print(f"[result] accuracy: {summary['evaluation']['accuracy']:.4f}")
    print(f"[ok] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
