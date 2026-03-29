#!/usr/bin/env python3
"""Entity-level sentiment classification with TF-IDF + Logistic Regression.

Pipeline:
1) Train sentiment classifier on labeled entity sentiment contexts.
2) Load detected NER entities for news.
3) Build per-entity context from original news text.
4) Predict sentiment per entity mention.
5) Save final table with columns: title, entity, type, sentiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)


ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF + LogisticRegression sentiment on detected NER entities.")
    parser.add_argument(
        "--train-sentiment-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "entity_sentiment_dataset.csv",
    )
    parser.add_argument(
        "--ner-predictions-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "spacy_ruler_ner_predictions.csv",
    )
    parser.add_argument(
        "--news-corpus-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "russian_financial_news_clean_1500.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "news_entity_sentiment_tfidf.csv",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "sentiment_tfidf_metrics.json",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--dedupe-entities", action="store_true", help="Keep unique entity per (doc_id, entity, type).")
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_sentiment(value: object) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
        "позитивная": "Positive",
        "нейтральная": "Neutral",
        "негативная": "Negative",
    }
    return mapping.get(raw, str(value or "").strip())


def sentence_for_entity(text: str, start: int, end: int, fallback_entity: str) -> str:
    """Extract sentence-like context around entity offsets."""
    text = clean_text(text)
    if not text:
        return ""

    n = len(text)
    if start < 0 or end <= start or end > n:
        # Offset fallback: try to find entity in text.
        entity = str(fallback_entity or "").strip()
        if entity:
            m = re.search(re.escape(entity), text, flags=re.IGNORECASE)
            if m:
                start, end = m.start(), m.end()
            else:
                return text[:400]
        else:
            return text[:400]

    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    right_candidates = [i for i in (text.find(".", end), text.find("!", end), text.find("?", end)) if i != -1]
    right = min(right_candidates) if right_candidates else -1

    sent_start = 0 if left == -1 else left + 1
    sent_end = n if right == -1 else right + 1
    sentence = text[sent_start:sent_end].strip()
    if not sentence:
        sentence = text[max(0, start - 120) : min(n, end + 120)].strip()
    return sentence[:600]


def make_feature_text(context: str, entity: str, entity_type: str) -> str:
    """Feature string for classifier."""
    context = clean_text(context)
    entity = str(entity or "").strip()
    entity_type = str(entity_type or "").strip()
    return f"{context} [ENTITY] {entity} [TYPE] {entity_type}".strip()


def load_and_train_model(
    train_path: Path,
    test_size: float,
    random_state: int,
    max_features: int,
    ngram_max: int,
) -> tuple[Pipeline, dict]:
    if not train_path.exists():
        raise FileNotFoundError(f"Train sentiment file not found: {train_path}")

    df = pd.read_csv(train_path).copy()
    required = {"context", "entity", "entity_type", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {train_path}: {sorted(missing)}")

    df["context"] = df["context"].astype(str).apply(clean_text)
    df["entity"] = df["entity"].astype(str).str.strip()
    df["entity_type"] = df["entity_type"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)
    df = df[df["context"] != ""].copy()
    df = df[df["entity"] != ""].copy()
    df = df[df["sentiment"].isin(ALLOWED_SENTIMENTS)].copy()
    if df.empty:
        raise ValueError("No valid rows in sentiment train set after filtering.")

    df["feature_text"] = df.apply(
        lambda r: make_feature_text(r["context"], r["entity"], r["entity_type"]),
        axis=1,
    )

    X = df["feature_text"]
    y = df["sentiment"]

    stratify = y if y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, max(1, int(ngram_max))),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)

    report = classification_report(y_valid, valid_pred, output_dict=True, zero_division=0)
    report["accuracy"] = float(accuracy_score(y_valid, valid_pred))
    report["train_rows"] = int(len(X_train))
    report["valid_rows"] = int(len(X_valid))
    report["labels"] = sorted(df["sentiment"].unique().tolist())
    return model, report


def build_entity_sentiment_output(
    model: Pipeline,
    ner_path: Path,
    news_path: Path,
    dedupe_entities: bool,
) -> pd.DataFrame:
    if not ner_path.exists():
        raise FileNotFoundError(f"NER predictions file not found: {ner_path}")
    if not news_path.exists():
        raise FileNotFoundError(f"News corpus file not found: {news_path}")

    ner_df = pd.read_csv(ner_path).copy()
    news_df = pd.read_csv(news_path).copy()

    required_ner = {"doc_id", "title", "entity", "entity_type", "start_char", "end_char"}
    missing_ner = required_ner - set(ner_df.columns)
    if missing_ner:
        raise ValueError(f"Missing columns in {ner_path}: {sorted(missing_ner)}")

    if "news_id" not in news_df.columns:
        raise ValueError(f"Missing 'news_id' in {news_path}")
    if "text" not in news_df.columns:
        news_df["text"] = (news_df.get("title", "").fillna("") + ". " + news_df.get("body", "").fillna("")).astype(str)

    ner_df["doc_id"] = pd.to_numeric(ner_df["doc_id"], errors="coerce").fillna(-1).astype(int)
    ner_df["start_char"] = pd.to_numeric(ner_df["start_char"], errors="coerce").fillna(-1).astype(int)
    ner_df["end_char"] = pd.to_numeric(ner_df["end_char"], errors="coerce").fillna(-1).astype(int)

    merged = ner_df.merge(
        news_df[["news_id", "text"]],
        left_on="doc_id",
        right_on="news_id",
        how="left",
    )
    merged["text"] = merged["text"].astype(str).apply(clean_text)
    merged["entity"] = merged["entity"].astype(str).str.strip()
    merged["entity_type"] = merged["entity_type"].astype(str).str.strip()
    merged["title"] = merged["title"].astype(str).str.strip()
    merged = merged[(merged["entity"] != "") & (merged["entity_type"] != "")].copy()

    if dedupe_entities:
        merged = merged.sort_values(["doc_id", "entity", "entity_type", "start_char"]).drop_duplicates(
            subset=["doc_id", "entity", "entity_type"], keep="first"
        )

    merged["context"] = merged.apply(
        lambda r: sentence_for_entity(
            text=r["text"],
            start=int(r["start_char"]),
            end=int(r["end_char"]),
            fallback_entity=r["entity"],
        ),
        axis=1,
    )
    merged["feature_text"] = merged.apply(
        lambda r: make_feature_text(r["context"], r["entity"], r["entity_type"]),
        axis=1,
    )

    merged["sentiment"] = model.predict(merged["feature_text"])
    merged["type"] = merged["entity_type"]
    out = merged[["title", "entity", "type", "sentiment"]].copy()
    out["sentiment"] = out["sentiment"].apply(normalize_sentiment)
    return out


def main() -> int:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    model, metrics = load_and_train_model(
        train_path=args.train_sentiment_csv,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
    )
    out_df = build_entity_sentiment_output(
        model=model,
        ner_path=args.ner_predictions_csv,
        news_path=args.news_corpus_csv,
        dedupe_entities=bool(args.dedupe_entities),
    )

    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] Entity sentiment classification completed.")
    print(f"[ok] Output CSV:    {args.output_csv}")
    print(f"[ok] Metrics JSON:  {args.metrics_json}")
    print(f"[ok] Rows:          {len(out_df)}")
    print(f"[ok] Valid accuracy:{float(metrics.get('accuracy', 0.0)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
