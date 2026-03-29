#!/usr/bin/env python3
"""Compare entity-sentiment models: TF-IDF+LogReg vs fine-tuned RuBERT-base.

Outputs:
1) tfidf_test_metrics.json
2) rubert_test_metrics.json
3) sentiment_test_predictions_compare.csv
4) sentiment_comparison_summary.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Helps avoid known HF XET/CAS issues in some environments.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)

ALLOWED_SENTIMENTS = ("Negative", "Neutral", "Positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TF-IDF and RuBERT for entity-level sentiment.")
    parser.add_argument(
        "--train-sentiment-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "entity_sentiment_dataset.csv",
    )
    parser.add_argument(
        "--test-sentiment-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_entity_sentiment_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "sentiment_tfidf_rubert",
    )
    parser.add_argument("--hf-model", default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional debug limit for train rows.")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-rubert-train", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def make_feature_text(context: object, entity: object, entity_type: object) -> str:
    return f"{clean_text(context)} [ENTITY] {str(entity or '').strip()} [TYPE] {str(entity_type or '').strip()}".strip()


def load_sentiment_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {path}")

    df = pd.read_csv(path).copy()
    required = {"context", "entity", "entity_type", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    df["context"] = df["context"].astype(str).apply(clean_text)
    df["entity"] = df["entity"].astype(str).str.strip()
    df["entity_type"] = df["entity_type"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)
    df = df[(df["context"] != "") & (df["entity"] != "") & (df["entity_type"] != "")].copy()
    df = df[df["sentiment"].isin(ALLOWED_SENTIMENTS)].copy()
    if df.empty:
        raise ValueError(f"No valid rows in {path} after filtering.")

    df["feature_text"] = df.apply(
        lambda r: make_feature_text(r["context"], r["entity"], r["entity_type"]),
        axis=1,
    )
    return df


def _torch_version_tuple(version: str) -> tuple[int, int, int]:
    core = str(version).split("+", 1)[0]
    parts = core.split(".")
    nums: list[int] = []
    for p in parts[:3]:
        m = re.match(r"^(\d+)", p)
        nums.append(int(m.group(1)) if m else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]


def ensure_torch_safe_version(torch_version: str) -> None:
    if _torch_version_tuple(torch_version) < (2, 6, 0):
        raise RuntimeError(
            "Detected torch<2.6.0. Upgrade torch to >=2.6.0 to load .bin checkpoints safely."
        )


def build_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    return report


def compact(metrics: dict) -> dict:
    macro = metrics.get("macro avg", {})
    weighted = metrics.get("weighted avg", {})
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision_macro": float(macro.get("precision", 0.0)),
        "recall_macro": float(macro.get("recall", 0.0)),
        "f1_macro": float(macro.get("f1-score", 0.0)),
        "f1_weighted": float(weighted.get("f1-score", 0.0)),
    }


def train_eval_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> tuple[Pipeline, dict, pd.Series]:
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    )
    model.fit(train_df["feature_text"], train_df["sentiment"])
    pred = pd.Series(model.predict(test_df["feature_text"]), index=test_df.index)
    metrics = build_metrics(test_df["sentiment"], pred)
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    return model, metrics, pred


def _build_training_args_kwargs(
    training_args_cls,
    output_dir: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    fp16: bool,
    seed: int,
) -> dict:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(epochs),
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 20,
        "report_to": "none",
        "fp16": bool(fp16),
        "seed": int(seed),
        "do_train": True,
        "do_eval": True,
    }

    params = inspect.signature(training_args_cls.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def train_eval_rubert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict, pd.Series, str]:
    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import precision_recall_fscore_support
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError("Install dependencies: torch transformers datasets accelerate.") from exc

    ensure_torch_safe_version(torch.__version__)

    label_list = list(ALLOWED_SENTIMENTS)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    train_ds_df = train_df.copy()
    test_ds_df = test_df.copy()
    train_ds_df["label"] = train_ds_df["sentiment"].map(label2id).astype(int)
    test_ds_df["label"] = test_ds_df["sentiment"].map(label2id).astype(int)

    train_ds = Dataset.from_pandas(train_ds_df[["feature_text", "label"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_ds_df[["feature_text", "label"]], preserve_index=False)

    tok_kwargs: dict = {"use_fast": True}
    model_kwargs: dict = {}
    if args.force_download:
        tok_kwargs["force_download"] = True
        model_kwargs["force_download"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, **tok_kwargs)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["feature_text"], truncation=True, max_length=int(args.max_length))

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["feature_text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["feature_text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )

    use_cuda = bool(torch.cuda.is_available() and not args.disable_cuda)
    model_dir = output_dir / "rubert_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        _, _, f_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": float(acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f_macro),
            "f1_weighted": float(f_weighted),
        }

    train_args = TrainingArguments(
        **_build_training_args_kwargs(
            training_args_cls=TrainingArguments,
            output_dir=model_dir,
            epochs=float(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            fp16=bool(use_cuda),
            seed=int(args.seed),
        )
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    pred_out = trainer.predict(test_ds)

    pred_ids = np.argmax(pred_out.predictions, axis=-1)
    pred_labels = pd.Series([id2label[int(i)] for i in pred_ids], index=test_df.index)

    # Save model locally for reuse.
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    metrics = build_metrics(test_df["sentiment"], pred_labels)
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    metrics["train_runtime_seconds"] = float(train_result.metrics.get("train_runtime", 0.0))
    metrics["train_samples_per_second"] = float(train_result.metrics.get("train_samples_per_second", 0.0))
    metrics["eval_accuracy_runtime"] = float(eval_result.get("eval_accuracy", 0.0))
    metrics["device"] = "cuda" if use_cuda else "cpu"

    return metrics, pred_labels, str(model_dir)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_sentiment_df(args.train_sentiment_csv)
    test_df = load_sentiment_df(args.test_sentiment_csv)
    if args.max_train_rows > 0:
        train_df = train_df.head(int(args.max_train_rows)).copy()

    _, tfidf_metrics, tfidf_pred = train_eval_tfidf(train_df=train_df, test_df=test_df, seed=args.seed)

    rubert_metrics: dict = {}
    rubert_pred = pd.Series(["Neutral"] * len(test_df), index=test_df.index)
    rubert_model_dir = ""
    rubert_failed = False
    rubert_error = ""

    if not args.skip_rubert_train:
        try:
            rubert_metrics, rubert_pred, rubert_model_dir = train_eval_rubert(
                train_df=train_df,
                test_df=test_df,
                args=args,
                output_dir=args.output_dir,
            )
        except Exception as exc:  # pragma: no cover
            rubert_failed = True
            rubert_error = str(exc)
    else:
        rubert_error = "RuBERT training skipped by --skip-rubert-train."

    pred_df = test_df.copy()
    pred_df["gold_sentiment"] = pred_df["sentiment"]
    pred_df["tfidf_pred"] = tfidf_pred
    pred_df["tfidf_correct"] = (pred_df["tfidf_pred"] == pred_df["gold_sentiment"]).astype(int)
    pred_df["rubert_pred"] = rubert_pred
    pred_df["rubert_correct"] = (pred_df["rubert_pred"] == pred_df["gold_sentiment"]).astype(int)
    out_pred = pred_df[
        ["title", "entity", "entity_type", "context", "gold_sentiment", "tfidf_pred", "tfidf_correct", "rubert_pred", "rubert_correct"]
    ].rename(columns={"entity_type": "type"})

    tfidf_metrics_path = args.output_dir / "tfidf_test_metrics.json"
    rubert_metrics_path = args.output_dir / "rubert_test_metrics.json"
    pred_csv_path = args.output_dir / "sentiment_test_predictions_compare.csv"
    summary_path = args.output_dir / "sentiment_comparison_summary.json"

    save_json(tfidf_metrics_path, tfidf_metrics)
    save_json(rubert_metrics_path, rubert_metrics)
    out_pred.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "data": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "labels": list(ALLOWED_SENTIMENTS),
        },
        "runtime": {
            "hf_model": args.hf_model,
            "epochs": float(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_length": int(args.max_length),
            "disable_cuda": bool(args.disable_cuda),
            "rubert_failed": bool(rubert_failed),
            "rubert_error": rubert_error,
        },
        "tfidf": compact(tfidf_metrics),
        "rubert": compact(rubert_metrics) if rubert_metrics else {},
        "comparison": {},
        "artifacts": {
            "tfidf_metrics": str(tfidf_metrics_path),
            "rubert_metrics": str(rubert_metrics_path),
            "predictions_compare": str(pred_csv_path),
            "rubert_model_dir": rubert_model_dir,
            "summary": str(summary_path),
        },
    }
    if rubert_metrics:
        summary["comparison"] = {
            "delta_f1_macro_rubert_minus_tfidf": (
                compact(rubert_metrics)["f1_macro"] - compact(tfidf_metrics)["f1_macro"]
            ),
            "delta_accuracy_rubert_minus_tfidf": (
                compact(rubert_metrics)["accuracy"] - compact(tfidf_metrics)["accuracy"]
            ),
        }
    save_json(summary_path, summary)

    print("[ok] Sentiment model comparison completed.")
    print(f"[ok] Output dir:         {args.output_dir}")
    print(f"[ok] TF-IDF metrics:     {tfidf_metrics_path}")
    print(f"[ok] RuBERT metrics:     {rubert_metrics_path}")
    print(f"[ok] Predictions:        {pred_csv_path}")
    print(f"[ok] Summary:            {summary_path}")
    print(f"[result] TF-IDF f1_macro: {compact(tfidf_metrics)['f1_macro']:.4f}")
    if rubert_metrics:
        print(f"[result] RuBERT f1_macro: {compact(rubert_metrics)['f1_macro']:.4f}")
        print(f"[result] Delta RuBERT-TF-IDF: {summary['comparison']['delta_f1_macro_rubert_minus_tfidf']:.4f}")
    else:
        print("[result] RuBERT f1_macro: n/a")
        print(f"[warn] {rubert_error}")

    return 0 if not rubert_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
