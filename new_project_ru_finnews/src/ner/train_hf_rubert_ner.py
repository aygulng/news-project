#!/usr/bin/env python3
"""Fine-tune HF ruBERT for Russian financial NER and evaluate on holdout.

Input train format (CSV):
  context, entity, entity_type, start_char, end_char

Input test format (CSV):
  context, entity, entity_type, start_char, end_char

Output:
  - hf_train_metrics.json
  - hf_holdout_metrics.json
  - hf_holdout_predictions.csv
  - hf_training_summary.json
  - hf_model/ (fine-tuned checkpoint)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Helps avoid known HF XET/CAS range-download issues in some environments.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)


ENTITY_TYPES = [
    "Company",
    "Index",
    "Event",
    "Commodity",
    "Currency",
    "Person",
    "Location",
    "Sector",
    "Time",
]

ENTITY_TO_TAG = {
    "Company": "Company",
    "Index": "Index",
    "Event": "Event",
    "Commodity": "Commodity",
    "Currency": "Currency",
    "Person": "Person",
    "Location": "Location",
    "Sector": "Sector",
    "Time": "Time",
}

TAG_TO_ENTITY = {v: k for k, v in ENTITY_TO_TAG.items()}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label_tag: str
    text: str


@dataclass(frozen=True)
class PredictedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    score: float


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
    """Require torch>=2.6 for secure torch.load with modern transformers."""
    if _torch_version_tuple(torch_version) < (2, 6, 0):
        raise RuntimeError(
            "Detected torch<2.6.0. "
            "With current transformers, loading .bin checkpoints (e.g. ruBERT-base) is blocked for security. "
            "Upgrade torch to >=2.6.0 first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HF ruBERT NER on local Russian financial labels.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "ner_entity_dataset.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_ner_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "hf_rubert_base_ner",
    )
    parser.add_argument(
        "--hf-model",
        default="DeepPavlov/rubert-base-cased",
        help="HF checkpoint (ruBERT family). Example: DeepPavlov/rubert-base-cased.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-docs", type=int, default=0, help="Debug limit on train docs.")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--balance-train", action="store_true", help="Oversample minority label docs.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force fresh HF model/tokenizer download.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value: object) -> str:
    text = str(value or "")
    text = text.casefold().replace("ё", "е")
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


def find_span(text: str, entity: str, start: int, end: int) -> tuple[int, int] | None:
    text = str(text or "")
    entity = str(entity or "").strip()
    if not text or not entity:
        return None

    n = len(text)
    start = max(0, min(int(start), n))
    end = max(start, min(int(end), n))

    # 1) Trust offsets if they align with entity text.
    piece = text[start:end]
    if piece and normalize_text(piece) == normalize_text(entity):
        return start, end

    # 2) Find closest case-insensitive entity match.
    matches = list(re.finditer(re.escape(entity), text, flags=re.IGNORECASE))
    if not matches:
        return None
    best = min(matches, key=lambda m: abs(m.start() - start))
    return best.start(), best.end()


def remove_overlaps(spans: list[Span]) -> tuple[list[Span], int]:
    """Keep non-overlapping spans; prefer longer span then earlier start."""
    dropped = 0
    accepted: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start), s.label_tag)):
        overlap = False
        for cur in accepted:
            if span.start < cur.end and cur.start < span.end:
                overlap = True
                break
        if overlap:
            dropped += 1
            continue
        accepted.append(span)
    accepted.sort(key=lambda s: (s.start, s.end, s.label_tag))
    return accepted, dropped


def load_ner_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"NER csv not found: {path}")
    df = pd.read_csv(path).copy()
    required = {"context", "entity", "entity_type", "start_char", "end_char"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["context"] = df["context"].astype(str).apply(clean_text)
    df["entity"] = df["entity"].astype(str).str.strip()
    df["entity_type"] = df["entity_type"].astype(str).str.strip()
    # Backward compatibility for old schema names.
    df["entity_type"] = df["entity_type"].replace(
        {
            "Sector / Industry": "Sector",
            "SectorIndustry": "Sector",
        }
    )
    df["start_char"] = pd.to_numeric(df["start_char"], errors="coerce").fillna(-1).astype(int)
    df["end_char"] = pd.to_numeric(df["end_char"], errors="coerce").fillna(-1).astype(int)

    df = df[df["context"] != ""].copy()
    df = df[df["entity"] != ""].copy()
    df = df[df["entity_type"].isin(ENTITY_TYPES)].copy()
    if df.empty:
        raise ValueError(f"No valid rows after filtering: {path}")
    return df


def build_documents(df: pd.DataFrame) -> tuple[list[dict], dict]:
    docs: list[dict] = []
    dropped_unaligned = 0
    dropped_overlap = 0

    group_cols = ["context"]
    if "news_id" in df.columns:
        group_cols.append("news_id")
    elif "test_news_id" in df.columns:
        group_cols.append("test_news_id")

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for _, grp in grouped:
        context = str(grp.iloc[0]["context"])
        spans: list[Span] = []
        for row in grp.itertuples(index=False):
            entity = str(getattr(row, "entity"))
            entity_type = str(getattr(row, "entity_type"))
            resolved = find_span(context, entity, int(getattr(row, "start_char")), int(getattr(row, "end_char")))
            if not resolved:
                dropped_unaligned += 1
                continue
            s, e = resolved
            spans.append(
                Span(
                    start=s,
                    end=e,
                    label_tag=ENTITY_TO_TAG[entity_type],
                    text=context[s:e],
                )
            )

        if not spans:
            continue
        spans, dropped = remove_overlaps(spans)
        dropped_overlap += dropped
        if not spans:
            continue

        primary = TAG_TO_ENTITY[spans[0].label_tag]
        doc_id = int(len(docs) + 1)
        docs.append(
            {
                "doc_id": doc_id,
                "text": context,
                "spans": spans,
                "primary_label": primary,
            }
        )

    stats = {
        "documents_total": int(len(docs)),
        "rows_input": int(len(df)),
        "dropped_unaligned": int(dropped_unaligned),
        "dropped_overlap": int(dropped_overlap),
    }
    return docs, stats


def split_documents(docs: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if len(docs) < 2:
        raise ValueError("Need at least 2 docs for train/valid split.")
    valid_size = max(1, int(round(len(docs) * valid_ratio)))
    if valid_size >= len(docs):
        valid_size = len(docs) - 1

    labels = [str(d["primary_label"]) for d in docs]
    counts = pd.Series(labels).value_counts()
    stratify = labels if len(counts) > 1 and int(counts.min()) >= 2 else None

    try:
        train_docs, valid_docs = train_test_split(
            docs,
            test_size=valid_size,
            random_state=seed,
            stratify=stratify,
        )
    except Exception:
        rng = random.Random(seed)
        shuffled = list(docs)
        rng.shuffle(shuffled)
        valid_docs = shuffled[:valid_size]
        train_docs = shuffled[valid_size:]

    return list(train_docs), list(valid_docs)


def balance_docs(train_docs: list[dict], seed: int) -> tuple[list[dict], dict]:
    if not train_docs:
        return train_docs, {"enabled": True, "input_rows": 0, "output_rows": 0}

    by_label: dict[str, list[dict]] = {}
    for doc in train_docs:
        by_label.setdefault(str(doc["primary_label"]), []).append(doc)

    target = max(len(v) for v in by_label.values())
    rng = random.Random(seed)
    out: list[dict] = []
    for label, items in by_label.items():
        if len(items) >= target:
            out.extend(items)
            continue
        need = target - len(items)
        out.extend(items)
        out.extend(rng.choice(items) for _ in range(need))
    rng.shuffle(out)
    stats = {
        "enabled": True,
        "input_rows": int(len(train_docs)),
        "output_rows": int(len(out)),
        "target_per_label": int(target),
        "input_by_label": {k: int(len(v)) for k, v in by_label.items()},
    }
    return out, stats


def build_bio_maps() -> tuple[dict[int, str], dict[str, int]]:
    id2label = {0: "O"}
    idx = 1
    for entity_type in ENTITY_TYPES:
        tag = ENTITY_TO_TAG[entity_type]
        id2label[idx] = f"B-{tag}"
        idx += 1
        id2label[idx] = f"I-{tag}"
        idx += 1
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def align_offsets_to_labels(
    offsets: list[tuple[int, int]],
    spans: list[Span],
    label2id: dict[str, int],
) -> list[int]:
    labels: list[int] = []
    for token_start, token_end in offsets:
        if token_start == token_end:
            labels.append(-100)
            continue

        matched: Span | None = None
        best_overlap = 0
        for span in spans:
            if span.end <= token_start:
                continue
            if span.start >= token_end:
                break
            overlap = min(token_end, span.end) - max(token_start, span.start)
            if overlap > best_overlap:
                best_overlap = overlap
                matched = span

        if matched is None or best_overlap <= 0:
            labels.append(label2id["O"])
            continue

        if token_start <= matched.start < token_end:
            tag = f"B-{matched.label_tag}"
        else:
            tag = f"I-{matched.label_tag}"
        labels.append(label2id.get(tag, label2id["O"]))
    return labels


def build_tokenized_dataset(
    docs: list[dict],
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
    stride: int,
):
    from datasets import Dataset

    texts = [d["text"] for d in docs]
    spans_per_doc = [d["spans"] for d in docs]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    sample_map = tokenized["overflow_to_sample_mapping"]

    all_labels: list[list[int]] = []
    chunk_doc_ids: list[int] = []
    for chunk_idx, offsets in enumerate(tokenized["offset_mapping"]):
        doc_idx = sample_map[chunk_idx]
        spans = spans_per_doc[doc_idx]
        labels = align_offsets_to_labels(
            offsets=[(int(s), int(e)) for s, e in offsets],
            spans=spans,
            label2id=label2id,
        )
        all_labels.append(labels)
        chunk_doc_ids.append(int(docs[doc_idx]["doc_id"]))

    feats = {k: v for k, v in tokenized.items() if k not in {"offset_mapping", "overflow_to_sample_mapping"}}
    feats["labels"] = all_labels
    feats["doc_id"] = chunk_doc_ids
    return Dataset.from_dict(feats)


def compute_seqeval_metrics_factory(id2label: dict[int, str]):
    from seqeval.metrics import accuracy_score as seqeval_accuracy
    from seqeval.metrics import f1_score as seqeval_f1
    from seqeval.metrics import precision_score as seqeval_precision
    from seqeval.metrics import recall_score as seqeval_recall

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=2)
        true_preds: list[list[str]] = []
        true_labels: list[list[str]] = []
        for pred_row, gold_row in zip(preds, labels):
            p: list[str] = []
            g: list[str] = []
            for pred_id, gold_id in zip(pred_row, gold_row):
                if int(gold_id) == -100:
                    continue
                p.append(id2label[int(pred_id)])
                g.append(id2label[int(gold_id)])
            true_preds.append(p)
            true_labels.append(g)
        return {
            "precision": float(seqeval_precision(true_labels, true_preds)),
            "recall": float(seqeval_recall(true_labels, true_preds)),
            "f1": float(seqeval_f1(true_labels, true_preds)),
            "accuracy": float(seqeval_accuracy(true_labels, true_preds)),
        }

    return compute_metrics


def _training_args_kwargs(
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
        "metric_for_best_model": "f1",
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


def train_hf_model(
    train_docs: list[dict],
    valid_docs: list[dict],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, dict]:
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "HF dependencies missing. Install: transformers torch datasets seqeval accelerate"
        ) from exc

    ensure_torch_safe_version(torch.__version__)

    id2label, label2id = build_bio_maps()

    tokenizer_kwargs = {"use_fast": True}
    model_kwargs = {}
    if args.force_download:
        tokenizer_kwargs["force_download"] = True
        model_kwargs["force_download"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, **tokenizer_kwargs)
    train_ds = build_tokenized_dataset(
        docs=train_docs,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        stride=args.stride,
    )
    valid_ds = build_tokenized_dataset(
        docs=valid_docs,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        stride=args.stride,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.hf_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )

    use_cuda = bool(torch.cuda.is_available() and not args.disable_cuda)
    model_dir = output_dir / "hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        **_training_args_kwargs(
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
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_seqeval_metrics_factory(id2label),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    save_warning = ""
    try:
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
    except Exception as exc:
        save_warning = str(exc)
        checkpoints = []
        for p in model_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            try:
                step = int(p.name.split("-")[-1])
            except Exception:
                step = -1
            checkpoints.append((step, p))
        checkpoints.sort(key=lambda x: x[0])

        if checkpoints:
            model_dir = checkpoints[-1][1]
            print(
                "[warn] Final model save failed; using latest checkpoint instead: "
                f"{model_dir}\n[warn] Save error: {save_warning}"
            )
        else:
            fallback_dir = output_dir / "hf_model_fallback_bin"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(str(fallback_dir), safe_serialization=False)
            tokenizer.save_pretrained(str(fallback_dir))
            model_dir = fallback_dir
            print(
                "[warn] Final model save failed; fallback .bin model saved to: "
                f"{model_dir}\n[warn] Save error: {save_warning}"
            )

    summary = {
        "hf_model": args.hf_model,
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_chunks": int(len(train_ds)),
        "valid_chunks": int(len(valid_ds)),
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_result.metrics.get("train_samples_per_second", 0.0)),
        "eval_precision": float(eval_metrics.get("eval_precision", 0.0)),
        "eval_recall": float(eval_metrics.get("eval_recall", 0.0)),
        "eval_f1": float(eval_metrics.get("eval_f1", 0.0)),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
        "device": "cuda" if use_cuda else "cpu",
    }
    if save_warning:
        summary["save_warning"] = save_warning
    return model_dir, summary


def match_strength(gold_norm: str, pred_norm: str) -> int:
    if not gold_norm or not pred_norm:
        return 0
    if gold_norm == pred_norm:
        return 3
    if len(gold_norm) >= 4 and len(pred_norm) >= 4 and (gold_norm in pred_norm or pred_norm in gold_norm):
        return 2
    g = set(gold_norm.split())
    p = set(pred_norm.split())
    if g and p and len(g & p) / max(1, len(g | p)) >= 0.6:
        return 1
    return 0


def create_hf_predictor(
    model_dir: Path,
    max_length: int,
    stride: int,
    disable_cuda: bool,
):
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except Exception as exc:
        raise RuntimeError("HF inference dependencies are missing.") from exc

    ensure_torch_safe_version(torch.__version__)

    device = -1
    if torch.cuda.is_available() and not disable_cuda:
        device = 0

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)

    ner_pipe = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    def normalize_label(raw: str) -> str:
        val = str(raw or "").strip()
        if val.startswith("B-") or val.startswith("I-"):
            val = val[2:]
        return TAG_TO_ENTITY.get(val, "")

    def predict(text: str) -> list[PredictedEntity]:
        text = clean_text(text)
        if not text:
            return []
        try:
            raw = ner_pipe(text, truncation=True, max_length=max_length, stride=stride)
        except TypeError:
            raw = ner_pipe(text)

        entities: list[PredictedEntity] = []
        for item in raw:
            label = normalize_label(item.get("entity_group", item.get("entity", "")))
            if label not in ENTITY_TYPES:
                continue
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            if 0 <= start < end <= len(text):
                value = text[start:end]
            else:
                value = str(item.get("word", "")).strip()
            entities.append(
                PredictedEntity(
                    text=value.strip(),
                    entity_type=label,
                    start=start,
                    end=end,
                    score=float(item.get("score", 0.0)),
                )
            )
        return entities

    return predict


def evaluate_holdout(test_df: pd.DataFrame, predictor) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for row in test_df.itertuples(index=False):
        context = clean_text(getattr(row, "context", ""))
        gold_entity = str(getattr(row, "entity", "")).strip()
        gold_type = str(getattr(row, "entity_type", "")).strip()
        preds = predictor(context)
        gold_norm = normalize_text(gold_entity)

        scored: list[tuple[int, float, PredictedEntity]] = []
        for ent in preds:
            strength = match_strength(gold_norm, normalize_text(ent.text))
            if strength > 0:
                scored.append((strength, ent.score, ent))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        if scored:
            best = scored[0][2]
            pred_entity = best.text
            pred_type = best.entity_type
            pred_score = float(best.score)
        else:
            pred_entity = ""
            pred_type = "None"
            pred_score = 0.0

        rows.append(
            {
                "sample_id": int(getattr(row, "sample_id", 0)) if "sample_id" in test_df.columns else 0,
                "context": context,
                "gold_entity": gold_entity,
                "gold_type": gold_type,
                "pred_entity": pred_entity,
                "pred_type": pred_type,
                "pred_score": pred_score,
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


def compact_metrics(metrics: dict) -> dict:
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision_macro": metric(metrics, "macro avg", "precision"),
        "recall_macro": metric(metrics, "macro avg", "recall"),
        "f1_macro": metric(metrics, "macro avg", "f1-score"),
        "f1_weighted": metric(metrics, "weighted avg", "f1-score"),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_ner_csv(args.train_csv)
    test_df = load_ner_csv(args.test_csv)

    all_docs, build_stats = build_documents(train_df)
    if args.max_train_docs > 0:
        all_docs = all_docs[: args.max_train_docs]
    if len(all_docs) < 2:
        raise ValueError("Not enough train docs after preprocessing.")

    train_docs, valid_docs = split_documents(all_docs, valid_ratio=args.valid_ratio, seed=args.seed)
    balance_stats = {"enabled": False}
    if args.balance_train:
        train_docs, balance_stats = balance_docs(train_docs, seed=args.seed)

    model_dir, train_metrics = train_hf_model(
        train_docs=train_docs,
        valid_docs=valid_docs,
        args=args,
        output_dir=args.output_dir,
    )

    predictor = create_hf_predictor(
        model_dir=model_dir,
        max_length=args.max_length,
        stride=args.stride,
        disable_cuda=args.disable_cuda,
    )
    holdout_metrics, holdout_df = evaluate_holdout(test_df=test_df, predictor=predictor)

    train_metrics_path = args.output_dir / "hf_train_metrics.json"
    holdout_metrics_path = args.output_dir / "hf_holdout_metrics.json"
    holdout_pred_path = args.output_dir / "hf_holdout_predictions.csv"
    summary_path = args.output_dir / "hf_training_summary.json"

    save_json(train_metrics_path, train_metrics)
    save_json(holdout_metrics_path, holdout_metrics)
    holdout_df.to_csv(holdout_pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "data": {
            "train_rows": int(len(train_df)),
            "train_docs": int(len(all_docs)),
            "valid_docs": int(len(valid_docs)),
            "test_rows": int(len(test_df)),
            "build_stats": build_stats,
            "balance_stats": balance_stats,
        },
        "runtime": {
            "hf_model": args.hf_model,
            "epochs": float(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_length": int(args.max_length),
            "stride": int(args.stride),
            "seed": int(args.seed),
            "disable_cuda": bool(args.disable_cuda),
        },
        "train_metrics": train_metrics,
        "holdout_metrics": compact_metrics(holdout_metrics),
        "artifacts": {
            "model_dir": str(model_dir),
            "train_metrics": str(train_metrics_path),
            "holdout_metrics": str(holdout_metrics_path),
            "holdout_predictions": str(holdout_pred_path),
        },
    }
    save_json(summary_path, summary)

    print("[ok] HF ruBERT fine-tuning completed.")
    print(f"[ok] Model dir:              {model_dir}")
    print(f"[ok] Train metrics:         {train_metrics_path}")
    print(f"[ok] Holdout metrics:       {holdout_metrics_path}")
    print(f"[ok] Holdout predictions:   {holdout_pred_path}")
    print(f"[ok] Summary:               {summary_path}")
    print(f"[result] Holdout F1 macro: {summary['holdout_metrics']['f1_macro']:.4f}")
    print(f"[result] Holdout accuracy: {summary['holdout_metrics']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
