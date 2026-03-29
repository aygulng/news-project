#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create gold holdout labels (manual-guided) for NER + entity sentiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABEL_SOURCE = "gold_manual_v1"
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create gold-manual holdout labels.")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test",
    )
    parser.add_argument(
        "--backup-suffix",
        default="_backup",
        help="Suffix for backup CSV files.",
    )
    parser.add_argument(
        "--rebuild-from-backup",
        action="store_true",
        help="Use backup files as source (recommended for re-runs).",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def score_hits(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


POS_PATTERNS = [
    r"\bрост\w*",
    r"\bподраст\w*",
    r"\bвырос\w*",
    r"\bприбав\w*",
    r"\bв плюс",
    r"\bлидер\w* роста",
    r"\bпозитив\w*",
    r"\bрекорд\w*",
    r"\bувелич\w*",
    r"\bчист\w+ прибыл\w+",
    r"\bукрепля\w* сотруднич",
    r"\bвыплат\w* дивиденд",
    r"\bрекоменд\w*.*дивиденд",
    r"\bпланир\w*.*дивиденд",
    r"\bвозвращени\w*.*дивиденд",
    r"\bпо верхней границе",
]

NEG_PATTERNS = [
    r"\bпаден\w*",
    r"\bпадает\w*",
    r"\bупал\w*",
    r"\bснижа\w*",
    r"\bснизил\w*",
    r"\bпросад\w*",
    r"\bпод давлен\w*",
    r"\bдавит\w*",
    r"\bсанкц\w*",
    r"\bжестк\w* денеж",
    r"\bнесостоявш\w*",
    r"\bрухнул\w*",
    r"\bсократ\w*",
    r"\bупала на \d+",
    r"\bне видит перспектив",
    r"\bбанкрот\w*",
    r"\bразбила надежды",
]

NEUTRAL_GATING_PATTERNS = [
    r"\bопределитс\w*",
    r"\bобсудит\w*",
    r"\bможет\b",
    r"\bвозможност\w*",
    r"\bожидает\w*",
    r"\bожидани\w*",
    r"\bизуча\w*",
    r"\bкалендар\w*",
    r"\bосновные события\b",
    r"\bпрогноз\w* и комментар\w*",
]


NER_TYPE_OVERRIDES = {
    # No dedicated 'Organization' class in target schema.
    "ОПЕК": "Company",
}


# Small set of explicitly reviewed corrections.
SENTIMENT_OVERRIDES = {
    15: "Negative",   # Сбербанк не видит перспектив улучшения
    26: "Negative",   # не приняли решение о выплате дивидендов
    27: "Negative",   # Акции ОВК упали на 5%
    35: "Negative",   # прибыль банковского сектора упала
    77: "Positive",   # нулевой НДПИ на газ (поддержка отрасли)
    79: "Neutral",    # рассмотрит вопрос о дивидендах
    83: "Neutral",    # публикация отчетности (смешанные сигналы)
    124: "Neutral",   # календарь событий сектора (нейтрально)
    141: "Neutral",   # Банк России выпускает монеты
    142: "Neutral",   # условия для выплаты дивидендов
    143: "Neutral",   # условия для выплаты дивидендов (не факт выплаты)
    144: "Neutral",   # условия для выплаты дивидендов
    228: "Negative",  # дивидендная отсечка
}


def sentiment_gold(row: pd.Series) -> str:
    sample_id = int(row["sample_id"])
    if sample_id in SENTIMENT_OVERRIDES:
        return SENTIMENT_OVERRIDES[sample_id]

    text = normalize_text(f"{row.get('title', '')}. {row.get('context', '')}")
    entity = normalize_text(row.get("entity", ""))
    entity_type = str(row.get("entity_type", "")).strip()

    pos = score_hits(text, POS_PATTERNS)
    neg = score_hits(text, NEG_PATTERNS)
    gated = has_any(text, NEUTRAL_GATING_PATTERNS)

    if entity in {"цб", "банк россии"}:
        if has_any(text, [r"\bразбила надежды", r"\bжестк\w* денеж", r"\bподдержал\w* медвед", r"\bдавит\w*"]):
            return "Negative"
        return "Neutral"

    # Entity-specific handling.
    if entity in {"санкции", "санкциями"}:
        return "Negative"

    if entity in {"дивиденды", "дивидендов", "дивиденда", "дивидендами", "дивидендную", "дивидендной", "дивидендных"}:
        if has_any(text, [r"\bвыплат\w*", r"\bрекоменд\w*", r"\bпланир\w*"]):
            return "Positive"
        if has_any(text, [r"\bне прин\w* решени\w* о выплате", r"\bдивидендн\w+ гэп"]):
            return "Negative"
        return "Neutral"

    if entity in {"ipo", "spo", "размещение", "размещению", "размещения"}:
        if has_any(text, [r"\bпо верхней границе"]):
            return "Positive"
        return "Neutral"

    if entity in {"ставка", "ставки", "ставке", "ставку", "ставкой"}:
        if has_any(text, [r"\bдавит\w*", r"\bжестк\w* денеж", r"\bподдержал\w* медвед"]):
            return "Negative"
        return "Neutral"

    # Conservative policy for macro classes.
    if entity_type in {"Location", "Sector", "Sector / Industry"}:
        if neg > pos and neg > 0:
            return "Negative"
        return "Neutral"

    if neg > pos and neg > 0:
        return "Negative"
    if pos > neg and pos > 0 and not gated:
        return "Positive"

    return "Neutral"


def main() -> int:
    args = parse_args()
    test_dir = args.test_dir
    ner_path = test_dir / "test_ner_dataset.csv"
    sent_path = test_dir / "test_entity_sentiment_dataset.csv"
    if not ner_path.exists() or not sent_path.exists():
        raise FileNotFoundError("Expected holdout files not found in data/test.")

    ner_df = pd.read_csv(ner_path).copy()
    sent_df = pd.read_csv(sent_path).copy()
    if set(ner_df["sample_id"]) != set(sent_df["sample_id"]):
        raise ValueError("NER and sentiment sample_id sets do not match.")

    # Save backups once.
    ner_bak = ner_path.with_name(f"{ner_path.stem}{args.backup_suffix}{ner_path.suffix}")
    sent_bak = sent_path.with_name(f"{sent_path.stem}{args.backup_suffix}{sent_path.suffix}")
    if not ner_bak.exists():
        ner_df.to_csv(ner_bak, index=False, encoding="utf-8-sig")
    if not sent_bak.exists():
        sent_df.to_csv(sent_bak, index=False, encoding="utf-8-sig")

    # Optionally rebuild from original backups.
    if args.rebuild_from_backup and ner_bak.exists() and sent_bak.exists():
        ner_df = pd.read_csv(ner_bak).copy()
        sent_df = pd.read_csv(sent_bak).copy()

    ner_orig_type = ner_df["entity_type"].astype(str).copy()
    sent_orig = sent_df["sentiment"].astype(str).copy()

    # NER corrections.
    ner_df["entity_type"] = ner_df.apply(
        lambda r: NER_TYPE_OVERRIDES.get(str(r.get("entity", "")), str(r.get("entity_type", ""))),
        axis=1,
    )
    sent_df = sent_df.merge(
        ner_df[["sample_id", "entity_type"]],
        on="sample_id",
        how="left",
        suffixes=("", "_gold_from_ner"),
    )
    sent_df["entity_type"] = sent_df["entity_type_gold_from_ner"].fillna(sent_df["entity_type"])
    sent_df.drop(columns=["entity_type_gold_from_ner"], inplace=True)

    # Sentiment corrections.
    sent_df["sentiment"] = sent_df.apply(sentiment_gold, axis=1)

    # Mark as gold.
    ner_df["label_source"] = LABEL_SOURCE
    sent_df["label_source"] = LABEL_SOURCE

    ner_df = ner_df.sort_values("sample_id").reset_index(drop=True)
    sent_df = sent_df.sort_values("sample_id").reset_index(drop=True)
    ner_df.to_csv(ner_path, index=False, encoding="utf-8-sig")
    sent_df.to_csv(sent_path, index=False, encoding="utf-8-sig")

    summary = {
        "label_source": LABEL_SOURCE,
        "rows": int(len(ner_df)),
        "ner_type_changed_rows": int((ner_df["entity_type"].astype(str) != ner_orig_type).sum()),
        "sentiment_changed_rows": int((sent_df["sentiment"].astype(str) != sent_orig).sum()),
        "ner_type_distribution": ner_df["entity_type"].value_counts().to_dict(),
        "sentiment_distribution": sent_df["sentiment"].value_counts().to_dict(),
        "files": {
            "ner": str(ner_path),
            "sentiment": str(sent_path),
            "ner_backup": str(ner_bak),
            "sentiment_backup": str(sent_bak),
        },
    }
    summary_path = test_dir / "gold_holdout_annotation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] Gold holdout annotation completed.")
    print(f"[ok] NER file:       {ner_path}")
    print(f"[ok] Sentiment file: {sent_path}")
    print(f"[ok] Summary:        {summary_path}")
    print(f"[result] NER type changed rows: {summary['ner_type_changed_rows']}")
    print(f"[result] Sentiment changed rows: {summary['sentiment_changed_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
