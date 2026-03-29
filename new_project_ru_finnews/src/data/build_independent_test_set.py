#!/usr/bin/env python3
"""Build an independent test set with zero overlap to the current main corpus."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset

DATASET_ID = "Kasymkhan/RussianFinancialNews"
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)
INVALID_TITLES = {"", "no title", "none", "nan", "untitled", "без названия"}
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACES_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

COMPANY_TERMS = [
    "Сбербанк",
    "Сбер",
    "Газпром",
    "Лукойл",
    "Роснефть",
    "Норникель",
    "ВТБ",
    "Т-Банк",
    "Альфа-Банк",
    "Яндекс",
    "Мосбиржа",
    "Московская биржа",
    "Банк России",
    "ЦБ",
    "НОВАТЭК",
    "МТС",
    "Северсталь",
    "ФосАгро",
    "Интер РАО",
]
INDEX_TERMS = [
    "Индекс МосБиржи",
    "Индекс Мосбиржи",
    "IMOEX",
    "RTS",
    "S&P 500",
    "Nasdaq",
    "Dow Jones",
    "Nikkei",
    "Hang Seng",
    "Shanghai Composite",
]
LOCATION_TERMS = ["Россия", "Москва", "США", "Европа", "Китай", "Индия", "ОПЕК", "ЕС", "АТР"]
COMMODITY_TERMS = ["нефть", "Нефть", "Brent", "WTI", "газ", "золото", "серебро", "платина", "палладий", "уголь"]
SECTOR_ROOTS = [
    "сектор",
    "отрасл",
    "индустр",
    "металлург",
    "нефтегаз",
    "энергет",
    "банковск",
    "технологическ",
    "финансов",
    "потребительск",
    "телеком",
    "ритейл",
    "недвижимост",
    "девелопер",
]
EVENT_ROOTS = [
    "дивиденд",
    "отчетност",
    "санкци",
    "эмисси",
    "размещени",
    "слияни",
    "поглощени",
    "дефолт",
    "банкрот",
    "IPO",
    "SPO",
    "ставк",
]
POSITIVE_KEYS = ["рост", "вырос", "выросл", "повыс", "прибыл", "прибыль", "рекорд", "улучш", "одобр", "укреп", "плюс"]
NEGATIVE_KEYS = ["пад", "сниж", "просел", "убыт", "дефолт", "банкрот", "риск", "обвал", "минус", "санкц"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent test set with no overlap to train corpus.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--train-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "russian_financial_news_clean_1500.csv",
    )
    parser.add_argument("--test-news", type=int, default=300)
    parser.add_argument("--test-ner-labels", type=int, default=600)
    parser.add_argument("--test-sent-labels", type=int, default=600)
    parser.add_argument("--max-entities-per-news", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test",
    )
    return parser.parse_args()


def clean_text(text: object) -> str:
    value = str(text or "")
    value = HTML_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    value = value.replace("\u00A0", " ")
    value = SPACES_RE.sub(" ", value)
    return value.strip()


def parse_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip().strip("#") for x in re.split(r"[;,|]", text) if x.strip()]


def parse_datetime(date_value: object, time_value: object) -> pd.Timestamp:
    date_text = str(date_value or "").strip()
    time_text = str(time_value or "").strip()
    if not date_text:
        return pd.NaT
    if not time_text or time_text.lower() in {"nan", "none", "not stated", "unknown"}:
        time_text = "00:00:00"
    elif re.fullmatch(r"\d{2}:\d{2}", time_text):
        time_text = f"{time_text}:00"
    return pd.to_datetime(f"{date_text} {time_text}", errors="coerce")


def is_valid_title(title: object) -> bool:
    return clean_text(title).lower() not in INVALID_TITLES


def pattern_signature(text: str, max_tokens: int = 100) -> str:
    value = clean_text(text).lower()
    value = URL_RE.sub(" ", value)
    value = re.sub(r"#[a-zа-яё0-9_+\-]+", " ", value)
    value = re.sub(r"\d+(?:[.,]\d+)?", " <num> ", value)
    value = re.sub(r"[^a-zа-яё<>\s]", " ", value)
    value = SPACES_RE.sub(" ", value).strip()
    return " ".join(value.split()[:max_tokens])


def compile_patterns() -> dict[str, list[re.Pattern[str]]]:
    patterns: dict[str, list[re.Pattern[str]]] = {
        "Company": [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in COMPANY_TERMS],
        "Index": [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in INDEX_TERMS],
        "Location": [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in LOCATION_TERMS],
        "Commodity": [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE) for t in COMMODITY_TERMS],
        "Sector": [re.compile(rf"(?<!\w){re.escape(r)}\w*(?!\w)", re.IGNORECASE) for r in SECTOR_ROOTS],
        "Event": [],
    }
    for root in EVENT_ROOTS:
        if root.isupper():
            patterns["Event"].append(re.compile(rf"(?<!\w){re.escape(root)}(?!\w)", re.IGNORECASE))
        else:
            patterns["Event"].append(re.compile(rf"(?<!\w){re.escape(root)}\w*(?!\w)", re.IGNORECASE))
    return patterns


def extract_entities(text: str, patterns: dict[str, list[re.Pattern[str]]]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for entity_type, pats in patterns.items():
        for pat in pats:
            for m in pat.finditer(text):
                value = m.group(0).strip()
                if not value:
                    continue
                key = (value.lower(), entity_type, m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "entity": value,
                        "entity_type": entity_type,
                        "start_char": int(m.start()),
                        "end_char": int(m.end()),
                    }
                )
    return sorted(rows, key=lambda x: (x["start_char"], x["end_char"], x["entity_type"]))


def sentence_for_entity(text: str, entity: str, start_char: int) -> str:
    if not text:
        return ""
    parts = SENT_SPLIT_RE.split(text)
    cursor = 0
    for part in parts:
        begin = cursor
        end = cursor + len(part)
        if begin <= start_char <= end:
            return part.strip()
        cursor = end + 1
    for part in parts:
        if entity.lower() in part.lower():
            return part.strip()
    return text[:300].strip()


def label_sentiment(context: str) -> str:
    text = str(context or "").lower()
    has_pos = any(k in text for k in POSITIVE_KEYS)
    has_neg = any(k in text for k in NEGATIVE_KEYS)
    if has_pos and not has_neg:
        return "Positive"
    if has_neg and not has_pos:
        return "Negative"
    return "Neutral"


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = {}
            for key, value in row.items():
                payload[key] = value.isoformat() if isinstance(value, pd.Timestamp) else value
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if not args.train_corpus.exists():
        raise FileNotFoundError(f"Train corpus not found: {args.train_corpus}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_corpus).copy()
    train_df["train_exact_signature"] = train_df["text"].astype(str).str.lower().str.strip()
    train_df["train_pattern_signature"] = train_df["text"].astype(str).apply(pattern_signature)
    train_exact = set(train_df["train_exact_signature"])
    train_pattern = set(train_df["train_pattern_signature"])

    ds = load_dataset(args.dataset_id, split="train")
    df = ds.to_pandas().copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["body_clean"] = df["body"].apply(clean_text)
    df["text"] = (df["title_clean"] + ". " + df["body_clean"]).apply(clean_text)
    df["published_at"] = [parse_datetime(d, t) for d, t in zip(df["date"], df["time"])]
    if "tags" in df.columns:
        df["tags_list"] = df["tags"].apply(parse_tags)
    else:
        df["tags_list"] = [[] for _ in range(len(df))]
    df["tags_joined"] = df["tags_list"].apply(lambda x: ", ".join(x))

    before = len(df)
    df = df[df["published_at"].notna()].copy()
    df = df[df["title"].apply(is_valid_title)].copy()
    df = df[df["text"].str.strip() != ""].copy()
    df["exact_signature"] = df["text"].str.lower().str.strip()
    df["pattern_signature"] = df["text"].apply(pattern_signature)
    df = df.drop_duplicates(subset=["exact_signature"], keep="first").copy()
    df = df.drop_duplicates(subset=["pattern_signature"], keep="first").copy()

    # Remove anything that overlaps with the current train corpus.
    non_overlap = df[
        (~df["exact_signature"].isin(train_exact)) & (~df["pattern_signature"].isin(train_pattern))
    ].copy()
    non_overlap.sort_values("published_at", ascending=False, inplace=True)
    test_df = non_overlap.head(args.test_news).copy().reset_index(drop=True)
    test_df["test_news_id"] = test_df.index + 1

    test_cols = ["test_news_id", "published_at", "source", "title", "body", "text", "tags_joined"]
    test_corpus_csv = args.output_dir / f"test_corpus_clean_{args.test_news}.csv"
    test_corpus_jsonl = args.output_dir / f"test_corpus_clean_{args.test_news}.jsonl"
    test_df[test_cols].to_csv(test_corpus_csv, index=False, encoding="utf-8-sig")
    save_jsonl(test_corpus_jsonl, test_df[test_cols].to_dict(orient="records"))

    patterns = compile_patterns()

    ner_rows: list[dict] = []
    for row in test_df.itertuples(index=False):
        entities = extract_entities(str(row.text), patterns)
        if not entities:
            continue
        per_news = 0
        for ent in entities:
            ner_rows.append(
                {
                    "sample_id": len(ner_rows) + 1,
                    "test_news_id": int(row.test_news_id),
                    "published_at": row.published_at,
                    "source": row.source,
                    "title": row.title,
                    "context": sentence_for_entity(str(row.text), ent["entity"], int(ent["start_char"])),
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "start_char": ent["start_char"],
                    "end_char": ent["end_char"],
                }
            )
            per_news += 1
            if args.max_entities_per_news > 0 and per_news >= args.max_entities_per_news:
                break
            if len(ner_rows) >= args.test_ner_labels:
                break
        if len(ner_rows) >= args.test_ner_labels:
            break

    ner_df = pd.DataFrame(ner_rows)
    ner_path = args.output_dir / "test_ner_dataset.csv"
    ner_df.to_csv(ner_path, index=False, encoding="utf-8-sig")

    sent_rows: list[dict] = []
    for row in ner_rows:
        if len(sent_rows) >= args.test_sent_labels:
            break
        sent_rows.append(
            {
                "sample_id": len(sent_rows) + 1,
                "test_news_id": row["test_news_id"],
                "source": row["source"],
                "title": row["title"],
                "context": row["context"],
                "entity": row["entity"],
                "entity_type": row["entity_type"],
                "sentiment": label_sentiment(str(row["context"])),
            }
        )
    sent_df = pd.DataFrame(sent_rows)
    sent_path = args.output_dir / "test_entity_sentiment_dataset.csv"
    sent_df.to_csv(sent_path, index=False, encoding="utf-8-sig")

    # Overlap check for report.
    overlap_exact = int(test_df["exact_signature"].isin(train_exact).sum())
    overlap_pattern = int(test_df["pattern_signature"].isin(train_pattern).sum())

    report = {
        "input_rows_before_cleaning": int(before),
        "train_corpus_rows": int(len(train_df)),
        "candidate_non_overlap_rows": int(len(non_overlap)),
        "test_corpus_rows": int(len(test_df)),
        "test_ner_rows": int(len(ner_df)),
        "test_sentiment_rows": int(len(sent_df)),
        "overlap_exact_count": overlap_exact,
        "overlap_pattern_count": overlap_pattern,
        "status": "ok" if overlap_exact == 0 and overlap_pattern == 0 else "warning_overlap_detected",
    }
    report_path = args.output_dir / "test_overlap_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] Independent test set created.")
    print(f"[ok] Test corpus CSV:       {test_corpus_csv}")
    print(f"[ok] Test corpus JSONL:     {test_corpus_jsonl}")
    print(f"[ok] Test NER dataset:      {ner_path}")
    print(f"[ok] Test sentiment set:    {sent_path}")
    print(f"[ok] Overlap report:        {report_path}")
    print(f"[ok] overlap_exact_count:   {overlap_exact}")
    print(f"[ok] overlap_pattern_count: {overlap_pattern}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
