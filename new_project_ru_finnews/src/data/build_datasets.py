#!/usr/bin/env python3
"""Build a new clean corpus and auto-annotated NER/sentiment datasets.

Project outputs:
1) Clean main corpus (1000-2000 news, no empty/no-title text fields)
2) Weak-labeled NER dataset:
   context | entity | entity_type
3) Weak-labeled entity-sentiment dataset:
   context | entity | sentiment
"""

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

LOCATION_TERMS = [
    "Россия",
    "Москва",
    "США",
    "Европа",
    "Китай",
    "Индия",
    "ОПЕК",
    "ЕС",
    "АТР",
]

COMMODITY_TERMS = [
    "нефть",
    "Нефть",
    "Brent",
    "WTI",
    "газ",
    "золото",
    "серебро",
    "платина",
    "палладий",
    "уголь",
]

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

# Match roots to capture inflections.
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

POSITIVE_KEYS = [
    "рост",
    "вырос",
    "выросл",
    "повыс",
    "прибыл",
    "прибыль",
    "рекорд",
    "улучш",
    "одобр",
    "укреп",
    "плюс",
]

NEGATIVE_KEYS = [
    "пад",
    "сниж",
    "просел",
    "убыт",
    "дефолт",
    "банкрот",
    "риск",
    "обвал",
    "минус",
    "санкц",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build new project datasets (clean corpus + NER + sentiment).")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--main-news", type=int, default=1500, help="Main corpus size (recommended 1000-2000).")
    parser.add_argument("--ner-labels", type=int, default=1200, help="Target NER rows.")
    parser.add_argument("--sent-labels", type=int, default=1200, help="Target sentiment rows.")
    parser.add_argument("--max-entities-per-news", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data",
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
            obj = ast.literal_eval(text)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
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
    tokens = value.split()[:max_tokens]
    return " ".join(tokens)


def compile_patterns() -> dict[str, list[re.Pattern[str]]]:
    out: dict[str, list[re.Pattern[str]]] = {
        "Company": [re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE) for term in COMPANY_TERMS],
        "Index": [re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE) for term in INDEX_TERMS],
        "Location": [re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE) for term in LOCATION_TERMS],
        "Commodity": [re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE) for term in COMMODITY_TERMS],
        "Sector": [],
        "Event": [],
    }
    for root in EVENT_ROOTS:
        if root.isupper():
            out["Event"].append(re.compile(rf"(?<!\w){re.escape(root)}(?!\w)", re.IGNORECASE))
        else:
            out["Event"].append(re.compile(rf"(?<!\w){re.escape(root)}\w*(?!\w)", re.IGNORECASE))
    for root in SECTOR_ROOTS:
        out["Sector"].append(re.compile(rf"(?<!\w){re.escape(root)}\w*(?!\w)", re.IGNORECASE))
    return out


def extract_entities(text: str, patterns: dict[str, list[re.Pattern[str]]]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for entity_type, pats in patterns.items():
        for pattern in pats:
            for m in pattern.finditer(text):
                entity = m.group(0).strip()
                if not entity:
                    continue
                key = (entity.lower(), entity_type, m.start(), m.end())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "entity": entity,
                        "entity_type": entity_type,
                        "start_char": int(m.start()),
                        "end_char": int(m.end()),
                    }
                )
    return sorted(candidates, key=lambda r: (r["start_char"], r["end_char"], r["entity_type"]))


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
    lowered = entity.lower()
    for part in parts:
        if lowered in part.lower():
            return part.strip()
    return text[:300].strip()


def label_sentiment(context: str) -> str:
    text = str(context or "").lower()
    has_neg = any(k in text for k in NEGATIVE_KEYS)
    has_pos = any(k in text for k in POSITIVE_KEYS)
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
                if isinstance(value, pd.Timestamp):
                    payload[key] = value.isoformat()
                else:
                    payload[key] = value
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.main_news < 1000 or args.main_news > 2000:
        raise ValueError("--main-news must be in [1000, 2000].")

    raw_dir = args.output_root / "raw"
    labeled_dir = args.output_root / "labeled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id, split="train")
    df = ds.to_pandas()
    print(f"[info] Loaded rows: {len(df)}")

    needed = {"title", "body", "date", "time", "source"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df["body_clean"] = df["body"].apply(clean_text)
    df["text"] = (df["title_clean"] + ". " + df["body_clean"]).apply(clean_text)
    df["published_at"] = [parse_datetime(d, t) for d, t in zip(df["date"], df["time"])]
    if "tags" in df.columns:
        df["tags_list"] = df["tags"].apply(parse_tags)
    else:
        df["tags_list"] = [[] for _ in range(len(df))]

    before = len(df)
    df = df[df["published_at"].notna()].copy()
    after_dt = len(df)
    df = df[df["title"].apply(is_valid_title)].copy()
    after_title = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    after_text = len(df)

    df["exact_signature"] = df["text"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["exact_signature"], keep="first").copy()
    after_exact = len(df)
    df["pattern_signature"] = df["text"].apply(pattern_signature)
    df = df.drop_duplicates(subset=["pattern_signature"], keep="first").copy()
    after_pattern = len(df)

    df.sort_values("published_at", ascending=False, inplace=True)
    main_df = df.head(args.main_news).copy().reset_index(drop=True)
    main_df["news_id"] = main_df.index + 1
    main_df["tags_joined"] = main_df["tags_list"].apply(lambda x: ", ".join(x))

    main_cols = ["news_id", "published_at", "source", "title", "body", "text", "tags_joined"]
    corpus_csv = raw_dir / f"russian_financial_news_clean_{args.main_news}.csv"
    corpus_jsonl = raw_dir / f"russian_financial_news_clean_{args.main_news}.jsonl"
    main_df[main_cols].to_csv(corpus_csv, index=False, encoding="utf-8-sig")
    save_jsonl(corpus_jsonl, main_df[main_cols].to_dict(orient="records"))

    patterns = compile_patterns()

    ner_rows: list[dict] = []
    for row in main_df.itertuples(index=False):
        entities = extract_entities(str(row.text), patterns=patterns)
        if not entities:
            continue
        count_for_news = 0
        for ent in entities:
            context = sentence_for_entity(str(row.text), ent["entity"], int(ent["start_char"]))
            ner_rows.append(
                {
                    "sample_id": len(ner_rows) + 1,
                    "news_id": int(row.news_id),
                    "published_at": row.published_at,
                    "source": row.source,
                    "title": row.title,
                    "context": context,
                    "entity": ent["entity"],
                    "entity_type": ent["entity_type"],
                    "start_char": ent["start_char"],
                    "end_char": ent["end_char"],
                }
            )
            count_for_news += 1
            if args.max_entities_per_news > 0 and count_for_news >= args.max_entities_per_news:
                break
            if len(ner_rows) >= args.ner_labels:
                break
        if len(ner_rows) >= args.ner_labels:
            break

    ner_df = pd.DataFrame(ner_rows)
    ner_path = labeled_dir / "ner_entity_dataset.csv"
    ner_df.to_csv(ner_path, index=False, encoding="utf-8-sig")

    sent_rows: list[dict] = []
    for row in ner_rows:
        if len(sent_rows) >= args.sent_labels:
            break
        sentiment = label_sentiment(str(row["context"]))
        sent_rows.append(
            {
                "sample_id": len(sent_rows) + 1,
                "news_id": row["news_id"],
                "source": row["source"],
                "title": row["title"],
                "context": row["context"],
                "entity": row["entity"],
                "entity_type": row["entity_type"],
                "sentiment": sentiment,
            }
        )

    sent_df = pd.DataFrame(sent_rows)
    sent_path = labeled_dir / "entity_sentiment_dataset.csv"
    sent_df.to_csv(sent_path, index=False, encoding="utf-8-sig")

    print("\n[ok] New project datasets created.")
    print(f"[ok] Raw corpus CSV:       {corpus_csv}")
    print(f"[ok] Raw corpus JSONL:     {corpus_jsonl}")
    print(f"[ok] NER dataset:          {ner_path}")
    print(f"[ok] Sentiment dataset:    {sent_path}")
    print(f"[ok] Rows before filters:  {before}")
    print(f"[ok] After datetime:       {after_dt}")
    print(f"[ok] After title filter:   {after_title}")
    print(f"[ok] After text filter:    {after_text}")
    print(f"[ok] After exact dedupe:   {after_exact}")
    print(f"[ok] After pattern dedupe: {after_pattern}")
    print(f"[ok] Main corpus rows:     {len(main_df)}")
    print(f"[ok] NER rows:             {len(ner_df)}")
    print(f"[ok] Sentiment rows:       {len(sent_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
