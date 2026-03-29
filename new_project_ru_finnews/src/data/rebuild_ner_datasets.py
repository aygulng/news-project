#!/usr/bin/env python3
"""Rebuild NER labeled datasets in strict schema (without auxiliary label-source columns).

Output columns:
sample_id,news_id,published_at,source,title,context,entity,entity_type,start_char,end_char
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)

TARGET_ENTITY_TYPES = {
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

SPACY_LABEL_MAP = {
    "ORG": "Company",
    "PERSON": "Person",
    "PER": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "EVENT": "Event",
    "MONEY": "Currency",
    "DATE": "Time",
    "TIME": "Time",
    # backward compatibility
    "Sector / Industry": "Sector",
    "SectorIndustry": "Sector",
}

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
SPACES_RE = re.compile(r"\s+")
NOISE_COMPANY = {"reuters", "bloomberg", "интерфакс", "тасс", "риа новости", "известия"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild NER train/test datasets in new schema.")
    parser.add_argument(
        "--train-corpus-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "russian_financial_news_clean_1500.csv",
    )
    parser.add_argument(
        "--test-corpus-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_corpus_clean_300.csv",
    )
    parser.add_argument(
        "--patterns-json",
        type=Path,
        default=PROJECT_ROOT / "config" / "rare_company_index_patterns.json",
    )
    parser.add_argument(
        "--train-output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "ner_entity_dataset.csv",
    )
    parser.add_argument(
        "--test-output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "test" / "test_ner_dataset.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ner_rebuild_summary.json",
    )
    parser.add_argument("--max-news", type=int, default=0, help="Debug limit for first N records.")
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00A0", " ")
    text = SPACES_RE.sub(" ", text)
    return text.strip()


def normalize_text(value: str) -> str:
    text = str(value or "").casefold().replace("ё", "е")
    text = text.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    text = SPACES_RE.sub(" ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text.strip()


def canonical_entity_type(label: object) -> str | None:
    raw = str(label or "").strip()
    raw = re.sub(r"^(B-|I-)", "", raw)
    if raw in TARGET_ENTITY_TYPES:
        return raw
    mapped = SPACY_LABEL_MAP.get(raw)
    if mapped in TARGET_ENTITY_TYPES:
        return mapped
    return None


def split_sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for m in SENTENCE_RE.finditer(text):
        raw = m.group(0)
        if not raw:
            continue
        ltrim = len(raw) - len(raw.lstrip())
        rtrim = len(raw) - len(raw.rstrip())
        s_start = m.start() + ltrim
        s_end = m.end() - rtrim
        if s_end <= s_start:
            continue
        sent = text[s_start:s_end]
        rows.append((s_start, s_end, sent))
    return rows


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start] in " \t\n\r\"'«»()[]{}.,;:!?-–—":
        start += 1
    while end > start and text[end - 1] in " \t\n\r\"'«»()[]{}.,;:!?-–—":
        end -= 1
    return start, end


def context_for_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    sentences = split_sentences_with_offsets(text)
    if not sentences:
        frag = text[max(0, start - 140) : min(len(text), end + 140)].strip()
        rel_s = max(0, start - max(0, start - 140))
        rel_e = min(len(frag), rel_s + max(0, end - start))
        return frag, rel_s, rel_e

    for s_start, s_end, sent in sentences:
        if max(start, s_start) < min(end, s_end):
            rel_s = max(0, start - s_start)
            rel_e = min(len(sent), rel_s + max(0, end - start))
            return sent, rel_s, rel_e

    # Fallback: nearest sentence by start distance.
    s_start, s_end, sent = min(sentences, key=lambda x: abs(start - x[0]))
    rel_s = max(0, min(len(sent), start - s_start))
    rel_e = max(rel_s, min(len(sent), end - s_start))
    return sent, rel_s, rel_e


def _compile_fallback_patterns() -> dict[str, list[re.Pattern[str]]]:
    return {
        "Currency": [
            re.compile(r"(?<!\w)(?:USD|EUR|CNY|RUB|JPY|GBP|CHF|HKD|KZT|TRY)(?!\w)", re.IGNORECASE),
            re.compile(r"(?<!\w)(?:доллар(?:а|ов)?|евро|юан(?:ь|я|ей)?|руб(?:ль|ля|лей|\.?)|тенге|лира)(?!\w)", re.IGNORECASE),
            re.compile(r"[$€£¥₽]"),
        ],
        "Time": [
            re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)"),
            re.compile(r"(?<!\w)(?:[1-4]|I{1,3}V?)\s*[-–]?\s*(?:квартал(?:а|е|е)?|кв\.?)(?:\s+(?:19|20)\d{2})?(?!\w)", re.IGNORECASE),
            re.compile(
                r"(?<!\w)(?:январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]|июл[ьяе]|август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])\s+(?:19|20)\d{2}(?!\w)",
                re.IGNORECASE,
            ),
            re.compile(r"(?<!\w)(?:сегодня|завтра|вчера|на следующей неделе|в этом квартале|в следующем квартале)(?!\w)", re.IGNORECASE),
        ],
        "Sector": [
            re.compile(r"(?<!\w)(?:IT|финансов\w*|банковск\w*|нефтегаз\w*|металлург\w*|энергет\w*|телеком\w*|ритейл\w*|технолог\w*|девелопер\w*|аграр\w*)(?!\w)", re.IGNORECASE),
        ],
        "Event": [
            re.compile(r"(?<!\w)(?:дивиденд\w*|отчетност\w*|санкц\w*|дефолт\w*|банкрот\w*|IPO|SPO|эмисси\w*|байб[еэ]к\w*|buyback|ставк\w*)(?!\w)", re.IGNORECASE),
        ],
    }


def extract_entities(nlp, text: str, fallback_patterns: dict[str, list[re.Pattern[str]]]) -> list[dict]:
    rows: list[dict] = []

    doc = nlp(text)
    for ent in doc.ents:
        entity_type = canonical_entity_type(ent.label_)
        if not entity_type:
            continue
        s, e = trim_span(text, int(ent.start_char), int(ent.end_char))
        if e <= s:
            continue
        rows.append(
            {
                "entity": text[s:e].strip(),
                "entity_type": entity_type,
                "abs_start": s,
                "abs_end": e,
            }
        )

    for entity_type, patterns in fallback_patterns.items():
        for patt in patterns:
            for m in patt.finditer(text):
                s, e = trim_span(text, int(m.start()), int(m.end()))
                if e <= s:
                    continue
                value = text[s:e].strip()
                if not value:
                    continue
                rows.append(
                    {
                        "entity": value,
                        "entity_type": entity_type,
                        "abs_start": s,
                        "abs_end": e,
                    }
                )

    # dedupe + cleanup
    cleaned: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    for row in sorted(rows, key=lambda x: (x["abs_start"], x["abs_end"], x["entity_type"])):
        entity = str(row["entity"]).strip()
        if not entity:
            continue
        if len(entity) < 2 and entity not in {"$", "€", "£", "¥", "₽"}:
            continue
        norm = normalize_text(entity)
        if not norm and entity not in {"$", "€", "£", "¥", "₽"}:
            continue
        if row["entity_type"] == "Company" and norm in NOISE_COMPANY:
            continue
        if row["entity_type"] == "Person":
            parts = [p for p in entity.split() if p]
            if len(parts) == 1:
                token = parts[0]
                if not re.search(r"(ов|ев|ин|ын|ский|цкий|енко|дзе|ук|ич)$", token, flags=re.IGNORECASE):
                    # keep common one-token public figures
                    if normalize_text(token) not in {"путин", "маск", "потанин", "набиуллина", "трамп", "байден"}:
                        continue
            if re.search(r"(банк|холдинг|group|capital|company|corp|inc|ltd|пao|ao|ооо)", norm, flags=re.IGNORECASE):
                continue
        key = (norm or entity, row["entity_type"], int(row["abs_start"]), int(row["abs_end"]))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned


def build_dataset(nlp, corpus_path: Path, id_col: str, fallback_patterns: dict[str, list[re.Pattern[str]]], max_news: int = 0) -> pd.DataFrame:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    df = pd.read_csv(corpus_path).copy()
    required = {id_col, "published_at", "source", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {corpus_path}: {sorted(missing)}")

    if max_news > 0:
        df = df.head(max_news).copy()

    records: list[dict] = []
    sample_id = 1

    for row in df.itertuples(index=False):
        news_id = int(getattr(row, id_col))
        published_at = str(getattr(row, "published_at", "")).strip()
        source = str(getattr(row, "source", "")).strip()
        title = str(getattr(row, "title", "")).strip()
        text = clean_text(getattr(row, "text", ""))
        if not text:
            continue

        entities = extract_entities(nlp=nlp, text=text, fallback_patterns=fallback_patterns)
        for ent in entities:
            context, rel_start, rel_end = context_for_span(text, int(ent["abs_start"]), int(ent["abs_end"]))
            entity_in_context = context[rel_start:rel_end].strip()
            if not entity_in_context:
                entity_in_context = ent["entity"]

            records.append(
                {
                    "sample_id": sample_id,
                    "news_id": news_id,
                    "published_at": published_at,
                    "source": source,
                    "title": title,
                    "context": context,
                    "entity": entity_in_context,
                    "entity_type": ent["entity_type"],
                    "start_char": int(rel_start),
                    "end_char": int(rel_end),
                }
            )
            sample_id += 1

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out = out.sort_values(["news_id", "sample_id"]).reset_index(drop=True)
    out["sample_id"] = range(1, len(out) + 1)
    return out


def load_nlp_with_ruler(patterns_path: Path):
    import spacy

    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

    nlp = spacy.load("ru_core_news_sm")
    patterns = json.loads(patterns_path.read_text(encoding="utf-8"))

    # Canonicalize labels in patterns for new schema.
    for item in patterns:
        raw = str(item.get("label", "")).strip()
        canonical = canonical_entity_type(raw) or raw
        if canonical == "Sector / Industry":
            canonical = "Sector"
        item["label"] = canonical
        item["id"] = f"ruler:{canonical}"

    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"},
    )
    ruler.add_patterns(patterns)
    return nlp


def main() -> int:
    args = parse_args()
    args.train_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.test_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    nlp = load_nlp_with_ruler(args.patterns_json)
    fallback_patterns = _compile_fallback_patterns()

    train_df = build_dataset(
        nlp=nlp,
        corpus_path=args.train_corpus_csv,
        id_col="news_id",
        fallback_patterns=fallback_patterns,
        max_news=args.max_news,
    )
    test_df = build_dataset(
        nlp=nlp,
        corpus_path=args.test_corpus_csv,
        id_col="test_news_id",
        fallback_patterns=fallback_patterns,
        max_news=args.max_news,
    )

    train_df.to_csv(args.train_output_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(args.test_output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_type_distribution": train_df["entity_type"].value_counts().to_dict() if not train_df.empty else {},
        "test_type_distribution": test_df["entity_type"].value_counts().to_dict() if not test_df.empty else {},
        "schema": [
            "sample_id",
            "news_id",
            "published_at",
            "source",
            "title",
            "context",
            "entity",
            "entity_type",
            "start_char",
            "end_char",
        ],
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] train: {args.train_output_csv} ({len(train_df)} rows)")
    print(f"[ok] test:  {args.test_output_csv} ({len(test_df)} rows)")
    print(f"[ok] summary: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
