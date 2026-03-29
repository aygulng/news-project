#!/usr/bin/env python3
"""Streamlit UI for NER + entity-level sentiment on Russian investment news."""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)
PATTERNS_PATH = PROJECT_ROOT / "config" / "rare_company_index_patterns.json"
TRAIN_SENTIMENT_PATH = PROJECT_ROOT / "data" / "labeled" / "entity_sentiment_dataset.csv"
RUBERT_SENTIMENT_MODEL_DIR = PROJECT_ROOT / "data" / "experiments" / "sentiment_tfidf_rubert" / "rubert_model"
HF_NER_MODEL_DIR_V2_CPU = PROJECT_ROOT / "data" / "experiments" / "hf_rubert_base_ner_v2_cpu" / "hf_model"
HF_NER_MODEL_DIR_LEGACY = PROJECT_ROOT / "data" / "experiments" / "hf_rubert_base_ner" / "hf_model"

ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}
ALLOWED_ENTITY_TYPES = {"Company", "Index", "Event", "Commodity", "Currency", "Person", "Location", "Sector", "Time"}
TYPE_RU = {
    "Company": "Компания",
    "Index": "Индекс",
    "Event": "Событие",
    "Commodity": "Сырьё",
    "Currency": "Валюта",
    "Person": "Персона",
    "Location": "Локация",
    "Sector": "Сектор",
    "Time": "Время",
}
SENTIMENT_RU = {
    "Positive": "Позитивная",
    "Neutral": "Нейтральная",
    "Negative": "Негативная",
}
NER_METHOD_RU = {
    "ruler": "EntityRuler",
    "spacy": "spaCy",
    "hf_rubert": "HF ruBERT",
    "fallback_ruler": "Fallback: EntityRuler",
    "fallback_spacy": "Fallback: spaCy",
}

POSITIVE_SENTIMENT_PATTERNS = [
    r"\bрост\w*\b",
    r"\bвырос\w*\b",
    r"\bприбав(ил|или|ила|ились|ляет|ляют|ка)\w*\b",
    r"\bподорожал\w*\b",
    r"\bукрепил\w*\b",
    r"\bпозитив\w*\b",
    r"\bрекорд\w*\b",
    r"\bулучш\w*\b",
    r"\bповыш\w*\b",
    r"\bсильн\w*\b",
    r"\bподдержива\w*\b",
]
NEGATIVE_SENTIMENT_PATTERNS = [
    r"\bпаден\w*\b",
    r"\bупал\w*\b",
    r"\bсниз(ил|илась|ились|ился|ить|ился|илась|ились)\w*\b",
    r"\bснижен\w*\b",
    r"\bобвал\w*\b",
    r"\bпросадк\w*\b",
    r"\bминус\w*\b",
    r"\bнегатив\w*\b",
    r"\bухудш\w*\b",
    r"\bубыт\w*\b",
    r"\bдавлени\w*\b",
]
POSITIVE_MOVE_PATTERN = re.compile(r"(рост|вырос\w*|прибав\w*|подорожал\w*).{0,25}\b\d{1,3}(?:[.,]\d+)?\s*%", flags=re.IGNORECASE)
NEGATIVE_MOVE_PATTERN = re.compile(r"(паден\w*|упал\w*|снизил\w*|просел\w*).{0,25}\b\d{1,3}(?:[.,]\d+)?\s*%", flags=re.IGNORECASE)


def canonical_entity_type(value: object) -> str | None:
    label = str(value or "").strip()
    label = re.sub(r"^(B-|I-)", "", label)
    if label in {"SectorIndustry", "Sector / Industry"}:
        return "Sector"
    if label in ALLOWED_ENTITY_TYPES:
        return label
    return None


def is_suspicious_pattern(entry: dict) -> bool:
    pattern = entry.get("pattern")
    if isinstance(pattern, str) and pattern.count("?") >= 3:
        return True
    return False


def clean_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_entity_char(ch: str) -> bool:
    return ch.isalnum() or ch in "._-+/&%$€£₽"


def expand_span_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    n = len(text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    while start > 0 and is_entity_char(text[start - 1]):
        start -= 1
    while end < n and is_entity_char(text[end]):
        end += 1
    return start, end


def safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


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


def make_feature_text(context: str, entity: str, entity_type: str) -> str:
    return f"{clean_text(context)} [ENTITY] {entity.strip()} [TYPE] {entity_type.strip()}".strip()


def sentence_for_entity(text: str, start: int, end: int, fallback_entity: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    n = len(text)
    if start < 0 or end <= start or end > n:
        ent = str(fallback_entity or "").strip()
        if ent:
            m = re.search(re.escape(ent), text, flags=re.IGNORECASE)
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


def split_sentences_with_offsets(text: str) -> list[tuple[int, int, str]]:
    text = clean_text(text)
    if not text:
        return []
    parts = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text, flags=re.DOTALL):
        chunk = m.group(0).strip()
        if not chunk:
            continue
        parts.append((m.start(), m.end(), chunk))
    return parts


def sentence_index_for_span(sentences: list[tuple[int, int, str]], start: int, end: int) -> int:
    for i, (s, e, _) in enumerate(sentences):
        if max(s, start) < min(e, end):
            return i
    for i, (s, e, _) in enumerate(sentences):
        if start >= s and start < e:
            return i
    return -1


def sentiment_hits(text: str, patterns: list[str]) -> int:
    lc = str(text or "").casefold()
    return sum(1 for p in patterns if re.search(p, lc, flags=re.IGNORECASE))


def should_attach_next_sentence(next_sentence: str, entity_type: str) -> bool:
    s = str(next_sentence or "").casefold()
    if not s:
        return False
    has_proxy = bool(
        re.search(
            r"\b(акции|бумаги|котировки|индекс|цены|фьючерс|контракт|инвестор\w*|компани\w*|рынок)\b",
            s,
            flags=re.IGNORECASE,
        )
    )
    if not has_proxy:
        return False
    pos_hits = sentiment_hits(s, POSITIVE_SENTIMENT_PATTERNS)
    neg_hits = sentiment_hits(s, NEGATIVE_SENTIMENT_PATTERNS)
    if entity_type in {"Company", "Index", "Commodity"} and (pos_hits > 0 or neg_hits > 0):
        return True
    return False


def build_entity_context(text: str, start: int, end: int, entity: str, entity_type: str) -> str:
    base = sentence_for_entity(text=text, start=start, end=end, fallback_entity=entity)
    sentences = split_sentences_with_offsets(text)
    if not sentences:
        return base

    idx = sentence_index_for_span(sentences, start, end)
    if idx == -1:
        return base

    context_parts = [sentences[idx][2].strip()]
    if idx + 1 < len(sentences):
        next_sent = sentences[idx + 1][2].strip()
        if should_attach_next_sentence(next_sent, entity_type):
            context_parts.append(next_sent)
    return clean_text(" ".join(context_parts))[:900]


def adjust_sentiment_with_rules(
    context: str,
    entity: str,
    entity_type: str,
    predicted: str,
    score: float,
) -> str:
    pred = normalize_sentiment(predicted)
    if pred not in ALLOWED_SENTIMENTS:
        pred = "Neutral"

    lc = clean_text(context).casefold()
    entity_lc = normalize_entity_text(entity)
    if not lc:
        return pred

    pos_hits = sentiment_hits(lc, POSITIVE_SENTIMENT_PATTERNS)
    neg_hits = sentiment_hits(lc, NEGATIVE_SENTIMENT_PATTERNS)
    strong_pos = bool(POSITIVE_MOVE_PATTERN.search(lc) or re.search(r"\bпозитив\w+\s+реакц\w*\b", lc))
    strong_neg = bool(NEGATIVE_MOVE_PATTERN.search(lc) or re.search(r"\bнегатив\w+\s+реакц\w*\b", lc))

    has_entity_or_proxy = bool(entity_lc and entity_lc in normalize_entity_text(lc)) or bool(
        re.search(r"\b(акции|бумаги|котировки|индекс|цены|компани\w*)\b", lc, flags=re.IGNORECASE)
    )
    if not has_entity_or_proxy:
        return pred

    # Conservative override: only move away from Neutral on strong lexical evidence.
    if pred == "Neutral":
        if strong_pos or (pos_hits >= 2 and pos_hits > neg_hits):
            return "Positive"
        if strong_neg or (neg_hits >= 2 and neg_hits > pos_hits):
            return "Negative"
    return pred


def normalize_entity_text(text: str) -> str:
    value = str(text or "").casefold().replace("ё", "е")
    value = value.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\W_]+|[\W_]+$", "", value)
    return value.strip()


def normalize_ru_token_for_dedupe(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    suffixes = (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ому",
        "ией",
        "ией",
        "ов",
        "ев",
        "ей",
        "ом",
        "ем",
        "ам",
        "ям",
        "ах",
        "ях",
        "ой",
        "ый",
        "ий",
        "ая",
        "яя",
        "ую",
        "юю",
        "а",
        "я",
        "ы",
        "и",
        "у",
        "е",
        "о",
        "ь",
    )
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 4:
            return token[: -len(suf)]
    return token


def normalize_entity_for_doc_dedupe(entity: str, entity_type: str) -> str:
    base = normalize_entity_text(entity)
    if not base:
        return base
    if entity_type == "Commodity":
        toks = [normalize_ru_token_for_dedupe(t) for t in base.split()]
        return " ".join(toks).strip()
    return base


def entity_method_priority(method: str) -> int:
    order = {
        "hf_rubert": 0,
        "ruler": 1,
        "spacy": 2,
        "fallback_ruler": 3,
        "fallback_spacy": 4,
    }
    return order.get(str(method), 99)


def reduce_commodity_generics(entities: list[dict]) -> list[dict]:
    if not entities:
        return entities

    norm_commodities = {
        normalize_entity_for_doc_dedupe(e.get("entity", ""), e.get("type", ""))
        for e in entities
        if e.get("type") == "Commodity"
    }
    has_specific_oil = any(x in norm_commodities for x in {"brent", "wti"})
    if not has_specific_oil:
        return entities

    out: list[dict] = []
    for e in entities:
        if e.get("type") != "Commodity":
            out.append(e)
            continue
        norm = normalize_entity_for_doc_dedupe(e.get("entity", ""), e.get("type", ""))
        if norm in {"нефт", "нефть"}:
            continue
        out.append(e)
    return out


def map_spacy_label(label: str) -> str | None:
    if label in ALLOWED_ENTITY_TYPES:
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
    return canonical_entity_type(mapping.get(label))


def resolve_hf_ner_model_dir() -> Path:
    # Prefer latest fine-tuned run, keep legacy as fallback for compatibility.
    if HF_NER_MODEL_DIR_V2_CPU.exists():
        return HF_NER_MODEL_DIR_V2_CPU
    if HF_NER_MODEL_DIR_LEGACY.exists():
        return HF_NER_MODEL_DIR_LEGACY
    # Return preferred path in error message even if it does not exist yet.
    return HF_NER_MODEL_DIR_V2_CPU


@st.cache_resource(show_spinner=False)
def load_spacy_ner_pipeline(spacy_model_name: str = "ru_core_news_sm", use_ruler: bool = True):
    import spacy

    nlp = spacy.load(spacy_model_name)
    if use_ruler:
        if not PATTERNS_PATH.exists():
            raise FileNotFoundError(f"Patterns file not found: {PATTERNS_PATH}")
        patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
        patterns = [p for p in patterns if not is_suspicious_pattern(p)]
        if "entity_ruler" in nlp.pipe_names:
            nlp.remove_pipe("entity_ruler")
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={"overwrite_ents": False, "phrase_matcher_attr": "LOWER"},
        )
        ruler.add_patterns(patterns)
    return nlp


@st.cache_resource(show_spinner=False)
def load_hf_ner_pipeline(model_dir: Path):
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Модель HF ruBERT NER не найдена: {model_dir}. "
            "Сначала запустите train_hf_rubert_ner.py и сохраните модель в этот путь."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=False,
    )
    tokenizer.model_max_length = 512
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    ner_pipe = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU
    )
    return ner_pipe


def dedupe_entities(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    seen: set[tuple[str, str, int, int]] = set()
    out: list[dict] = []
    for row in sorted(rows, key=lambda r: (r["start_char"], r["end_char"], r["type"], r["entity"].lower())):
        key = (
            normalize_entity_text(row["entity"]),
            row["type"],
            int(row["start_char"]),
            int(row["end_char"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def extract_entities_spacy(nlp, text: str) -> list[dict]:
    doc = nlp(text)
    rows: list[dict] = []
    for ent in doc.ents:
        mapped = map_spacy_label(ent.label_)
        if not mapped:
            continue
        method = "ruler" if ent.ent_id_.startswith("ruler:") else "spacy"
        rows.append(
            {
                "entity": ent.text.strip(),
                "type": mapped,
                "start_char": int(ent.start_char),
                "end_char": int(ent.end_char),
                "ner_method": method,
            }
        )
    return dedupe_entities(rows)


def extract_entities_hf(ner_pipe, text: str) -> list[dict]:
    text = clean_text(text)
    if not text:
        return []

    out = ner_pipe(text)
    rows: list[dict] = []
    for ent in out:
        raw_label = ent.get("entity_group") or ent.get("entity") or ""
        mapped = canonical_entity_type(raw_label)
        if not mapped:
            continue

        entity_text = str(ent.get("word") or "").replace("##", "").strip()
        start = safe_int(ent.get("start", -1), default=-1)
        end = safe_int(ent.get("end", -1), default=-1)
        if start < 0 or end <= start or end > len(text):
            if entity_text:
                m = re.search(re.escape(entity_text), text, flags=re.IGNORECASE)
                if m:
                    start, end = m.start(), m.end()
                else:
                    continue
            else:
                continue
        start, end = expand_span_to_word_boundaries(text, start, end)
        entity_text = text[start:end].strip(" ,.;:!?()[]{}\"'")
        if not entity_text:
            continue

        rows.append(
            {
                "entity": entity_text.strip(),
                "type": mapped,
                "start_char": int(start),
                "end_char": int(end),
                "ner_method": "hf_rubert",
            }
        )

    return dedupe_entities(rows)


@st.cache_resource(show_spinner=False)
def load_tfidf_sentiment_model(train_path: Path):
    if not train_path.exists():
        raise FileNotFoundError(f"Sentiment train file not found: {train_path}")

    df = pd.read_csv(train_path).copy()
    required = {"context", "entity", "entity_type", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {train_path}: {sorted(missing)}")

    df["context"] = df["context"].astype(str).apply(clean_text)
    df["entity"] = df["entity"].astype(str).str.strip()
    df["entity_type"] = df["entity_type"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment)
    df = df[(df["context"] != "") & (df["entity"] != "") & (df["entity_type"] != "")].copy()
    df = df[df["sentiment"].isin(ALLOWED_SENTIMENTS)].copy()
    if df.empty:
        raise ValueError("Sentiment train set is empty after filtering.")

    df["feature_text"] = df.apply(
        lambda r: make_feature_text(r["context"], r["entity"], r["entity_type"]),
        axis=1,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(df["feature_text"], df["sentiment"])
    return model


@st.cache_resource(show_spinner=False)
def load_rubert_sentiment_pipeline(model_dir: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Модель RuBERT для тональности не найдена: {model_dir}. "
            "Сначала запустите compare_entity_sentiment_tfidf_rubert.py."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    clf_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
    )
    return clf_pipe


def predict_sentiment_tfidf(model: Pipeline, context: str, entity: str, entity_type: str) -> tuple[str, float]:
    feature = make_feature_text(context, entity, entity_type)
    pred = str(model.predict([feature])[0])
    score = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([feature])[0]
        score = float(proba.max())
    return normalize_sentiment(pred), score


def predict_sentiment_rubert(clf_pipe, context: str, entity: str, entity_type: str) -> tuple[str, float]:
    feature = make_feature_text(context, entity, entity_type)
    out = clf_pipe(feature, truncation=True, max_length=192)
    if isinstance(out, list):
        out = out[0]
    label = normalize_sentiment(out.get("label", "Neutral"))
    score = float(out.get("score", 0.0))
    return label, score


def dataframe_from_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        raw = uploaded_file.getvalue()
        text = ""
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        text = clean_text(text)
        if not text:
            return pd.DataFrame(columns=["title", "text"])
        return pd.DataFrame([{"title": uploaded_file.name, "text": text}])

    if name.endswith(".csv"):
        raw = uploaded_file.getvalue()
        df = None
        last_error = None
        for enc in ("utf-8-sig", "utf-8", "cp1251", "windows-1251"):
            try:
                df = pd.read_csv(BytesIO(raw), encoding=enc, sep=None, engine="python").copy()
                break
            except Exception as exc:
                last_error = exc
                continue
        if df is None:
            raise ValueError(f"Не удалось прочитать CSV. Проверьте кодировку файла (UTF-8/CP1251). Ошибка: {last_error}")
        cols = {c.lower(): c for c in df.columns}
        title_col = None
        for c in ("title", "headline", "name"):
            if c in cols:
                title_col = cols[c]
                break

        text_col = None
        for c in ("text", "body", "content", "article", "news"):
            if c in cols:
                text_col = cols[c]
                break

        out = pd.DataFrame()
        if title_col:
            out["title"] = df[title_col].astype(str)
        else:
            out["title"] = [f"Строка из файла #{i + 1}" for i in range(len(df))]

        if text_col:
            out["text"] = df[text_col].astype(str)
        else:
            body_col = cols.get("body")
            if body_col:
                out["text"] = df[body_col].astype(str)
            else:
                # fallback: combine all columns as text
                out["text"] = df.astype(str).agg(" ".join, axis=1)

        out["title"] = out["title"].astype(str).str.strip()
        out["text"] = out["text"].astype(str).apply(clean_text)
        out = out[out["text"] != ""].copy()
        return out

    raise ValueError("Поддерживаются только файлы .txt и .csv")


def run_pipeline(
    news_df: pd.DataFrame,
    ner_model_choice: str,
    include_sentiment: bool,
    sentiment_model_choice: str | None,
    dedupe_entity_per_doc: bool = True,
    enable_ner_fallback: bool = True,
) -> pd.DataFrame:
    spacy_nlp = None
    hf_ner_pipe = None
    fallback_spacy_nlp = None

    if ner_model_choice == "hf_rubert":
        hf_ner_pipe = load_hf_ner_pipeline(resolve_hf_ner_model_dir())
    elif ner_model_choice == "spacy":
        spacy_nlp = load_spacy_ner_pipeline(spacy_model_name="ru_core_news_sm", use_ruler=False)
    else:
        spacy_nlp = load_spacy_ner_pipeline(spacy_model_name="ru_core_news_sm", use_ruler=True)
    # Do not use spaCy fallback when user selected HF ruBERT.
    # Fallback is only meaningful for plain spaCy mode.
    if enable_ner_fallback and ner_model_choice == "spacy":
        fallback_spacy_nlp = load_spacy_ner_pipeline(spacy_model_name="ru_core_news_sm", use_ruler=True)

    tfidf_model = None
    rubert_pipe = None

    if include_sentiment:
        if sentiment_model_choice == "tfidf":
            tfidf_model = load_tfidf_sentiment_model(TRAIN_SENTIMENT_PATH)
        elif sentiment_model_choice == "rubert":
            rubert_pipe = load_rubert_sentiment_pipeline(RUBERT_SENTIMENT_MODEL_DIR)
        else:
            raise ValueError("Неизвестная модель тональности.")

    rows: list[dict] = []
    for item in news_df.itertuples(index=False):
        title = str(getattr(item, "title", "")).strip()
        text = clean_text(getattr(item, "text", ""))
        if not text:
            continue

        if hf_ner_pipe is not None:
            ents = extract_entities_hf(hf_ner_pipe, text)
        else:
            ents = extract_entities_spacy(spacy_nlp, text)

        # For HF/spaCy-only modes, fallback can both rescue empty output and add missed entities.
        if fallback_spacy_nlp is not None:
            fb_ents = extract_entities_spacy(fallback_spacy_nlp, text)
            for e in fb_ents:
                if e["ner_method"] == "ruler":
                    e["ner_method"] = "fallback_ruler"
                elif e["ner_method"] == "spacy":
                    e["ner_method"] = "fallback_spacy"
            ents = dedupe_entities(ents + fb_ents)

        if dedupe_entity_per_doc:
            tmp = {}
            for e in ents:
                key = (normalize_entity_for_doc_dedupe(e["entity"], e["type"]), e["type"])
                prev = tmp.get(key)
                if prev is None or entity_method_priority(e.get("ner_method", "")) < entity_method_priority(
                    prev.get("ner_method", "")
                ):
                    tmp[key] = e
            ents = list(tmp.values())
            ents = reduce_commodity_generics(ents)

        for e in ents:
            context = ""
            sentiment = None
            score = None
            if include_sentiment:
                context = build_entity_context(
                    text=text,
                    start=int(e["start_char"]),
                    end=int(e["end_char"]),
                    entity=e["entity"],
                    entity_type=e["type"],
                )

            if include_sentiment and tfidf_model is not None:
                sentiment, score = predict_sentiment_tfidf(tfidf_model, context, e["entity"], e["type"])
            elif include_sentiment:
                sentiment, score = predict_sentiment_rubert(rubert_pipe, context, e["entity"], e["type"])
            if include_sentiment:
                sentiment = adjust_sentiment_with_rules(
                    context=context,
                    entity=e["entity"],
                    entity_type=e["type"],
                    predicted=sentiment,
                    score=score,
                )
                score = round(float(score), 4)
            else:
                sentiment = None
                score = None

            rows.append(
                {
                    "title": title,
                    "entity": e["entity"],
                    "type": e["type"],
                    "sentiment": sentiment,
                    "score": score,
                    "context": context,
                    "ner_method": e["ner_method"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["title", "entity", "type"]).reset_index(drop=True)
    return out


def main() -> None:
    st.set_page_config(page_title="Анализ новостей: сущности и тональность", page_icon="📰", layout="wide")
    st.title("Анализ инвестиционных новостей")
    st.caption("")

    with st.sidebar:
        st.subheader("Настройки")
        ner_options = {
            "spaCy ru_core_news_sm + EntityRuler (гибрид)": "spacy_ruler",
            # "spaCy ru_core_news_sm (только модель)": "spacy",
            "HF ruBERT-base NER (дообученная)": "hf_rubert",
        }
        ner_label = st.selectbox("Модель NER (сущности)", options=list(ner_options.keys()), index=0)
        ner_model_choice = ner_options[ner_label]

        mode_options = {
            "Только сущности (основной режим)": "entities_only",
            "Сущности + тональность (дополнительно)": "entities_with_sentiment",
        }
        mode_label = st.radio("Режим анализа", options=list(mode_options.keys()), index=0)
        analysis_mode = mode_options[mode_label]
        include_sentiment = analysis_mode == "entities_with_sentiment"

        sentiment_model_choice = None
        sentiment_options = {
            "RuBERT-base (дообученная)": "rubert",
            "TF-IDF + логистическая регрессия": "tfidf",
        }
        if include_sentiment:
            sentiment_label = st.selectbox("Модель тональности", options=list(sentiment_options.keys()), index=0)
            sentiment_model_choice = sentiment_options[sentiment_label]
        else:
            st.caption("")

        if ner_model_choice == "hf_rubert":
            active_model_dir = resolve_hf_ner_model_dir()
            st.caption(f"Активная NER-модель: `{active_model_dir}`")

        # Служебные настройки фиксированы (не отображаются в UI по требованию).
        dedupe_entity_per_doc = True
        enable_ner_fallback = True
        st.markdown("---")
        # if ner_model_choice == "spacy_ruler":
        #     st.write("NER: spaCy ru_core_news_sm + EntityRuler")
        # elif ner_model_choice == "spacy":
        #     st.write("NER: spaCy ru_core_news_sm")
        # else:
        #     st.write("NER: HF ruBERT-base (дообученная)")
        #     st.caption("Режим HF может быть менее стабильным на коротких текстах; включен fallback на гибрид.")
        # st.write("Типы сущностей: Company, Index, Event, Location, Commodity, Sector / Industry")

    tab_text, tab_file = st.tabs(["Вставить текст", "Загрузить файл"])
    input_df = pd.DataFrame(columns=["title", "text"])

    with tab_text:
        title = st.text_input("Заголовок новости", value="Заголовок новости")
        text = st.text_area("Текст новости", height=220, placeholder="Вставьте текст новости...")
        if st.button("Запустить анализ текста", use_container_width=True):
            input_df = pd.DataFrame([{"title": title or "Заголовок новости", "text": text or ""}])

    with tab_file:
        uploaded = st.file_uploader("Загрузите .txt или .csv", type=["txt", "csv"])
        if uploaded is not None and st.button("Запустить анализ файла", use_container_width=True):
            try:
                input_df = dataframe_from_uploaded_file(uploaded)
            except Exception as exc:
                st.error(str(exc))
                return

    if input_df.empty:
        st.info("Введите текст новости или загрузите файл, затем запустите анализ.")
        return

    spinner_text = "Выполняется NER и анализ тональности..." if include_sentiment else "Выполняется извлечение сущностей (NER)..."
    with st.spinner(spinner_text):
        try:
            result = run_pipeline(
                news_df=input_df,
                ner_model_choice=ner_model_choice,
                include_sentiment=include_sentiment,
                sentiment_model_choice=sentiment_model_choice,
                dedupe_entity_per_doc=dedupe_entity_per_doc,
                enable_ner_fallback=enable_ner_fallback,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    if result.empty:
        st.warning("Сущности в тексте не найдены.")
        return

    st.success(f"Готово. Найдено упоминаний сущностей: {len(result)}.")
    st.subheader("Таблица результатов")
    base_columns = ["title", "entity", "type"]
    show_columns = base_columns + (["sentiment"] if include_sentiment else [])
    ui_df = result[show_columns].copy()
    ui_df["type"] = ui_df["type"].map(lambda x: TYPE_RU.get(str(x), str(x)))
    if include_sentiment:
        ui_df["sentiment"] = ui_df["sentiment"].map(lambda x: SENTIMENT_RU.get(str(x), str(x)))
    ui_df = ui_df.rename(
        columns={
            "title": "Заголовок",
            "entity": "Сущность",
            "type": "Тип",
            "sentiment": "Тональность",
        }
    )
    st.dataframe(ui_df, use_container_width=True)

    with st.expander("Расширенный просмотр"):
        ext_df = result.copy()
        if not include_sentiment:
            ext_df = ext_df.drop(columns=["sentiment", "score", "context"], errors="ignore")
        ext_df = ext_df.rename(
            columns={
                "title": "Заголовок",
                "entity": "Сущность",
                "type": "Тип",
                "sentiment": "Тональность",
                "score": "Уверенность",
                "context": "Контекст",
                "ner_method": "Метод NER",
            }
        )
        ext_df["Тип"] = ext_df["Тип"].map(lambda x: TYPE_RU.get(str(x), str(x)))
        if "Тональность" in ext_df.columns:
            ext_df["Тональность"] = ext_df["Тональность"].map(lambda x: SENTIMENT_RU.get(str(x), str(x)))
        ext_df["Метод NER"] = ext_df["Метод NER"].map(lambda x: NER_METHOD_RU.get(str(x), str(x)))
        st.dataframe(ext_df, use_container_width=True)

    csv_columns = ["title", "entity", "type"] + (["sentiment"] if include_sentiment else [])
    csv_bytes = result[csv_columns].to_csv(index=False).encode("utf-8-sig")
    csv_name = "news_entities_sentiment_result.csv" if include_sentiment else "news_entities_result.csv"
    csv_label = "Скачать CSV (title, entity, type, sentiment)" if include_sentiment else "Скачать CSV (title, entity, type)"
    st.download_button(
        csv_label,
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
