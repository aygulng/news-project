#!/usr/bin/env python3
"""Train spaCy transformer+ner model on prepared DocBin files.

Pipeline:
1) spacy init config (transformer,ner)
2) patch transformer model name in config
3) optional spacy debug data
4) spacy train

Outputs:
- data/experiments/spacy_transformer_ner/model-best
- data/experiments/spacy_transformer_ner/model-last
- data/experiments/spacy_transformer_ner/spacy_transformer_train_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in CURRENT_FILE.parents if p.name == "new_project_ru_finnews"), CURRENT_FILE.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spaCy transformer+ner for RU financial NER.")
    parser.add_argument("--train-spacy", type=Path, default=PROJECT_ROOT / "data" / "spacy" / "train.spacy")
    parser.add_argument("--dev-spacy", type=Path, default=PROJECT_ROOT / "data" / "spacy" / "dev.spacy")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "spacy_transformer_ner" / "config.cfg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments" / "spacy_transformer_ner",
    )
    parser.add_argument(
        "--hf-transformer",
        default="DeepPavlov/rubert-base-cased",
        help="HF model for spaCy-transformers encoder.",
    )
    parser.add_argument("--gpu-id", type=int, default=-1, help="Use -1 for CPU.")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-frequency", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=1600, help="Early stopping patience (in score checks).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-debug-data", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[spacy-trf] {message}", flush=True)


def run_cmd(args: list[str], verbose: bool = True) -> None:
    if verbose:
        _log("[cmd] " + " ".join(shlex.quote(x) for x in args))
    proc = subprocess.run(args, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(args)}")


def validate_transformer_config(config_path: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    has_pipeline_transformer = bool(re.search(r'(?m)^pipeline\s*=\s*\["transformer"\s*,\s*"ner"\]\s*$', text))
    has_transformer_block = "[components.transformer]" in text
    if not (has_pipeline_transformer and has_transformer_block):
        raise RuntimeError(
            "Config is not transformer-based. Expected pipeline=['transformer','ner'] and [components.transformer].\n"
            "Install/verify spacy-transformers in THIS Python environment and regenerate config.\n"
            "Check commands:\n"
            "  python -m pip show spacy spacy-transformers\n"
            "  python -c \"import spacy, spacy_transformers; print(spacy.__version__)\""
        )


def load_data_snapshot(train_spacy: Path, dev_spacy: Path) -> dict:
    summary_path = train_spacy.parent / "spacy_data_summary.json"
    if not summary_path.exists():
        return {
            "summary_path": str(summary_path),
            "train_docs": None,
            "dev_docs": None,
            "holdout_docs": None,
        }
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "summary_path": str(summary_path),
            "train_docs": None,
            "dev_docs": None,
            "holdout_docs": None,
        }
    return {
        "summary_path": str(summary_path),
        "train_docs": payload.get("docbin", {}).get("train", {}).get("docs"),
        "dev_docs": payload.get("docbin", {}).get("dev", {}).get("docs"),
        "holdout_docs": payload.get("docbin", {}).get("holdout", {}).get("docs"),
    }


def patch_config(
    config_path: Path,
    train_path: Path,
    dev_path: Path,
    transformer_name: str,
    max_steps: int,
    eval_frequency: int,
    dropout: float,
    patience: int,
    seed: int,
) -> None:
    text = config_path.read_text(encoding="utf-8")

    # Set paths section.
    text = re.sub(r"(?m)^train\s*=\s*null\s*$", f'train = "{train_path.as_posix()}"', text)
    text = re.sub(r"(?m)^dev\s*=\s*null\s*$", f'dev = "{dev_path.as_posix()}"', text)

    # Ensure transformer model name.
    text = re.sub(
        r'(?ms)(\[components\.transformer\.model\].*?^name\s*=\s*)".*?"',
        rf'\1"{transformer_name}"',
        text,
    )

    # Training controls.
    text = re.sub(r"(?m)^max_steps\s*=\s*\d+\s*$", f"max_steps = {int(max_steps)}", text)
    text = re.sub(r"(?m)^eval_frequency\s*=\s*\d+\s*$", f"eval_frequency = {int(eval_frequency)}", text)
    text = re.sub(r"(?m)^dropout\s*=\s*[-+]?\d*\.?\d+\s*$", f"dropout = {float(dropout)}", text)
    text = re.sub(r"(?m)^patience\s*=\s*\d+\s*$", f"patience = {int(patience)}", text)
    text = re.sub(r"(?m)^seed\s*=\s*\d+\s*$", f"seed = {int(seed)}", text)

    config_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.train_spacy.exists():
        raise FileNotFoundError(f"train.spacy not found: {args.train_spacy}")
    if not args.dev_spacy.exists():
        raise FileNotFoundError(f"dev.spacy not found: {args.dev_spacy}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    exp_name = args.output_dir.name
    runtime_device = "cpu" if int(args.gpu_id) < 0 else f"gpu:{int(args.gpu_id)}"
    snapshot = load_data_snapshot(args.train_spacy, args.dev_spacy)

    _log("=" * 80)
    _log("SPACY TRANSFORMER+NER: ОБУЧЕНИЕ")
    _log("=" * 80)
    _log(f"Дата (UTC): {datetime.now(timezone.utc).isoformat()}")
    _log(f"Эксперимент: {exp_name}")
    _log(f"Датасет train: {args.train_spacy}")
    _log(f"Датасет dev:   {args.dev_spacy}")
    _log(f"Запрошенное устройство: gpu-id={args.gpu_id}")
    _log(f"Фактическое устройство: {runtime_device}")
    _log(f"Конфиг: {args.config_path}")
    if snapshot.get("train_docs") is not None:
        _log(f"Сводка данных: {snapshot['summary_path']}")
        _log(f"  - train документов: {snapshot['train_docs']}")
        _log(f"  - dev документов:   {snapshot['dev_docs']}")
        _log(f"  - holdout документов: {snapshot['holdout_docs']}")

    _log("ШАГ 1: Генерация базового config...")

    # 1) Generate base config.
    # NOTE: For spaCy 3.8 CLI, transformer template for NER is selected with --gpu.
    init_cmd = [
        sys.executable,
        "-m",
        "spacy",
        "init",
        "config",
        str(args.config_path),
        "--lang",
        "ru",
        "--pipeline",
        "ner",
        "--optimize",
        "accuracy",
        "--gpu",
        "--force",
    ]
    run_cmd(init_cmd, verbose=args.verbose or True)

    # 2) Patch with project-specific settings.
    _log("ШАГ 2: Патч config (модель, анти-overfit параметры)...")
    patch_config(
        config_path=args.config_path,
        train_path=args.train_spacy,
        dev_path=args.dev_spacy,
        transformer_name=args.hf_transformer,
        max_steps=args.max_steps,
        eval_frequency=args.eval_frequency,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
    validate_transformer_config(args.config_path)
    _log(
        "  - "
        f"hf_transformer={args.hf_transformer}, max_steps={args.max_steps}, "
        f"eval_frequency={args.eval_frequency}, dropout={args.dropout}, patience={args.patience}, seed={args.seed}"
    )

    # 3) Optional debug.
    if args.run_debug_data:
        _log("ШАГ 3: Проверка данных (spacy debug data)...")
        debug_cmd = [
            sys.executable,
            "-m",
            "spacy",
            "debug",
            "data",
            str(args.config_path),
            "--paths.train",
            str(args.train_spacy),
            "--paths.dev",
            str(args.dev_spacy),
        ]
        run_cmd(debug_cmd, verbose=True)
    else:
        _log("ШАГ 3: Проверка данных пропущена (--run-debug-data не задан).")

    # 4) Train.
    _log("ШАГ 4: Запуск обучения...")
    train_cmd = [
        sys.executable,
        "-m",
        "spacy",
        "train",
        str(args.config_path),
        "--output",
        str(args.output_dir),
        "--paths.train",
        str(args.train_spacy),
        "--paths.dev",
        str(args.dev_spacy),
        "--gpu-id",
        str(args.gpu_id),
    ]
    run_cmd(train_cmd, verbose=True)
    _log("ШАГ 5: Обучение завершено.")

    model_best = args.output_dir / "model-best"
    model_last = args.output_dir / "model-last"
    best_meta_path = model_best / "meta.json"
    best_meta: dict = {}
    if best_meta_path.exists():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
        except Exception:
            best_meta = {}
    best_perf = best_meta.get("performance", {}) if isinstance(best_meta, dict) else {}
    if best_perf:
        ents_f = float(best_perf.get("ents_f", 0.0))
        ents_p = float(best_perf.get("ents_p", 0.0))
        ents_r = float(best_perf.get("ents_r", 0.0))
        _log(f"  - best ents_f={ents_f:.4f}, ents_p={ents_p:.4f}, ents_r={ents_r:.4f}")
    _log(f"  - лучший чекпоинт: {model_best}")
    _log(f"  - последний чекпоинт: {model_last}")

    summary = {
        "config_path": str(args.config_path),
        "train_spacy": str(args.train_spacy),
        "dev_spacy": str(args.dev_spacy),
        "output_dir": str(args.output_dir),
        "hf_transformer": str(args.hf_transformer),
        "gpu_id": int(args.gpu_id),
        "max_steps": int(args.max_steps),
        "eval_frequency": int(args.eval_frequency),
        "dropout": float(args.dropout),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "runtime_device": runtime_device,
        "artifacts": {
            "model_best": str(model_best),
            "model_last": str(model_last),
        },
        "best_meta": best_meta,
    }
    summary_path = args.output_dir / "spacy_transformer_train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _log("[ok] spaCy transformer+ner training completed.")
    _log(f"[ok] model-best: {model_best}")
    _log(f"[ok] summary:    {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
